"""Trovare gli oggetti in un'immagine: cosa c'e', dove, di che colore.

E' il passo in cui il pixel smette di essere noto e va **cercato**. Finche' si
deproietta da un pixel calcolato dalla posizione vera si verifica la geometria;
qui si verifica la **percezione**, che e' un'altra capacita' e ha un altro
transfer (D18).

**Si segmenta sulla depth, non sul colore.** Le zone di posa sono verde e
arancione: un filtro "trova il verde" catturerebbe anche la zona verde, che non
e' un oggetto. La depth non si lascia ingannare — la zona e' *disegnata sul*
tavolo (1 mm), un cubo **sporge** di 40.

    la depth dice **cosa sporge**, il colore dice **cos'e'**

E' la separazione di D18: la geometria trasferisce all'85-95%, il colore al
50-70%. Mescolarli in un criterio solo renderebbe un fallimento non
attribuibile — la stessa ragione dei contatori separati del nastro (D17).

**Si guarda solo dentro un volume dichiarato.** Sporgere dal piano non basta:
il **braccio del robot** sporge, e viene rilevato come un oggetto da 16.538
pixel contro i ~900 di un cubo. Il filtro non e' una soglia di area — sarebbe un
numero magico tarato su questi cubi, che si rompe col primo oggetto piu' grande,
come ``ARM_REACH`` ricavato dalla dimensione del tavolo (D17).

Il filtro e' **geometrico**: si considerano solo i punti dentro il volume in cui
il sistema dichiara di aspettarsi oggetti. Non usa informazione privilegiata —
solo coordinate di punti gia' deproiettati — quindi **lo stesso codice gira su
una camera vera**. Ed e' cio' che fa l'industria: in una cella si dichiara il
volume di lavoro e il resto si ignora.

*Limite da sapere:* cio' che esce dal volume **smette di esistere** per la
percezione. Un cubo che rotola oltre il bordo non viene visto. E' accettabile
solo perche' e' scritto.

**Il volume da solo non basta, e la ragione non era prevedibile.** Misurato: col
volume il braccio scende da 16.538 a 5.778 pixel, e **resta**. La pinza a riposo
sta a 211 mm sopra il tavolo, dentro i bordi — cioe' **esattamente dove stanno i
cubi**, perche' e' li' che lavora. Nessun volume puo' contenere i cubi ed
escludere il braccio: restringerlo non e' un tentativo da fare, e' un tentativo
impossibile, come lo scivolo del cestino in D17.

Serve quindi una seconda domanda, diversa dalla prima:

    il **volume** chiede "questo punto e' dove mi aspetto oggetti?"
    il **self-filter** chiede "questo punto e' una parte di me?"

``ignore_mask`` risponde alla seconda. Si riceve **gia' pronta** invece di
calcolarla qui, e non e' un dettaglio: cosi' il modulo non sa da dove venga. In
simulazione la produce il renderer (``segmentation_mask`` in ``sim/render.py``),
sul robot vero si proietta il modello del braccio nell'immagine partendo dalla
cinematica, che e' nota. **L'informazione esiste in entrambi i mondi**, cambia
solo chi la calcola — ed e' la tecnica standard, il *robot self-filtering*.

**Un bordo produce misure di distanza a cui non c'e' niente.** Dove un oggetto
vicino confina con uno lontano, un pixel copre parte dell'uno e parte
dell'altro, e la depth che ne esce e' una via di mezzo: un punto **sospeso nel
vuoto** fra i due. Si chiamano *flying pixel*, e non sono un artefatto della
simulazione — ogni camera di profondita' li ha, e sulle camere vere sono
peggiori.

Misurato sul bordo fra tavolo e piedistallo del robot: una macchia da 808 pixel,
etichettata ``table_visual`` dalla segmentazione (quindi **non** filtrabile come
robot), dentro il volume di lavoro (quindi non filtrabile come fuori posto), e
alta **239 mm** sopra un piano che e' piatto.

Li prende ``max_depth_span``, e il criterio non e' una soglia arbitraria ma una
**coerenza**: un oggetto solido sta dentro la propria dimensione. I sei cubi
occupano 30 mm di quota ciascuno; il fantasma ne occupa 239. Un oggetto piu'
alto che largo, di molto, non e' un oggetto: e' un bordo.

**Il centro percepito non e' il centro dell'oggetto.** Una camera vede le
superfici: la depth misura la faccia rivolta all'obiettivo. Misurato sulla
scena di smistamento: 23-28 mm di scarto, crescente con l'angolo di vista, cioe'
mezzo cubo. ``estimate_center`` compensa lungo la direzione di vista, e la
compensazione si **misura** (vedi ``half_extent``) invece di assumerla.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from robotica.perception.geometry import deproject
from robotica.transforms import Pose

__all__ = ["Detection", "detect_objects", "classify_color"]

# Soglia in metri sopra il piano: piu' del rumore della depth e dello spessore
# dei marker delle zone (1 mm), meno dell'altezza di un cubo (40 mm).
MIN_HEIGHT = 0.010

# Pixel: sotto questa dimensione una macchia e' rumore, non un oggetto.
MIN_AREA_PX = 30

# Quota massima del volume di lavoro, sopra il piano. Piu' alta di un cubo
# (40 mm) con margine per una pila di due; piu' bassa del braccio che transita.
MAX_HEIGHT = 0.30

# Quanto puo' essere alto un oggetto rispetto a quanto e' largo, prima di
# smettere di essere credibile. 3.0 lascia passare oggetti allungati in verticale
# e scarta i bordi, che sono **archi** fra due superfici a distanze diverse.
MAX_DEPTH_SPAN_RATIO = 3.0

# I colori nominabili, in RGB 0-1. Sono gli stessi di ``sim/env.py``, ma qui
# servono come **riferimenti da confrontare**, non come verita': la camera
# misura un colore illuminato e ombreggiato, che non coincide col valore
# dichiarato nel modello.
COLOR_REFS: dict[str, tuple[float, float, float]] = {
    "rosso": (0.85, 0.2, 0.2),
    "blu": (0.2, 0.35, 0.85),
    "verde": (0.2, 0.7, 0.3),
    "giallo": (0.9, 0.75, 0.15),
}


@dataclass(frozen=True)
class Detection:
    """Un oggetto visto dalla camera.

    Attributes:
        center: punto 3D nel mondo, **compensato** di mezzo oggetto lungo la
            direzione di vista (vedi il modulo).
        surface: il punto sulla faccia vista, non compensato. E' cio' che la
            camera misura davvero, e si tiene separato perche' un giorno la
            compensazione potrebbe essere sbagliata e serve poterlo dire.
        color: il colore nominato, o ``None`` se nessun riferimento e'
            abbastanza vicino.
        color_distance: quanto dista il colore misurato dal riferimento scelto.
            **E' l'incertezza**: un fatto percepito non e' un fatto noto, e
            ``safety`` deve poterlo sapere.
        pixel: ``(u, v)`` del centroide nell'immagine.
        area_px: quanti pixel occupa. Distingue un cubo da un riflesso.
        half_extent: mezza dimensione stimata in metri, dalla larghezza
            apparente. Serve alla compensazione e a sapere se un oggetto entra
            nella pinza.
    """

    center: np.ndarray
    surface: np.ndarray
    color: str | None
    color_distance: float
    pixel: tuple[float, float]
    area_px: int
    half_extent: float


def classify_color(rgb: np.ndarray) -> tuple[str | None, float]:
    """Il nome di colore piu' vicino, e **quanto** e' vicino.

    Args:
        rgb: colore medio osservato, in 0-1.

    Returns:
        ``(nome, distanza)``. Il nome e' ``None`` se il piu' vicino dista piu'
        di 0.45: meglio dire "non lo so" che nominare a caso.

    Si normalizza la luminosita' prima di confrontare: una faccia in ombra e una
    illuminata hanno lo stesso **colore** e luminosita' diverse, e senza
    normalizzare un cubo rosso in ombra somiglia piu' al nero che al rosso. E'
    la prima cosa che si rompe passando al reale, dove l'illuminazione non e'
    uniforme.
    """
    v = np.asarray(rgb, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm < 1e-6:
        return None, float("inf")
    v = v / norm

    migliore, dist_migliore = None, float("inf")
    for nome, ref in COLOR_REFS.items():
        r = np.asarray(ref, dtype=float)
        r = r / np.linalg.norm(r)
        dist = float(np.linalg.norm(v - r))
        if dist < dist_migliore:
            migliore, dist_migliore = nome, dist

    return (migliore, dist_migliore) if dist_migliore <= 0.45 else (None, dist_migliore)


def detect_objects(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_world_camera: Pose,
    plane_z: float,
    min_height: float = MIN_HEIGHT,
    min_area_px: int = MIN_AREA_PX,
    xy_bounds: tuple[float, float, float, float] | None = None,
    max_height: float = MAX_HEIGHT,
    ignore_mask: np.ndarray | None = None,
    max_depth_span_ratio: float = MAX_DEPTH_SPAN_RATIO,
) -> list[Detection]:
    """Gli oggetti che sporgono dal piano, con posizione e colore.

    Args:
        rgb: immagine ``(H, W, 3)`` uint8.
        depth: mappa di profondita' ``(H, W)`` in metri.
        plane_z: quota del piano di appoggio nel mondo (il tavolo). **Si legge
            dal modello, non si deriva** — e' la quota che in Fase 0 e' costata
            25 mm e ogni presa.
        min_height: quanto deve sporgere un pixel per contare come oggetto.
        xy_bounds: ``(x_min, x_max, y_min, y_max)`` del volume di lavoro nel
            mondo. ``None`` significa nessun limite orizzontale, e allora il
            braccio del robot **verra' rilevato**: si passa ``None`` solo per
            dimostrarlo.
        max_height: quota massima sopra il piano. Riduce il braccio ma **non lo
            elimina**: la pinza a riposo sta a 211 mm, dentro il volume.
        ignore_mask: maschera booleana ``(H, W)``, ``True`` sui pixel da
            ignorare — tipicamente il robot stesso. Senza, il braccio viene
            rilevato come un oggetto.
        max_depth_span_ratio: quanto una macchia puo' estendersi in quota
            rispetto alla propria larghezza prima di essere scartata come
            bordo. Vedi i *flying pixel* nella descrizione del modulo.

    Returns:
        Le detection ordinate per area decrescente.

    Nessun ordinamento per colore o posizione: l'ordine non deve suggerire una
    priorita' che la percezione non ha titolo per decidere. **Che farne lo
    decide il planning.**
    """
    h, w = depth.shape
    if rgb.shape[:2] != (h, w):
        raise ValueError(f"rgb {rgb.shape[:2]} e depth {(h, w)} non coincidono")

    # 1. Ogni pixel -> la sua quota nel mondo. Vettoriale: su 2 core un ciclo
    #    su 307.200 pixel costa secondi (D17: il budget e' gia' stretto).
    vv, uu = np.mgrid[0:h, 0:w]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    valida = np.isfinite(depth) & (depth > 0) & (depth < 50)
    d = np.where(valida, depth, 0.0)

    T = np.asarray(T_world_camera, dtype=float)
    x_c = (uu - cx) / fx * d
    y_c = -(vv - cy) / fy * d
    z_c = -d
    z_world = T[2, 0] * x_c + T[2, 1] * y_c + T[2, 2] * z_c + T[2, 3]

    dentro = valida & (z_world > plane_z + min_height)
    dentro &= z_world < plane_z + max_height

    if xy_bounds is not None:
        x_min, x_max, y_min, y_max = xy_bounds
        x_world = T[0, 0] * x_c + T[0, 1] * y_c + T[0, 2] * z_c + T[0, 3]
        y_world = T[1, 0] * x_c + T[1, 1] * y_c + T[1, 2] * z_c + T[1, 3]
        dentro &= (x_world > x_min) & (x_world < x_max)
        dentro &= (y_world > y_min) & (y_world < y_max)

    if ignore_mask is not None:
        if ignore_mask.shape != (h, w):
            raise ValueError(
                f"ignore_mask {ignore_mask.shape} non coincide con depth {(h, w)}"
            )
        dentro &= ~ignore_mask.astype(bool)

    sporge = dentro

    # 2. Pixel contigui -> una macchia per oggetto. Lo fa OpenCV (D5).
    n, etichette, stats, centroidi = cv2.connectedComponentsWithStats(
        sporge.astype(np.uint8), connectivity=8
    )

    rgb01 = rgb.astype(float) / 255.0
    out: list[Detection] = []

    for i in range(1, n):  # 0 e' lo sfondo
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area_px:
            continue

        maschera = etichette == i
        u_c, v_c = float(centroidi[i][0]), float(centroidi[i][1])

        # Il centroide puo' cadere appena fuori dalla macchia (forme concave):
        # si usa la depth **mediana** della macchia, non quella di un pixel.
        dep = float(np.median(d[maschera]))
        superficie = deproject(u_c, v_c, dep, K, T)

        # Mezza dimensione dalla larghezza apparente: larghezza_px / focale *
        # distanza da' la larghezza in metri alla distanza dell'oggetto.
        largh_px = float(stats[i, cv2.CC_STAT_WIDTH])
        half = 0.5 * (largh_px / fx * dep)

        # Coerenza: un oggetto solido sta dentro la propria dimensione. Una
        # macchia molto piu' alta che larga e' un bordo fra due superfici a
        # distanze diverse, non una cosa.
        span = float(z_world[maschera].max() - z_world[maschera].min())
        if half > 0 and span > max_depth_span_ratio * (2 * half):
            continue

        # Compensazione: il centro sta mezzo oggetto **oltre** la superficie,
        # lungo la direzione in cui la camera guarda.
        vista = superficie - T[:3, 3]
        vista /= np.linalg.norm(vista)
        centro = superficie + vista * half

        colore, dist = classify_color(rgb01[maschera].mean(axis=0))

        out.append(
            Detection(
                center=centro,
                surface=superficie,
                color=colore,
                color_distance=dist,
                pixel=(u_c, v_c),
                area_px=area,
                half_extent=half,
            )
        )

    out.sort(key=lambda t: t.area_px, reverse=True)
    return out
