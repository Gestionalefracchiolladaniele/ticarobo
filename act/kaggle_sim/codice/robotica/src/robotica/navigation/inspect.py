"""Guardare una stazione e decidere: il colore diventa una destinazione.

**Cosa cambia rispetto a prima.** Un ordine in ``mission.py`` dice gia' dove
prendere e dove portare: il robot **esegue**. Qui l'ordine dice solo *dove
andare a guardare*, e la destinazione **non esiste finche' il robot non ha
visto**. E' la differenza fra uno script e una decisione, ed e' quella che sul
nastro (D17) veniva dalla camera.

**La terza categoria, di nuovo.** Sul nastro c'erano due zone piu' **un colore
che si lascia passare**, e non agire non era fallire. Qui un colore senza
destinazione dichiarata si **lascia dov'e'**: il robot ha guardato, ha
riconosciuto, e ha deciso che non e' suo. Contarlo come fallimento
nasconderebbe la differenza fra *"non ce l'ho fatta"* e *"non era mio"*.

**Quattro esiti, non due**, perche' un esito unico non e' attribuibile:

``riconosciuto``
    visto e c'e' una destinazione: il robot lo porta
``non_gestito``
    visto, colore noto, **nessuna destinazione**: si lascia li'. Decisione.
``incerto``
    visto ma il colore non si nomina (``color_distance`` oltre soglia).
    Diverso dal precedente: qui il robot **non sa**, non e' che non gli
    compete. Un sistema che confonde "non lo so" con "non e' mio" prende la
    decisione giusta per il motivo sbagliato, e il giorno che
    l'illuminazione cambia nessuno se ne accorge.
``vuoto``
    nessun oggetto nella stazione

**Cosa non fa, deliberatamente.** Non sa cosa sia MuJoCo, un pacco o un
magazzino: riceve immagini e matrici, come ``perception/detect.py`` (D18).
Non chiama la scena, non legge posizioni note. Si puo' passargli le immagini
di una camera vera senza cambiare una riga -- che e' l'unico modo di sapere
se trasferisce.

**Perche' non ``planning/``.** Stessa ragione di ``mission.py``: non c'e' una
sequenza da scoprire. E' una **classificazione seguita da una scelta in
tabella**, e modellarla come ricerca simbolica vorrebbe dire inventare un
problema per poterlo risolvere.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from robotica.perception.detect import Detection, detect_objects
from robotica.transforms import Pose

__all__ = [
    "Outcome",
    "Inspection",
    "ColorRouting",
    "inspect_station",
]

# Oltre questa distanza dal riferimento, un colore non si nomina. E'
# **piu' stretta** della soglia interna di ``classify_color`` (0.45): li'
# serve a non dire sciocchezze, qui a non *agire* su un dubbio. Muovere un
# robot e' piu' caro che stampare un nome.
MAX_COLOR_DISTANCE = 0.30

# Quanto lontano dal centro della stazione puo' stare un oggetto per essere
# considerato "in quella stazione". Si passa esplicitamente (il raggio lo
# conosce la scena, non questo modulo) ma serve un default sensato.
DEFAULT_STATION_RADIUS = 0.45


class Outcome(Enum):
    """L'esito di un'ispezione. **Quattro, e ognuno dice una cosa diversa.**"""

    RICONOSCIUTO = "riconosciuto"
    NON_GESTITO = "non gestito"
    INCERTO = "incerto"
    VUOTO = "vuoto"


@dataclass(frozen=True)
class Inspection:
    """Cosa il robot ha visto guardando una stazione, e cosa ne consegue.

    Attributes:
        outcome: uno dei quattro esiti.
        color: il colore **percepito**, o ``None``. Non e' il colore vero:
            quello lo conosce solo la simulazione, e serve a misurare
            l'errore, non a decidere.
        destination: dove portarlo, o ``None`` se non si porta da nessuna
            parte. **E' il risultato della decisione**, e prima
            dell'ispezione non esiste.
        confidence: quanto era vicino il colore al riferimento, come
            distanza (piu' piccolo = piu' sicuro). Si tiene perche' un fatto
            percepito non e' un fatto noto, e chi decide deve poterlo sapere
            -- la stessa ragione di ``color_distance`` in ``Detection``.
        position: dove sta l'oggetto nel mondo, dalla percezione.
        reason: perche' questo esito, in parole. Un sistema che si arrende
            deve dire di essersi arreso (D14).
        n_detections: quanti oggetti c'erano nel volume. Piu' di uno non e'
            un errore, ma chi legge deve saperlo.
    """

    outcome: Outcome
    color: str | None = None
    destination: str | None = None
    confidence: float = float("inf")
    position: np.ndarray | None = None
    reason: str = ""
    n_detections: int = 0

    @property
    def actionable(self) -> bool:
        """C'e' qualcosa da fare? Solo ``RICONOSCIUTO`` lo e'."""
        return self.outcome is Outcome.RICONOSCIUTO

    def __str__(self) -> str:
        if self.outcome is Outcome.RICONOSCIUTO:
            return f"{self.color} -> {self.destination}"
        return f"{self.outcome.value}: {self.reason}"


@dataclass(frozen=True)
class ColorRouting:
    """Quale colore va dove. I colori assenti si lasciano dove sono.

    **E' una tabella, non una gerarchia.** Che il verde non compaia non e'
    una dimenticanza: e' la dichiarazione che il verde non si tocca. Un
    colore senza riga e' la terza categoria del nastro (D17).

    Args:
        table: ``{"rosso": "spedizione", "blu": "scaffale_b"}``.
    """

    table: dict[str, str]

    def destination_for(self, color: str | None) -> str | None:
        return self.table.get(color) if color else None

    @property
    def handled(self) -> set[str]:
        return set(self.table)

    def __str__(self) -> str:
        righe = ", ".join(f"{c}->{d}" for c, d in sorted(self.table.items()))
        return f"[{righe}]"


def inspect_station(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_world_camera: Pose,
    station_xy: np.ndarray,
    routing: ColorRouting,
    plane_z: float = 0.0,
    station_radius: float = DEFAULT_STATION_RADIUS,
    max_object_height: float = 0.40,
    max_color_distance: float = MAX_COLOR_DISTANCE,
    ignore_mask: np.ndarray | None = None,
) -> Inspection:
    """Guarda una stazione e decide cosa fare di cio' che c'e'.

    Args:
        rgb, depth, K, T_world_camera: cio' che la camera ha misurato. Da
            dove venga -- simulazione o hardware -- non riguarda questo
            modulo.
        station_xy: il centro della stazione nel mondo. **Non si deriva
            dall'immagine**: e' dove il robot ha deciso di andare, quindi lo
            sa gia'.
        routing: la tabella colore -> destinazione.
        plane_z: quota del pavimento. Su un AMR e' 0; su un tavolo no, ed e'
            l'unica riga che cambia fra i due casi.
        station_radius: quanto e' larga la stazione. **Definisce il volume**
            in cui si cerca: oltre, si rischia di rilevare i muri --
            misurato, a +/-0.90 m dalla banchina entra ``muro_ovest`` con
            34.517 px contro i 1.734 di un pacco.
        max_color_distance: oltre questa distanza il colore non si nomina e
            l'esito e' ``INCERTO``.
        ignore_mask: pixel da ignorare, tipicamente il robot stesso. Qui la
            camera guarda **avanti e in basso** e il corpo del robot resta
            fuori, ma su un montaggio diverso servirebbe (D18).

    Returns:
        Un'``Inspection``. **Mai un'eccezione per "non ho visto niente"**:
        una stazione vuota e' un esito, non un guasto.

    La detection scelta e' **la piu' vicina al centro della stazione**, non
    la piu' grande. ``detect_objects`` ordina per area decrescente, e
    prendere la prima presume che cio' che si cerca sia l'oggetto piu'
    grosso dell'inquadratura -- presupposto falso appena entra un muro.
    """
    xy = np.asarray(station_xy, dtype=float)[:2]
    bounds = (
        xy[0] - station_radius,
        xy[0] + station_radius,
        xy[1] - station_radius,
        xy[1] + station_radius,
    )

    detections = detect_objects(
        rgb,
        depth,
        K,
        T_world_camera,
        plane_z=plane_z,
        xy_bounds=bounds,
        max_height=max_object_height,
        ignore_mask=ignore_mask,
    )

    # Dentro il quadrato, ma la stazione e' un **disco**: l'angolo del
    # quadrato sta a radius*sqrt(2), cioe' il 41% piu' lontano.
    vicini: list[tuple[float, Detection]] = []
    for d in detections:
        dist = float(np.linalg.norm(np.asarray(d.center)[:2] - xy))
        if dist <= station_radius:
            vicini.append((dist, d))

    if not vicini:
        return Inspection(
            outcome=Outcome.VUOTO,
            reason=(
                f"nessun oggetto entro {station_radius:.2f} m dal centro"
                + (f" ({len(detections)} fuori raggio)" if detections else "")
            ),
            n_detections=len(detections),
        )

    vicini.sort(key=lambda t: t[0])
    _, scelta = vicini[0]
    pos = np.asarray(scelta.center, dtype=float)

    if scelta.color is None or scelta.color_distance > max_color_distance:
        return Inspection(
            outcome=Outcome.INCERTO,
            color=None,
            confidence=scelta.color_distance,
            position=pos,
            reason=(
                f"colore non riconoscibile (distanza {scelta.color_distance:.3f} "
                f"oltre {max_color_distance:.2f})"
            ),
            n_detections=len(vicini),
        )

    dest = routing.destination_for(scelta.color)
    if dest is None:
        return Inspection(
            outcome=Outcome.NON_GESTITO,
            color=scelta.color,
            confidence=scelta.color_distance,
            position=pos,
            reason=f"'{scelta.color}' non ha destinazione: si lascia dov'e'",
            n_detections=len(vicini),
        )

    return Inspection(
        outcome=Outcome.RICONOSCIUTO,
        color=scelta.color,
        destination=dest,
        confidence=scelta.color_distance,
        position=pos,
        reason=f"'{scelta.color}' va a {dest}",
        n_detections=len(vicini),
    )
