"""Il colore letto dalla camera invece che dal dizionario.

E' il punto in cui la percezione entra nel sistema, ed era **previsto**:
``SortingEnv.color_of`` esiste da D15 proprio come punto di sostituzione, e la
sua docstring lo dice. Il colore non e' mai stato nel nome dell'oggetto
(``cube_red_1``) ma una **proprieta'** interrogabile, perche' una camera puo'
produrre una proprieta' e non puo' rinominare gli oggetti.

Qui quella previsione viene riscossa: ``PerceivedScene`` espone ``color_of`` e
``object_names`` come l'ambiente, ma le risposte vengono da un'immagine.
``planning/bridge.py`` non se ne accorge — chiede *"sai rispondere a
``color_of``?"* e non chiede a chi.

**Cosa cambia davvero, e non e' il colore.** Finora un fatto era vero o falso:
``color(cube_a, rosso)`` lo era. Ora e' *"il colore misurato dista 0.03 dal
rosso"* — un'**ipotesi con un margine**. Da qui ``uncertainty``, che il
pianificatore oggi ignora e che esiste perche' la domanda si possa porre: cosa
fa ``safety`` quando il fatto su cui verifica e' incerto? E' il territorio nuovo
della Fase 3, e cancellare il margine lo renderebbe invisibile.

**L'associazione ai nomi e' il punto debole, e va detto.** La camera vede
*oggetti*, non *nomi*: non esiste nulla nell'immagine che dica "questo e'
``cube_a``". Qui i nomi si assegnano per vicinanza alle posizioni note, che e'
informazione privilegiata — **serve solo a confrontarsi col resto del sistema,
che ragiona per nomi**. Sul robot vero i nomi nascono dal tracciamento nel
tempo: il primo oggetto visto diventa ``obj_0`` e resta tale finche' lo si
segue. Il limite e' l'associazione, non la percezione.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from robotica.perception.detect import Detection, detect_objects
from robotica.transforms import Pose

__all__ = ["PerceivedObject", "PerceivedScene", "perceive"]

# Oltre questa distanza un oggetto visto non e' lo stesso di un oggetto noto.
# 5 cm: piu' dell'errore di percezione misurato (~8 mm) e meno della distanza
# fra due cubi vicini (16 cm), quindi non puo' confondere l'uno con l'altro.
MAX_MATCH_DISTANCE = 0.05


@dataclass(frozen=True)
class PerceivedObject:
    """Un oggetto visto, col nome che gli e' stato associato."""

    name: str
    color: str | None
    position: np.ndarray
    uncertainty: float
    detection: Detection


@dataclass
class PerceivedScene:
    """La scena come la vede la camera.

    Espone ``color_of`` e ``object_names`` con la stessa firma
    dell'ambiente, cosi' ``planning/bridge.py`` puo' riceverla al posto della
    scena senza modifiche. **Non eredita da nulla**: la compatibilita' e' nei
    metodi, non in una gerarchia, ed e' il modo di non legare il planning alla
    simulazione.
    """

    objects: dict[str, PerceivedObject] = field(default_factory=dict)
    unmatched: list[Detection] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def object_names(self) -> list[str]:
        """I nomi degli oggetti visti. **Solo quelli visti**: un cubo nascosto
        dal braccio non compare, ed e' cio' che accade anche sul robot vero."""
        return sorted(self.objects)

    def color_of(self, name: str) -> str:
        """Il colore percepito.

        Raises:
            KeyError: se l'oggetto non e' stato visto, o se e' stato visto ma il
                colore non e' riconoscibile. Sono due casi diversi e il
                messaggio li distingue: "non c'e'" e "non so dirlo" portano a
                decisioni diverse.
        """
        if name not in self.objects:
            raise KeyError(
                f"'{name}' non e' fra gli oggetti percepiti: {self.object_names()}"
            )
        colore = self.objects[name].color
        if colore is None:
            raise KeyError(
                f"'{name}' e' visibile ma il colore non e' riconoscibile "
                f"(distanza {self.objects[name].uncertainty:.2f})"
            )
        return colore

    def position_of(self, name: str) -> np.ndarray:
        """Dove sta l'oggetto, secondo la camera."""
        if name not in self.objects:
            raise KeyError(f"'{name}' non percepito")
        return self.objects[name].position


def perceive(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    T_world_camera: Pose,
    plane_z: float,
    known_positions: dict[str, np.ndarray],
    max_match_distance: float = MAX_MATCH_DISTANCE,
    **detect_kwargs,
) -> PerceivedScene:
    """Guarda la scena e produce i fatti che il planning consuma.

    Args:
        known_positions: le posizioni note degli oggetti, **usate solo per
            associare un nome a cio' che si vede**. Vedi il limite dichiarato
            in cima al modulo.
        **detect_kwargs: passati a ``detect_objects`` (``xy_bounds``,
            ``ignore_mask``, ...).

    Returns:
        La scena percepita, con **tre** esiti tenuti separati: gli oggetti
        riconosciuti, quelli visti e non associati a un nome, e i nomi noti che
        non sono stati visti. Un totale unico direbbe "4 su 6" senza dire se il
        problema e' che non si vede o che non si riconosce — la stessa ragione
        dei contatori separati del nastro (D17).
    """
    rilevati = detect_objects(rgb, depth, K, T_world_camera, plane_z, **detect_kwargs)

    scena = PerceivedScene()
    gia_preso: set[str] = set()

    # Ogni detection al nome noto piu' vicino, se abbastanza vicino. Si procede
    # in ordine di area decrescente — gli oggetti piu' grandi sono i piu'
    # affidabili — e un nome gia' assegnato non si riassegna.
    for det in rilevati:
        candidati = [
            (n, float(np.linalg.norm(det.center - p)))
            for n, p in known_positions.items()
            if n not in gia_preso
        ]
        if not candidati:
            scena.unmatched.append(det)
            continue

        nome, dist = min(candidati, key=lambda x: x[1])
        if dist > max_match_distance:
            scena.unmatched.append(det)
            continue

        gia_preso.add(nome)
        scena.objects[nome] = PerceivedObject(
            name=nome,
            color=det.color,
            position=det.center,
            uncertainty=det.color_distance,
            detection=det,
        )

    scena.missing = sorted(set(known_positions) - gia_preso)
    return scena
