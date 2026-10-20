"""Il SO-101 vero e il suo modello: da gradi LeRobot a radianti MuJoCo, e ritorno.

Il driver (LeRobot, in un processo e un venv suoi: D22) parla **gradi LeRobot**,
il modello di menagerie parla **radianti**. Fra le due lingue, per ogni giunto
del corpo:

    q_modello = SEGNO * radians(gradi_lerobot) + SCOSTAMENTO

- **Lo zero dei gradi LeRobot e' il centro del range calibrato**, (min+max)/2,
  non la posa a meta' della calibrazione (``motors_bus._normalize``). Sulla
  spalla le due cose distano circa 5 gradi.
- **Lo zero del modello** e' quello di ``so101.xml``: range simmetrici attorno
  a zero.

Che i due zeri coincidano e i versi siano concordi e' un **indizio** (range
simmetrici da tutte e due le parti, ampiezze simili: gomito 194 gradi contro
194), **non una misura**. Un segno sbagliato qui trasforma un gesto approvato da
``safety/`` in un altro movimento sul ferro, e ``safety/`` non se ne accorge:
e' un errore di frame sui giunti. Si verifica col gemello in sola lettura
(``scripts/gemello.py``): il braccio vero mosso a mano, il modello che lo copia.

La pinza in LeRobot e' 0..100 sul range calibrato; nel modello e' un angolo.
Ipotesi: lineare, 0 = chiusa = estremo basso del giunto del modello.
"""

from __future__ import annotations

import numpy as np

from robotica.sim.so101_rps import GIUNTI

#: I cinque giunti in gradi; la pinza e' a parte (0..100).
CORPO: tuple[str, ...] = GIUNTI[:5]
PINZA = GIUNTI[5]

#: (segno, scostamento in radianti) per giunto.
#:
#: - **Segni: misurati** col gemello il 2026-10-10, concordi (+1) sui primi
#:   quattro. ``wrist_roll`` e' costato tre prove: ne' il verso visto nella
#:   finestra ne' il confronto di posizione dicevano niente, perche' lo zero
#:   era sbagliato di 101 gradi e mascherava tutto. **Prima lo zero, poi il
#:   segno**; e una rotazione vista su uno schermo non e' una misura.
#: - **Scostamento di ``wrist_roll``: misurato**, -101,5 gradi. In calibrazione
#:   il roll non si muove: LeRobot gli da' "giro intero" e il suo zero diventa
#:   la rotazione, qualsiasi, in cui stava il polso nella posa a meta'. Lo zero
#:   del modello e' dichiarato: ganascia mobile **sopra**, la pinza si apre nel
#:   piano in cui si piega il braccio. Polso vero messo cosi' (foto di lato e da
#:   dietro, avambraccio orizzontale): LeRobot segnava +101,5.
#: - **Scostamenti degli altri giunti: zero, entro ~8 gradi.** Previsti dal
#:   file di calibrazione fra -5 e +8; la posa "lampada" tenuta a mano non li
#:   ha ne' confermati ne' smentiti. Per misurarli serve un riferimento migliore
#:   della mano (livella).
CONVERSIONE: dict[str, tuple[float, float]] = {
    "shoulder_pan": (1.0, 0.0),
    "shoulder_lift": (1.0, 0.0),
    "elbow_flex": (1.0, 0.0),
    "wrist_flex": (1.0, 0.0),
    "wrist_roll": (1.0, float(np.radians(-101.5))),
}

#: True se 0 di LeRobot e' l'estremo **alto** del giunto pinza nel modello.
#: Misurato il 2026-10-10: 0 = chiusa, concorde.
PINZA_INVERTITA = False


def da_lerobot(pos: dict[str, float], range_pinza: tuple[float, float]) -> np.ndarray:
    """Gradi LeRobot (dict per nome, pinza 0..100) -> sei radianti, ordine ``GIUNTI``.

    ``range_pinza`` e' il range del giunto pinza **letto dal modello**
    (``model.joint("gripper").range``), non una costante copiata qui.
    """
    q = np.empty(len(GIUNTI))
    for i, nome in enumerate(CORPO):
        segno, scostamento = CONVERSIONE[nome]
        q[i] = segno * np.radians(pos[nome]) + scostamento
    frazione = pos[PINZA] / 100.0
    if PINZA_INVERTITA:
        frazione = 1.0 - frazione
    lo, hi = range_pinza
    q[5] = lo + frazione * (hi - lo)
    return q


def a_lerobot(q: np.ndarray, range_pinza: tuple[float, float]) -> dict[str, float]:
    """L'inverso di :func:`da_lerobot`: sei radianti -> gradi LeRobot per nome."""
    q = np.asarray(q, dtype=float)
    pos: dict[str, float] = {}
    for i, nome in enumerate(CORPO):
        segno, scostamento = CONVERSIONE[nome]
        pos[nome] = float(np.degrees((q[i] - scostamento) / segno))
    lo, hi = range_pinza
    frazione = (q[5] - lo) / (hi - lo)
    if PINZA_INVERTITA:
        frazione = 1.0 - frazione
    pos[PINZA] = float(frazione * 100.0)
    return pos
