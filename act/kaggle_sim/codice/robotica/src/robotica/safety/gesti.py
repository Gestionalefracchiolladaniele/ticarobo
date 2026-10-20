"""Verifica dei gesti del SO-101, prima di mandarli al braccio.

Un gesto e' un comando come un altro: passa da qui, nessuna eccezione. Tre
regole, e tutte trasferiscono al braccio vero perche' sono limiti e distanze:

1. **Ogni angolo dentro il range del giunto, non dell'attuatore.** Nel
   modello ``wrist_roll`` accetta comandi fino a 2,841 rad ma il giunto si
   ferma a 2,744: controllare il ``ctrlrange`` darebbe "comando accettato" a
   un braccio che non ci arriva.
2. **Il TCP resta lontano dalla mano del giocatore.** Davanti c'e' una
   persona. Il controllo e' sulla punta e sulla posa finale: basta per gesti
   statici raggiunti lentamente, **non** per movimenti verso la persona.
3. **Il braccio non e' steso** (avviso). Vicino a ``elbow_flex = 0`` e' in
   configurazione singolare e non si muove piu' bene (D17).
"""

from __future__ import annotations

import mujoco
import numpy as np

from robotica.safety.checks import SafetyReport, Severity
from robotica.sim.so101_rps import GIUNTI, ZONA_MANO, ZONA_MANO_MEZZA, tcp_di

#: Distanza minima fra il TCP e il bordo della zona della mano.
DISTANZA_MANO = 0.15

#: Sotto questa piegatura del gomito il braccio e' quasi steso.
PIEGATURA_MINIMA = 0.35

#: Il TCP non scende sotto questa quota: sotto c'e' il tavolo.
QUOTA_MINIMA = 0.05


def verifica_gesto(model: mujoco.MjModel, ctrl: np.ndarray) -> SafetyReport:
    """Le tre regole su un comando di sei numeri (ordine ``GIUNTI``)."""
    rep = SafetyReport()
    ctrl = np.asarray(ctrl, dtype=float)

    for nome, q in zip(GIUNTI, ctrl, strict=True):
        lo, hi = model.joint(nome).range
        if not lo <= q <= hi:
            rep.add("joint_range", f"{nome}={q:.3f} fuori dal giunto [{lo:.3f}, {hi:.3f}]")

    tcp = tcp_di(model, ctrl)
    # Distanza dal box della zona mano (zero se ci sta dentro).
    fuori = np.maximum(np.abs(tcp - ZONA_MANO) - ZONA_MANO_MEZZA, 0.0)
    dist = float(np.linalg.norm(fuori))
    if dist < DISTANZA_MANO:
        rep.add(
            "mano", f"TCP a {dist * 1000:.0f} mm dalla mano (minimo {DISTANZA_MANO * 1000:.0f})"
        )

    if tcp[2] < QUOTA_MINIMA:
        rep.add("quota", f"TCP a z={tcp[2]:.3f}, sotto {QUOTA_MINIMA:.2f}")

    gomito = ctrl[GIUNTI.index("elbow_flex")]
    if abs(gomito) < PIEGATURA_MINIMA:
        rep.add(
            "steso",
            f"elbow_flex={gomito:.2f}: braccio quasi steso, vicino alla singolarita'",
            Severity.WARNING,
        )
    return rep
