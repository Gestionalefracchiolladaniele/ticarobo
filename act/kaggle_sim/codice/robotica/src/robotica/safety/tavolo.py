"""Verifica delle pose del SO-101 sul tavolo, prima di mandarle al braccio.

Come ``gesti.py`` per il gioco: ogni posa passa da qui, nessuna eccezione, e
le regole sono limiti e distanze, quindi trasferiscono al braccio vero.

1. **Ogni angolo dentro il range del giunto** (non dell'attuatore: ``gesti.py``).
2. **Nessun pezzo del braccio sotto il tavolo**, non solo la punta: chiudendo
   sulla caramella sono le dita a scendere, e un polso basso tocca prima della
   punta. Si guardano **tutti i punti** delle geometrie del braccio (angoli dei
   blocchi, vertici delle mesh), non un campo comodo che misura un'altra cosa
   (``geom_rbound``, D20). La base e' esclusa: appoggia sul tavolo.
3. **Nessun pezzo del braccio dentro un bicchiere** (un cilindro dichiarato):
   un bicchiere di plastica urtato si rovescia, e il robot non lo saprebbe.
4. **La punta dentro la zona raggiungibile**, misurata il 2026-10-11.
5. **Il gomito non steso** (D17): li' il braccio non si muove piu' bene.

Controlla **pose**, non percorsi: fra una posa e l'altra il braccio passa per
pose intermedie. Chi compone il movimento deve farlo passare **in alto**
(sopra i bicchieri) e scendere in verticale: e' una regola di chi muove, e
``verifica_percorso`` la controlla interpolando.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from robotica.safety.checks import SafetyReport, Severity
from robotica.sim.so101_presa import ASSE_BASE, PIEGATURA_MINIMA, TAVOLO
from robotica.sim.so101_rps import GIUNTI, _qpos_da

#: Margine sopra il tavolo per ogni pezzo del braccio.
MARGINE_TAVOLO = 0.002

#: La zona dove la punta puo' stare, distanza dall'asse della base (metri).
#: Presa verticale fino a 19 cm, rilascio inclinato fino a 23 (smistamento.md).
RAGGIO_MIN, RAGGIO_MAX = 0.08, 0.24

#: Margine attorno a un bicchiere, in orizzontale e sopra il bordo.
MARGINE_BICCHIERE = 0.015


@dataclass(frozen=True)
class Bicchiere:
    """Un bicchiere sul tavolo: centro (x, y) nel frame del modello, raggio, altezza."""

    centro: tuple[float, float]
    raggio: float = 0.04
    altezza: float = 0.10


def punti_del_braccio(model: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    """Tutti i punti delle geometrie di collisione del braccio, base esclusa (N x 3)."""
    base = model.body("base").id
    punti = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] in (0, base) or model.geom_contype[g] == 0:
            continue
        R, c = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        tipo = model.geom_type[g]
        if tipo == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            punti.append(model.mesh_vert[a:a + n] @ R.T + c)
        elif tipo == mujoco.mjtGeom.mjGEOM_BOX:
            s = model.geom_size[g]
            angoli = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * s
            punti.append(angoli @ R.T + c)
        else:  # sfere, capsule, cilindri: centro e raggio, per difetto in basso
            punti.append((c - np.array([0, 0, model.geom_rbound[g]]))[None, :])
    return np.vstack(punti)


def verifica_posa(
    model: mujoco.MjModel, ctrl: np.ndarray, bicchieri: tuple[Bicchiere, ...] = (),
    tavolo: float = TAVOLO,
) -> SafetyReport:
    """Le cinque regole su un comando di sei numeri (ordine ``GIUNTI``).

    ``tavolo``: la quota del tavolo nel frame del modello. Di default quella del
    modello; sul ferro quella **misurata** (``reale/mani.py``), perche' il braccio
    vero non tiene le dita dove crede il modello."""
    rep = SafetyReport()
    ctrl = np.asarray(ctrl, dtype=float)
    for nome, q in zip(GIUNTI, ctrl, strict=True):
        lo, hi = model.joint(nome).range
        if not lo <= q <= hi:
            rep.add("joint_range", f"{nome}={q:.3f} fuori dal giunto [{lo:.3f}, {hi:.3f}]")

    d = _qpos_da(model, ctrl)
    punti = punti_del_braccio(model, d)
    basso = float(punti[:, 2].min())
    if basso < tavolo + MARGINE_TAVOLO:
        rep.add("tavolo", f"un pezzo del braccio a {(basso - tavolo) * 1000:.1f} mm dal tavolo "
                f"(minimo {MARGINE_TAVOLO * 1000:.0f})")

    for b in bicchieri:
        dentro = np.linalg.norm(punti[:, :2] - np.asarray(b.centro), axis=1) < (
            b.raggio + MARGINE_BICCHIERE)
        bassi = punti[dentro, 2] < tavolo + b.altezza + MARGINE_BICCHIERE
        if bassi.any():
            rep.add("bicchiere", f"il braccio entra nel bicchiere in {b.centro}: "
                    f"{int(bassi.sum())} punti sotto il bordo")

    punta = d.site("gripperframe").xpos
    r = float(np.linalg.norm(punta[:2] - ASSE_BASE))
    if not RAGGIO_MIN <= r <= RAGGIO_MAX:
        rep.add("portata", f"punta a {r * 100:.1f} cm dall'asse, zona "
                f"[{RAGGIO_MIN * 100:.0f}, {RAGGIO_MAX * 100:.0f}]")

    gomito = ctrl[GIUNTI.index("elbow_flex")]
    if abs(gomito) < PIEGATURA_MINIMA:
        rep.add("steso", f"elbow_flex={gomito:.2f}: braccio quasi steso (D17)", Severity.WARNING)
    return rep


def verifica_percorso(
    model: mujoco.MjModel, tappe: list[np.ndarray], bicchieri: tuple[Bicchiere, ...] = (),
    passi: int = 10, tavolo: float = TAVOLO,
) -> SafetyReport:
    """Ogni tappa **e** le pose intermedie fra due tappe (interpolate nei giunti,
    come fa l'esecutore): una posa giusta raggiunta passando dentro un bicchiere
    non e' un movimento giusto."""
    rep = SafetyReport()
    for i, (a, b) in enumerate(zip(tappe, tappe[1:], strict=False)):
        for s in np.linspace(0.0, 1.0, passi + 1)[1:]:
            parziale = verifica_posa(model, a + s * (np.asarray(b) - a), bicchieri, tavolo)
            for v in parziale.violations:
                rep.add(v.rule, f"tratto {i}->{i + 1} al {s:.0%}: {v.detail}", v.severity)
            if not parziale.ok:
                break
    if tappe:
        for v in verifica_posa(model, tappe[0], bicchieri, tavolo).violations:
            rep.add(v.rule, f"tappa 0: {v.detail}", v.severity)
    return rep
