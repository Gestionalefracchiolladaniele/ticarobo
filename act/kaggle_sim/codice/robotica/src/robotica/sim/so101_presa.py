"""Il SO-101 che prende dal tavolo: dove mettere la punta, e con che inclinazione.

Per il gioco bastava la pinza orizzontale (``so101_rps._postura_planare``).
Per prendere una caramella serve la pinza **puntata in giu'**, e per lasciarla
in un bicchiere **inclinata**: si dichiara la punta (x, y, z) e l'inclinazione,
e gli angoli ne discendono.

**Quote lette dal modello, non derivate** (misurate il 2026-10-11, vedi
``esperimenti/so101/smistamento.md``): la base appoggia a z = 0 e il suo punto
piu' basso e' a −2,4 mm, quindi il tavolo e' a ``TAVOLO``. Nello stesso
documento la mappa di dove il braccio arriva: presa verticale fra 10 e 19 cm
dall'asse della base, rilascio sopra un bicchiere solo **inclinato** (verticale
a 13 cm il polso dovrebbe piegarsi oltre ±95°).

Il braccio ha 5 giunti, e il polso roll non cambia ne' la punta ne' dove
puntano le dita: restano 4 giunti (pan, spalla, gomito, polso) contro 4
richieste (x, y, z, inclinazione). Si risolve con i minimi quadrati smorzati
sullo jacobiano di MuJoCo, come ``_postura_planare``, da **pose di partenza
fisse**: lo stesso bersaglio deve dare sempre gli stessi angoli, o non si
puo' provare.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from robotica.paths import robot_dir
from robotica.sim.so101_rps import GIUNTI, _apertura_pinza, _qpos_da, distanza_punte

#: Quota del tavolo nel frame del modello: il punto piu' basso della base.
TAVOLO = -0.0024

#: Dove sta l'asse della base (``shoulder_pan``) nel piano del tavolo.
ASSE_BASE = np.array([0.0388, 0.0])

#: Inclinazione della pinza: 90 gradi = dita in giu', 0 = orizzontali in avanti.
VERTICALE = np.pi / 2
RILASCIO = np.pi / 4

#: Sotto questa piegatura il gomito e' quasi steso: singolarita' (D17).
PIEGATURA_MINIMA = 0.35

#: Pose di partenza del calcolo (pan escluso, lo si punta verso il bersaglio).
#: Piegate: da steso lo jacobiano perde rango e il primo passo e' casuale.
PARTENZE = (
    (-0.5, 0.8, 1.2), (-1.0, 1.2, 0.8), (0.0, 0.5, 1.4), (-0.3, 1.4, 0.2), (0.3, 0.9, 0.6),
)


@dataclass(frozen=True)
class Postura:
    """Quattro angoli (pan, spalla, gomito, polso) e quanto lontano sono dalla richiesta."""

    q: np.ndarray
    errore: float  # metri, punta
    scarto: float  # radianti, inclinazione


def modello() -> mujoco.MjModel:
    """Il braccio del produttore, senza scena: per la cinematica basta."""
    return mujoco.MjModel.from_xml_path(str(robot_dir("so101") / "scene.xml"))


def verso_dita(inclinazione: float, dove: np.ndarray) -> np.ndarray:
    """Il verso delle dita: inclinato verso il basso, nel piano verticale che va
    dall'asse della base verso ``dove`` (le dita guardano lontano dal robot)."""
    u = np.asarray(dove[:2], dtype=float) - ASSE_BASE
    u = u / (np.linalg.norm(u) + 1e-12)
    return np.array([np.cos(inclinazione) * u[0], np.cos(inclinazione) * u[1],
                     -np.sin(inclinazione)])


def postura(
    model: mujoco.MjModel, punta: np.ndarray, inclinazione: float, iterazioni: int = 400
) -> Postura:
    """Gli angoli per mettere la punta (site ``gripperframe``) in ``punta`` con le dita
    inclinate di ``inclinazione``. Solleva ``ValueError`` se non c'e' una postura
    col gomito piegato: dire "non ci arrivo" e' una risposta, inventare no."""
    punta = np.asarray(punta, dtype=float)
    verso = verso_dita(inclinazione, punta)
    nomi = GIUNTI[:4]
    idx = [model.joint(n).qposadr[0] for n in nomi]
    dof = [model.joint(n).dofadr[0] for n in nomi]
    lo = np.array([model.joint(n).range[0] for n in nomi])
    hi = np.array([model.joint(n).range[1] for n in nomi])
    sid = model.site("gripperframe").id
    pan = float(np.arctan2(punta[1] - ASSE_BASE[1], punta[0] - ASSE_BASE[0]))
    d = mujoco.MjData(model)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

    def risolvi(q: np.ndarray) -> Postura:
        for _ in range(iterazioni):
            d.qpos[idx] = q
            mujoco.mj_kinematics(model, d)
            mujoco.mj_comPos(model, d)
            z = d.body("gripper").xmat.reshape(3, 3)[:, 2]  # le dita puntano lungo -z
            err = np.concatenate([punta - d.site_xpos[sid], verso + z])
            mujoco.mj_jacSite(model, d, jacp, jacr, sid)
            jd = np.cross(z[:, None], jacr[:, dof], axis=0)  # derivata di -z: z x w
            J = np.vstack([jacp[:, dof], jd])
            q = np.clip(q + 0.5 * np.linalg.solve(J.T @ J + 1e-3 * np.eye(4), J.T @ err), lo, hi)
        d.qpos[idx] = q
        mujoco.mj_kinematics(model, d)
        z = d.body("gripper").xmat.reshape(3, 3)[:, 2]
        scarto = float(np.arccos(np.clip(-z @ verso, -1.0, 1.0)))
        return Postura(q.copy(), float(np.linalg.norm(punta - d.site_xpos[sid])), scarto)

    buone = []
    for spalla, gomito, polso in PARTENZE:
        p = risolvi(np.array([pan, spalla, gomito, polso]))
        if p.errore < 1e-3 and p.scarto < np.radians(2) and abs(p.q[2]) >= PIEGATURA_MINIMA:
            buone.append(p)
    if not buone:
        r = np.linalg.norm(punta[:2] - ASSE_BASE)
        raise ValueError(
            f"nessuna postura col gomito piegato per la punta a {r * 100:.1f} cm dall'asse, "
            f"{(punta[2] - TAVOLO) * 100:.1f} cm sopra il tavolo, inclinata di "
            f"{np.degrees(inclinazione):.0f} gradi: fuori portata o oltre un giunto"
        )
    # Fra le posture buone la piu' lontana dal gomito steso: e' la piu' mobile.
    return max(buone, key=lambda p: abs(p.q[2]))


def comando(postura_: Postura, roll: float, pinza: float) -> np.ndarray:
    """I sei numeri per ``data.ctrl`` (ordine ``GIUNTI``)."""
    return np.array([*postura_.q, roll, pinza])


def pinza_aperta_chiusa(model: mujoco.MjModel, apertura: float = 0.05) -> tuple[float, float]:
    """(chiusa, aperta di ``apertura`` metri fra le punte), misurati sul modello."""
    return _apertura_pinza(model, apertura)


def pinza_su_caramella(model: mujoco.MjModel, larghezza: float = 0.02) -> float:
    """L'angolo della pinza quando stringe un oggetto largo ``larghezza``: le punte
    si fermano li', qualunque cosa si comandi. Misurato sul modello."""
    lo, hi = model.joint("gripper").range
    angoli = np.linspace(lo, hi, 800)
    d = mujoco.MjData(model)
    dist = np.array([distanza_punte(model, a, d) for a in angoli])
    return float(angoli[int(np.argmin(np.abs(dist - larghezza)))])


def centro_dita(model: mujoco.MjModel, ctrl: np.ndarray) -> np.ndarray:
    """Il punto fra le punte delle due dita: dove sta la caramella stretta.

    **Non** e' il site ``gripperframe`` (la "punta" del modello): con la pinza
    verticale e' 3 mm piu' avanti e 10 mm di lato, e le dita scendono 13 mm
    sotto di lui (misurato 2026-10-11). Si calcola a ogni posa, non si somma
    un numero fisso: lo spostamento ruota con la pinza.
    """
    d = _qpos_da(model, ctrl)
    return (d.geom("fixed_jaw_sph_tip1").xpos + d.geom("moving_jaw_sph_tip1").xpos) / 2


def postura_dita(
    model: mujoco.MjModel, punto: np.ndarray, inclinazione: float, q_pinza: float,
    passate: int = 4,
) -> Postura:
    """Come ``postura``, ma e' il **centro fra le dita** ad andare in ``punto``.

    Si risolve per la punta, si guarda quanto il centro delle dita e' spostato da
    lei in quella posa, si sposta il bersaglio di altrettanto e si rifa': lo
    spostamento cambia poco fra pose vicine, poche passate bastano.
    """
    punto = np.asarray(punto, dtype=float)
    bersaglio = punto.copy()
    for _ in range(passate):
        p = postura(model, bersaglio, inclinazione)
        cmd = comando(p, 0.0, q_pinza)
        bersaglio = bersaglio + (punto - centro_dita(model, cmd))
    p = postura(model, bersaglio, inclinazione)
    errore = float(np.linalg.norm(punto - centro_dita(model, comando(p, 0.0, q_pinza))))
    return Postura(p.q, errore, p.scarto)
