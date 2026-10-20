"""La scena del sasso-carta-forbici: il SO-101, la camera fissa, la mano.

Il modello del braccio e' quello del produttore (``robotstudio_so101``) e **non
si modifica**: la scena ci si costruisce intorno, con ``MjSpec``.

Gli attuatori del SO-101 sono di **posizione**: dentro ogni servo c'e' un
anello che porta il giunto all'angolo richiesto. Quindi un gesto non e' una
traiettoria da controllare ma **sei numeri da scrivere in** ``data.ctrl``. E'
anche cio' che trasferisce meglio: sul braccio vero gli stessi sei numeri
vanno ai servi Feetech, e il firmware fa il resto.


I gesti si calcolano, non si tarano
-----------------------------------

Si dichiarano tre grandezze fisiche e gli angoli ne discendono:

- **dove** si mostra il gesto (``TCP_GESTO``), con il braccio piegato;
- **verso dove** punta la pinza: orizzontale, verso il giocatore (+x);
- **quanto** si aprono le dita (``APERTURA_DITA``).

Letto dal modello (2026-09-24): a giunti zero il braccio e' **steso** lungo +x,
le dita si allungano lungo il -z della pinza e si **separano lungo la sua x**,
che a ``wrist_roll = 0`` e' verticale. Quindi:

- **sasso**: pinza chiusa
- **forbici**: aperta, dita una sopra l'altra (``wrist_roll = 0``): la V
- **carta**: aperta, dita affiancate (``wrist_roll = -pi/2``): la mano distesa

Quale orientamento sia la carta e' una **convenzione**, non geometria. Il
documento partiva dal contrario; invertito il 2026-09-24 su scelta del
progetto. Conseguenza misurata: la rotazione di 90° ora la paga la carta
(2,5 s dal tasto al gesto), le forbici scendono a 1,9 s.

**-pi/2, non +pi/2** (2026-10-10): per chi guarda e' la stessa posa a
specchio, ma sul **braccio vero** +pi/2 e' +191,5 gradi LeRobot, oltre il
cavo del polso e oltre +-180, dove il numero del motore salta. -pi/2 e'
+11,5 gradi: dentro. ``safety/`` guarda il modello (+-157 gradi) e la
approvava; l'ha fermata l'esecutore, che guarda il ferro (D23).

Il gesto non si fa a giunti zero: steso, il braccio e' in configurazione
**singolare** e non si muove piu' bene (D17).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from robotica.games.rps.gioco import Mossa
from robotica.paths import robot_dir
from robotica.transforms import camera_xyaxes

#: Ordine di ``data.ctrl``: e' l'ordine degli attuatori in so101.xml.
GIUNTI: tuple[str, ...] = (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
)

# --- quote dichiarate (metri, radianti) --------------------------------------

#: Dove sta il TCP mentre il robot mostra il gesto. Piu' vicino della portata
#: a braccio steso (0,39 m a z 0,25): la differenza e' la piegatura.
TCP_GESTO = np.array([0.30, 0.0, 0.22])

#: Dove il giocatore mostra la mano. Davanti al robot, alla stessa altezza.
ZONA_MANO = np.array([0.58, 0.0, 0.22])
ZONA_MANO_MEZZA = np.array([0.05, 0.09, 0.09])

#: Distanza fra le punte delle dita nei gesti aperti: si deve vedere da 1-2 m.
APERTURA_DITA = 0.06

#: Ampiezza del battito del pugno nel conto alla rovescia, sul polso.
BATTITO = 0.35

#: La camera fissa (eye-to-hand) sta su un treppiede di lato al giocatore e
#: guarda **a meta' fra la mano e la pinza**: deve vedere la mano per
#: riconoscerla, e il robot per il video. L'orientamento non si dichiara: si
#: ricava da queste due posizioni (``camera_xyaxes``).
CAMERA_POS = np.array([0.70, -0.55, 0.50])
CAMERA_GUARDA = (TCP_GESTO + ZONA_MANO) / 2
CAMERA_FOVY = 50.0


@dataclass(frozen=True)
class Gesti:
    """I comandi (sei numeri, ordine ``GIUNTI``) che il braccio sa fare."""

    pronto: np.ndarray  # pugno chiuso, in posizione: si parte e si torna qui
    battuta: np.ndarray  # il pugno alzato del conto alla rovescia
    sasso: np.ndarray
    carta: np.ndarray
    forbici: np.ndarray

    def di(self, mossa: Mossa) -> np.ndarray:
        return {Mossa.SASSO: self.sasso, Mossa.CARTA: self.carta, Mossa.FORBICI: self.forbici}[
            mossa
        ]

    def tutti(self) -> dict[str, np.ndarray]:
        return {
            "pronto": self.pronto, "battuta": self.battuta,
            "sasso": self.sasso, "carta": self.carta, "forbici": self.forbici,
        }


# --- scena --------------------------------------------------------------------


def costruisci() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Braccio del produttore + camera fissa + zona della mano, braccio in ``pronto``."""
    spec = mujoco.MjSpec.from_file(str(robot_dir("so101") / "scene.xml"))
    world = spec.worldbody

    # La zona della mano e' solo un segno: niente collisioni, niente massa.
    zona = world.add_body(name="zona_mano", pos=ZONA_MANO.tolist())
    zona.add_geom(
        name="zona_mano", type=mujoco.mjtGeom.mjGEOM_BOX, size=ZONA_MANO_MEZZA.tolist(),
        rgba=[0.95, 0.75, 0.6, 0.25], contype=0, conaffinity=0,
    )

    # Il treppiede: un'asta fino alla camera, per vedere dove sta.
    treppiede = world.add_body(name="treppiede", pos=CAMERA_POS.tolist())
    treppiede.add_geom(
        # MjSpec vuole sempre tre numeri: il cilindro usa (raggio, mezza altezza).
        type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.008, CAMERA_POS[2] / 2, 0.0],
        pos=[0, 0, -CAMERA_POS[2] / 2], rgba=[0.2, 0.2, 0.2, 1], contype=0, conaffinity=0,
    )
    treppiede.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, 0.015],
        rgba=[0.1, 0.1, 0.1, 1], contype=0, conaffinity=0,
    )
    cam = treppiede.add_camera(name="fissa", fovy=CAMERA_FOVY)
    cam.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
    cam.alt.xyaxes = camera_xyaxes(CAMERA_POS, CAMERA_GUARDA)

    model = spec.compile()
    data = mujoco.MjData(model)
    gesti = calcola_gesti(model)
    porta_a(model, data, gesti.pronto)
    return model, data


def porta_a(model: mujoco.MjModel, data: mujoco.MjData, ctrl: np.ndarray) -> None:
    """Mette il braccio **gia'** nella posa (qpos e ctrl), senza simulare il moto.

    Serve solo all'avvio: partire da giunti zero vorrebbe dire partire steso,
    e il primo secondo sarebbe il braccio che crolla sotto il proprio peso.
    """
    for nome, q in zip(GIUNTI, ctrl, strict=True):
        data.qpos[model.joint(nome).qposadr[0]] = q
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)


# --- gesti --------------------------------------------------------------------


def calcola_gesti(model: mujoco.MjModel) -> Gesti:
    """I cinque comandi, ricavati dalle quote dichiarate in cima al modulo."""
    lift, elbow, wflex = _postura_planare(model, TCP_GESTO)
    chiusa, aperta = _apertura_pinza(model, APERTURA_DITA)

    def ctrl(roll: float, pinza: float, dflex: float = 0.0) -> np.ndarray:
        return np.array([0.0, lift, elbow, wflex + dflex, roll, pinza])

    return Gesti(
        pronto=ctrl(0.0, chiusa),
        battuta=ctrl(0.0, chiusa, dflex=_segno_su(model, ctrl(0.0, chiusa)) * BATTITO),
        sasso=ctrl(0.0, chiusa),
        carta=ctrl(-np.pi / 2, aperta),
        forbici=ctrl(0.0, aperta),
    )


def _qpos_da(
    model: mujoco.MjModel, ctrl: np.ndarray, d: mujoco.MjData | None = None
) -> mujoco.MjData:
    """Cinematica per i giunti ``ctrl``. ``d`` si riusa nei cicli: creare un
    ``MjData`` alloca lo stato di tutto il modello, e costa."""
    d = d if d is not None else mujoco.MjData(model)
    for nome, q in zip(GIUNTI, ctrl, strict=True):
        d.qpos[model.joint(nome).qposadr[0]] = q
    mujoco.mj_kinematics(model, d)
    return d


def _beccheggio(d: mujoco.MjData) -> float:
    """Di quanto la pinza punta sotto l'orizzonte, ruotando attorno a +y.

    Ruotare +x di un angolo t attorno a +y da' (cos t, 0, -sin t): quindi
    t = atan2(-p_z, p_x), zero quando la pinza e' orizzontale verso +x.
    """
    p = -d.body("gripper").xmat.reshape(3, 3)[:, 2]  # le dita puntano lungo -z pinza
    return float(np.arctan2(-p[2], p[0]))


def _postura_planare(
    model: mujoco.MjModel, tcp: np.ndarray, iterazioni: int = 200
) -> tuple[float, float, float]:
    """``shoulder_lift``, ``elbow_flex``, ``wrist_flex`` per TCP in ``tcp`` e pinza orizzontale.

    I tre giunti ruotano tutti attorno a y: e' un braccio **planare**, e tre
    incognite contro tre richieste (x, z del TCP e beccheggio) hanno soluzione
    finita. Si risolve con i minimi quadrati smorzati sullo jacobiano di
    MuJoCo: lo jacobiano dice di quanto si sposta la punta per un piccolo
    movimento di ogni giunto, e si ripetono piccoli passi verso il bersaglio.

    Si parte **piegati**, non da zero: a braccio steso lo jacobiano perde rango
    (e' la singolarita') e il primo passo sarebbe casuale.
    """
    idx = [model.joint(n).qposadr[0] for n in GIUNTI[1:4]]
    dof = [model.joint(n).dofadr[0] for n in GIUNTI[1:4]]
    lo = np.array([model.joint(n).range[0] for n in GIUNTI[1:4]])
    hi = np.array([model.joint(n).range[1] for n in GIUNTI[1:4]])
    sid = model.site("gripperframe").id

    d = mujoco.MjData(model)
    q = np.array([-0.8, 0.8, 0.0])  # piegato: vedi la docstring
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(iterazioni):
        d.qpos[idx] = q
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        pos = d.site_xpos[sid]
        err = np.array([tcp[0] - pos[0], tcp[2] - pos[2], 0.0 - _beccheggio(d)])
        if np.linalg.norm(err) < 1e-6:
            break
        mujoco.mj_jacSite(model, d, jacp, jacr, sid)
        J = np.vstack([jacp[0, dof], jacp[2, dof], jacr[1, dof]])
        q = np.clip(q + np.linalg.solve(J.T @ J + 1e-4 * np.eye(3), J.T @ err), lo, hi)

    d.qpos[idx] = q
    mujoco.mj_kinematics(model, d)
    residuo = np.linalg.norm(d.site_xpos[sid][[0, 2]] - tcp[[0, 2]])
    if residuo > 1e-3 or abs(_beccheggio(d)) > 1e-2:
        raise ValueError(
            f"postura non trovata per TCP {tcp}: residuo {residuo * 1000:.1f} mm, "
            f"beccheggio {np.degrees(_beccheggio(d)):.1f} gradi. "
            "Il bersaglio e' fuori portata o chiede un giunto oltre il limite."
        )
    return float(q[0]), float(q[1]), float(q[2])


def distanza_punte(
    model: mujoco.MjModel, q_pinza: float, d: mujoco.MjData | None = None
) -> float:
    """Distanza fra le punte delle due dita, in metri, per un angolo della pinza.

    Le punte sono le sfere ``*_sph_tip1`` del modello: il produttore le ha messe
    proprio li'. Solo cinematica, niente contatti.
    """
    d = _qpos_da(model, np.array([0, 0, 0, 0, 0, q_pinza]), d)
    a = d.geom("fixed_jaw_sph_tip1").xpos
    b = d.geom("moving_jaw_sph_tip1").xpos
    return float(np.linalg.norm(a - b))


def _apertura_pinza(model: mujoco.MjModel, apertura: float) -> tuple[float, float]:
    """(angolo chiusa, angolo aperta di ``apertura``), **misurati** sul modello.

    Chiusa e' dove le punte sono piu' vicine; aperta e' il primo angolo, da li'
    in su, dove la distanza raggiunge ``apertura``.
    """
    lo, hi = model.joint("gripper").range
    angoli = np.linspace(lo, hi, 800)
    d = mujoco.MjData(model)
    dist = np.array([distanza_punte(model, a, d) for a in angoli])
    i_chiusa = int(np.argmin(dist))
    oltre = np.nonzero(dist[i_chiusa:] >= apertura)[0]
    if not len(oltre):
        raise ValueError(f"la pinza non si apre di {apertura * 1000:.0f} mm")
    return float(angoli[i_chiusa]), float(angoli[i_chiusa + oltre[0]])


def _segno_su(model: mujoco.MjModel, posa: np.ndarray) -> float:
    """Il segno di ``wrist_flex`` che **alza** la punta, letto dal modello.

    L'asse del giunto dipende da come e' orientato il body nel modello: invece
    di indovinarlo si prova un piccolo passo e si guarda dove va la punta.
    """
    passo = np.zeros(len(GIUNTI))
    passo[GIUNTI.index("wrist_flex")] = 0.05
    return 1.0 if tcp_di(model, posa + passo)[2] > tcp_di(model, posa)[2] else -1.0


def tcp_di(model: mujoco.MjModel, ctrl: np.ndarray) -> np.ndarray:
    """Dove starebbe il TCP se i giunti fossero esattamente a ``ctrl``."""
    return _qpos_da(model, ctrl).site("gripperframe").xpos.copy()


#: Velocita' dei servi STS3215 a vuoto: 0,238 s per 60 gradi (datasheet, vedi
#: il documento). Se ne usa la meta': sotto carico e a 12 V scarichi rallenta.
VELOCITA_SERVO = np.radians(60) / 0.238
VELOCITA_USATA = VELOCITA_SERVO / 2

#: Quanto resta fermo il gesto perche' si veda l'esito.
TENUTA_ESITO = 1.8


def durata(da: np.ndarray, a: np.ndarray) -> float:
    """Il tempo per andare da ``da`` ad ``a`` senza chiedere ai servi piu' di quanto fanno.

    **Si calcola, non si sceglie.** Con lo smoothstep la velocita' di picco e'
    1,5 volte la media, quindi il giunto che si muove di piu' detta il tempo:
    ``1,5 * max|delta| / velocita'``. Un tempo scelto a occhio in simulazione
    diventerebbe, sul braccio vero, un gesto che il servo non riesce a seguire.
    """
    return max(1.5 * float(np.max(np.abs(np.asarray(a) - np.asarray(da)))) / VELOCITA_USATA, 0.05)


def coreografia(gesti: Gesti, mossa: Mossa) -> list[tuple[np.ndarray, float, str]]:
    """Una mano, come tappe ``(comando, durata, etichetta)``, a partire da ``pronto``.

    Tre battute del pugno, come fa una persona; poi il gesto, fermo quanto
    basta a leggere l'esito; poi il ritorno. L'etichetta dice a chi disegna
    cosa sta succedendo, senza che debba conoscere i numeri.
    """
    su = durata(gesti.pronto, gesti.battuta)
    tappe: list[tuple[np.ndarray, float, str]] = []
    for conto in ("tre", "due", "uno"):
        tappe += [(gesti.battuta, su, conto), (gesti.pronto, su, conto)]
    gesto = gesti.di(mossa)
    va = durata(gesti.pronto, gesto)
    tappe += [(gesto, va, "via"), (gesto, TENUTA_ESITO, "esito"), (gesti.pronto, va, "torna")]
    return tappe


def interpola(da: np.ndarray, a: np.ndarray, s: float) -> np.ndarray:
    """Fra due comandi, con partenza e arrivo dolci (smoothstep).

    Un salto secco su un servo di posizione con kp ~1000 e' uno strattone:
    in simulazione si vede, sul braccio vero si sente. La curva 3s^2 - 2s^3
    ha velocita' zero ai due estremi.
    """
    s = float(np.clip(s, 0.0, 1.0))
    s = s * s * (3 - 2 * s)
    return da + (a - da) * s
