"""La scena delle caramelle per ACT: il SO-101 vero rifatto in simulazione.

Serve a **generare dimostrazioni** da mescolare ai 50 episodi veri
(co-training, esperimenti/so101/act.md). Non deve sembrare vera: deve
**stare dove sta il vero** (camera, tavolo, bicchieri) e **variare** in tutto
il resto.

Ogni quota viene dai dati, non da un righello:

- **tavolo a −30 mm** nel frame del modello: il punto piu' basso delle dita
  alle 227 prese dei 50 episodi (mediana). Non e' lo z = 0 del modello: gli
  angoli LeRobot, letti nel modello, mettono il tavolo vero piu' in basso
  (D25 lo aveva visto a mano: −27 mm);
- **bicchieri dove la pinza si e' aperta**: mediana dei rilasci, per lato;
- **larghezza delle caramelle dalla pinza**: in tenuta il follower si ferma
  sulla caramella, e l'angolo letto e' la sua larghezza (14 e 16,6 mm). Le
  altre misure sono stime dalle foto, con quella larghezza come righello;
- **campo visivo 60,6°**: calcolato dalla scheda della camera (OV9732, 100°
  in diagonale, 16:9 ritagliato a 4:3), verificato contro i fotogrammi veri.

La forma da **vedere** e' una mesh di Blender (``assets/_caramelle.py``), la
forma da **toccare** una primitiva di MuJoCo con le stesse misure: l'involucro
convesso di un bicchiere lo riempirebbe, e una caramella resterebbe sopra.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from robotica.paths import SCENES, robot_dir
from robotica.sim.so101_presa import ASSE_BASE

#: Dove stanno le mesh e le texture: versionate, perche' Kaggle non ha Blender.
MESH = SCENES / "so101_caramelle"

#: Quota del tavolo vero nel frame del modello, e quanto la si varia.
TAVOLO = -0.030
TAVOLO_VARIA = 0.005
#: Largo 2 m: con 1 m, a polso girato, la camera vedeva il bordo e il cielo.
LATO_TAVOLO = 2.0

#: Centro del fondo dei bicchieri (x, y), mediana dei rilasci; e quanto variano.
#: Sinistra = +y: li' si rilasciano le caramelle larghe 14 mm, le gialle.
BICCHIERI = {"sinistra": np.array([0.128, 0.163]), "destra": np.array([0.170, -0.127])}
BICCHIERI_VARIA = 0.015

#: Dove stanno le caramelle: 5°-95° percentile delle prese vere.
ZONA_X = (0.15, 0.30)
ZONA_Y = (-0.08, 0.08)
ZONA_RAGGIO = (0.12, 0.27)  # dall'asse della base
DISTANZA_MINIMA = 0.035  # sotto, due caramelle sono una presa sola
# 8 cm dal centro: il 95% delle prese vere sta oltre 8,2 cm (solo 10 su 227
# sotto gli 8). Piu' vicino il polso scendendo urta il bordo, fisso qui.
LONTANO_DAI_BICCHIERI = 0.08

#: Quante per episodio: 136 + 91 prese su 50 episodi, 2,7 gialle e 1,8 galatine.
QUANTE = {"gialla": (2, 4), "galatine": (1, 3), "rossana": (0, 1)}
ROSSANA_QUASI_SEMPRE = 0.85

#: Dove va ciascuna (regole di D26). La rossana non si tocca.
DESTINAZIONE = {"gialla": "sinistra", "galatine": "destra", "rossana": None}

#: Camera sul polso: verticale calcolato dalla scheda (OV9732, 100° diagonali,
#: 16:9 ritagliato a 4:3); varia per coprire il barilotto e la camera che si
#: sposta nel supporto (successo davvero, act.md).
FOVY = 60.6
FOVY_VARIA = 3.0
CAMERA_SPOSTA = 0.003  # metri, per episodio
CAMERA_RUOTA = np.radians(2.0)
SCALA_VARIA = 0.10  # dimensioni delle caramelle

MASSA = {"gialla": 0.005, "galatine": 0.006, "rossana": 0.008}  # kg, stima

#: I due corpi delle dita nel modello: ganascia fissa e mobile.
DITA = ("gripper", "moving_jaw_so101_v1")
ATTRITO = [1.0, 0.01, 0.0005]

#: Metri. ``corpo`` della gialla e della galatine: (x, y, z) con x il lato
#: lungo; della rossana: (lunghezza, diametro). y e' la larghezza che la pinza
#: stringe: e' quella misurata.
MISURE: dict[str, dict] = {
    "gialla": {"corpo": (0.026, 0.014, 0.012), "ala": 0.010},
    "galatine": {"corpo": (0.028, 0.0166, 0.008), "saldatura": 0.003},
    "rossana": {"corpo": (0.026, 0.013), "ala": 0.009},
    # Stima dai dati: al rilascio piu' basso (5° percentile) il centro delle dita
    # e' 8,8 cm sopra il tavolo e la caramella pende qualche mm sotto: il bordo
    # sta sotto gli ~8 cm. Il diametro e' una stima da foto, non misurato.
    "bicchiere": {
        "raggio_fondo": 0.025,
        "raggio_bordo": 0.0375,
        "altezza": 0.080,
        "spessore": 0.0012,
    },
}


@dataclass(frozen=True)
class Caramella:
    corpo: str  # nome del corpo nel modello: "gialla_0"
    tipo: str  # "gialla" | "galatine" | "rossana"
    destinazione: str | None  # "sinistra" | "destra" | None: non si tocca
    scala: float = 1.0  # sulle MISURE: ogni caramella e' un po' diversa

    @property
    def larghezza(self) -> float:
        """Il lato che la pinza stringe, metri: e' quello misurato dai dati."""
        return MISURE[self.tipo]["corpo"][1] * self.scala


@dataclass(frozen=True)
class Episodio:
    """Cio' che l'esperto sa e la rete no: la verita' della scena."""

    caramelle: tuple[Caramella, ...]
    bicchieri: dict[str, np.ndarray]  # centro del fondo (x, y, z)
    tavolo: float


def _quat_z(angolo: float) -> list[float]:
    return [float(np.cos(angolo / 2)), 0.0, 0.0, float(np.sin(angolo / 2))]


def _posizioni(
    rng: np.random.Generator, n: int, bicchieri: dict[str, np.ndarray]
) -> list[np.ndarray]:
    """``n`` punti nella zona delle prese vere, lontani fra loro e dai bicchieri."""
    punti: list[np.ndarray] = []
    for _ in range(10_000):
        if len(punti) == n:
            return punti
        p = np.array([rng.uniform(*ZONA_X), rng.uniform(*ZONA_Y)])
        if not ZONA_RAGGIO[0] <= np.linalg.norm(p - ASSE_BASE) <= ZONA_RAGGIO[1]:
            continue
        if any(np.linalg.norm(p - b[:2]) < LONTANO_DAI_BICCHIERI for b in bicchieri.values()):
            continue
        if any(np.linalg.norm(p - q) < DISTANZA_MINIMA for q in punti):
            continue
        punti.append(p)
    raise RuntimeError(f"non trovo posto per {n} caramelle: zona troppo piccola")


def _aspetto(spec: mujoco.MjSpec, rng: np.random.Generator) -> None:
    """Tovaglia (meta' delle volte la foto vera), luce, texture delle caramelle."""
    spec.add_texture(
        name="tovaglia", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(MESH / "tovaglia.png")
    )
    tinta = rng.uniform(0.3, 1.0, size=3)
    spec.add_texture(
        name="a_caso",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=int(
            rng.choice([mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mujoco.mjtBuiltin.mjBUILTIN_FLAT])
        ),
        rgb1=tinta.tolist(),
        rgb2=(tinta * rng.uniform(0.6, 0.95)).tolist(),
        width=128,
        height=128,
    )
    vera = rng.random() < 0.5
    # Il ritaglio della foto e' ~7 cm di tovaglia; il piano e' largo LATO_TAVOLO.
    lato = 0.07 * rng.uniform(0.8, 1.2) if vera else rng.uniform(0.02, 0.2)
    tav = spec.add_material(name="tavolo", texrepeat=[LATO_TAVOLO / lato, LATO_TAVOLO / lato])
    tav.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "tovaglia" if vera else "a_caso"
    tav.rgba = [*rng.uniform(0.85, 1.0, size=3), 1]
    for nome in ("strisce", "blu"):
        spec.add_texture(
            name=nome, type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(MESH / f"{nome}.png")
        )
        mat = spec.add_material(name=f"galatine_{nome}")
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = nome

    # Il modello e' giallo, il braccio vero bianco: e le dita sono in ogni
    # fotogramma. Lasciate gialle, separerebbero da sole i due mondi.
    # E in ombra sotto il braccio venivano grigie: luminosita' media delle dita
    # 135 contro 168 dei fotogrammi veri (8 viste, stessi pixel). Con
    # un'emissione di ~0,15 coincidono (0 -> 135, 0,2 -> 178) a luce fissa; con le
    # luci variate serve ~0,2 (misurato: 0,08-0,22 dava 150 contro 171).
    bianco = rng.uniform(0.86, 0.97)
    emissione = rng.uniform(0.12, 0.28)
    for mat in spec.materials:
        if np.allclose(mat.rgba[:3], [1, 0.82, 0.12], atol=0.05):
            mat.rgba = [bianco, bianco, bianco * rng.uniform(0.97, 1.0), 1]
            mat.emission = emissione

    for luce in spec.worldbody.lights:
        luce.diffuse = [rng.uniform(0.2, 0.5)] * 3
        luce.dir = [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), -1.0]
    spec.visual.headlight.diffuse = [rng.uniform(0.2, 0.45)] * 3
    # Le dita vere sono bianche brillanti: vicine alla lente, prendono luce.
    spec.visual.headlight.ambient = [rng.uniform(0.3, 0.45)] * 3


def _camera(spec: mujoco.MjSpec, rng: np.random.Generator) -> None:
    cam = spec.camera("wrist_cam")
    cam.sensor_size = [0, 0]  # altrimenti MuJoCo ignora fovy
    cam.focal_length = [0, 0]
    cam.resolution = [640, 480]
    cam.fovy = FOVY + rng.uniform(-FOVY_VARIA, FOVY_VARIA)
    cam.pos = (np.asarray(cam.pos) + rng.uniform(-CAMERA_SPOSTA, CAMERA_SPOSTA, 3)).tolist()
    cam.alt.euler = (
        np.asarray(cam.alt.euler) + rng.uniform(-CAMERA_RUOTA, CAMERA_RUOTA, 3)
    ).tolist()


def _bicchiere(spec: mujoco.MjSpec, nome: str, centro: np.ndarray) -> None:
    """Mesh da vedere, e pareti da toccare: lastrine inclinate come il cono."""
    m = MISURE["bicchiere"]
    rb, rt, h = m["raggio_fondo"], m["raggio_bordo"], m["altezza"]
    b = spec.worldbody.add_body(name=f"bicchiere_{nome}", pos=centro.tolist())
    b.add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="bicchiere",
        pos=[0, 0, h / 2],
        rgba=[0.85, 0.9, 0.95, 0.3],
        contype=0,
        conaffinity=0,
        group=1,
    )
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[rb, 0.001, 0], pos=[0, 0, 0.001], group=3)
    n, r_medio = 16, (rb + rt) / 2
    pendenza = np.arctan2(rt - rb, h)
    for k in range(n):
        a = 2 * np.pi * k / n
        q = np.zeros(4)
        mujoco.mju_mulQuat(
            q, np.array(_quat_z(a)), np.array([np.cos(pendenza / 2), 0, np.sin(pendenza / 2), 0])
        )
        b.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.001, np.pi * r_medio / n * 1.15, h / 2],
            pos=[r_medio * np.cos(a), r_medio * np.sin(a), h / 2],
            quat=q.tolist(),
            group=3,
        )


def _caramella(
    spec: mujoco.MjSpec, c: Caramella, xy: np.ndarray, tavolo: float, rng: np.random.Generator
) -> None:
    s = c.scala
    corpo = MISURE[c.tipo]["corpo"]
    mezza_altezza = (corpo[2] if c.tipo != "rossana" else corpo[1]) / 2 * s
    b = spec.worldbody.add_body(
        name=c.corpo,
        pos=[float(xy[0]), float(xy[1]), tavolo + mezza_altezza + 0.001],
        quat=_quat_z(rng.uniform(-np.pi, np.pi)),
    )
    b.add_freejoint()
    spec.add_mesh(name=f"{c.corpo}_mesh", file=str(MESH / f"{c.tipo}.obj"), scale=[s, s, s])
    vista = b.add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=f"{c.corpo}_mesh",
        contype=0,
        conaffinity=0,
        group=1,
        mass=0,
    )
    if c.tipo == "gialla":
        tinta = np.array([0.95, 0.68, 0.12]) * rng.uniform(0.85, 1.05, 3)
        vista.rgba = [*np.clip(tinta, 0, 1), 1]
        tocco = dict(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=(np.array(corpo) / 2 * s).tolist())
    elif c.tipo == "galatine":
        vista.material = f"galatine_{rng.choice(['strisce', 'blu'])}"
        tocco = dict(type=mujoco.mjtGeom.mjGEOM_BOX, size=(np.array(corpo) / 2 * s).tolist())
    else:
        vista.rgba = [0.62 * rng.uniform(0.85, 1.1), 0.05, 0.09, 1]
        lungo, diametro = corpo
        tocco = dict(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[diametro / 2 * s, lungo / 2 * s, 0],
            quat=[np.cos(np.pi / 4), 0, np.sin(np.pi / 4), 0],
        )
    b.add_geom(**tocco, group=3, mass=MASSA[c.tipo], friction=ATTRITO, condim=4)
    # L'aggancio, spento: lo accende l'esperto quando **tutte e due** le dita
    # toccano la caramella. Stretta fra due punte inclinate, una caramella
    # rotonda schizza via: farla tenere al contatto vorrebbe dire tarare
    # attriti e rigidezze (CLAUDE.md). ACT impara da immagini e angoli, e
    # quelli l'aggancio li da' giusti; una presa mancata resta mancata.
    spec.add_equality(
        name=f"presa_{c.corpo}",
        type=mujoco.mjtEq.mjEQ_WELD,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        name1=DITA[0],
        name2=c.corpo,
        active=False,
    )


def costruisci(rng: np.random.Generator) -> tuple[mujoco.MjModel, mujoco.MjData, Episodio]:
    """Un episodio nuovo: il braccio del produttore e, intorno, una scena estratta a caso."""
    spec = mujoco.MjSpec.from_file(str(robot_dir("so101") / "scene.xml"))
    tavolo = TAVOLO + rng.uniform(-TAVOLO_VARIA, TAVOLO_VARIA)
    _aspetto(spec, rng)
    _camera(spec, rng)
    for g in spec.worldbody.geoms:
        if g.name == "floor":
            g.pos = [0, 0, tavolo]
            g.size = [LATO_TAVOLO / 2, LATO_TAVOLO / 2, 0.05]
            g.material = "tavolo"
    # La base vera appoggia sul tavolo; nel modello sta a z = 0, 3 cm sopra.
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.05, 0.05, -tavolo / 2],
        pos=[0.0, 0.0, tavolo / 2],
        rgba=[0.2, 0.2, 0.22, 1],
    )

    spec.add_mesh(name="bicchiere", file=str(MESH / "bicchiere.obj"))
    bicchieri = {}
    for lato, xy in BICCHIERI.items():
        c = xy + rng.uniform(-BICCHIERI_VARIA, BICCHIERI_VARIA, 2)
        bicchieri[lato] = np.array([c[0], c[1], tavolo])
        _bicchiere(spec, lato, bicchieri[lato])

    tipi: list[str] = []
    for tipo, (lo, hi) in QUANTE.items():
        k = (
            int(rng.integers(lo, hi + 1))
            if tipo != "rossana"
            else int(rng.random() < ROSSANA_QUASI_SEMPRE)
        )
        tipi += [tipo] * k
    caramelle = tuple(
        Caramella(f"{t}_{i}", t, DESTINAZIONE[t], 1 + float(rng.uniform(-SCALA_VARIA, SCALA_VARIA)))
        for i, t in enumerate(tipi)
    )
    for c, xy in zip(caramelle, _posizioni(rng, len(caramelle), bicchieri), strict=True):
        _caramella(spec, c, xy, tavolo, rng)

    model = spec.compile()
    return model, mujoco.MjData(model), Episodio(caramelle, bicchieri, tavolo)


def genera_mesh(dest: Path = MESH) -> list[Path]:
    """Rifa' le mesh con Blender. Si lancia solo se cambiano le ``MISURE``."""
    from robotica.assets.blender import crea_caramelle

    return crea_caramelle(MISURE, dest)


if __name__ == "__main__":
    for f in genera_mesh():
        print(f)
