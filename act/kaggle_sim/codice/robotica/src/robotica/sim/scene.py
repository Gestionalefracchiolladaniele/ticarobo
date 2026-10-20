"""Costruzione della scena: robot da mujoco_menagerie + mondo di lavoro.

La scena si compone in Python con ``MjSpec`` invece che in XML, per due motivi:
i path delle mesh si risolvono correttamente, e si puo' cambiare robot senza
riscrivere il mondo. Lo stesso mondo funziona con Panda, UR5e o SO-101.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from robotica.paths import robot_dir

# Altezza del piano di lavoro. Tutto il resto vi si appoggia sopra.
TABLE_HEIGHT = 0.40
TABLE_CENTER = (0.50, 0.0)

# Per ogni robot: file del modello, body della pinza, offset dal body al TCP.
# Il TCP (Tool Center Point) e' il punto fra le dita che si porta sull'oggetto:
# e' il frame che l'IK deve raggiungere.
# Il TCP sta fra le dita, all'altezza dove il cubo viene stretto: non alla
# punta (scivolerebbe) ne' alla base (urterebbe la mano). Sul Panda le dita
# partono a 0.0584 dal frame mano e sono lunghe ~0.054: il punto di presa e'
# circa a meta'. Su hardware reale corrisponde al TCP configurato nel controller.
_ROBOT_SPEC = {
    "panda": {"xml": "panda.xml", "hand": "hand", "tcp_offset": (0.0, 0.0, 0.085)},
    "fr3": {"xml": "fr3.xml", "hand": "fr3_hand", "tcp_offset": (0.0, 0.0, 0.085)},
    "ur5e": {"xml": "ur5e.xml", "hand": "wrist_3_link", "tcp_offset": (0.0, 0.1, 0.0)},
    # Il SO-101 il TCP lo dichiara gia': e' il site ``gripperframe`` del
    # produttore, e l'offset e' il suo, letto da so101.xml.
    "so101": {"xml": "so101.xml", "hand": "gripper", "tcp_offset": (0.012, -0.000218, -0.098127)},
}


@dataclass
class SceneObject:
    """Un oggetto afferrabile appoggiato sul tavolo."""

    name: str
    pos: tuple[float, float]  # (x, y); la z si deriva dal tavolo
    rgba: tuple[float, float, float, float]
    half_size: float = 0.02
    mass: float = 0.05


DEFAULT_OBJECTS = [
    SceneObject("cube_red", (0.45, -0.15), (0.85, 0.2, 0.2, 1.0)),
    SceneObject("cube_blue", (0.55, 0.12), (0.2, 0.35, 0.85, 1.0)),
]


@dataclass
class SceneConfig:
    """Cosa mettere nella scena."""

    robot: str = "panda"
    objects: list[SceneObject] = field(default_factory=lambda: list(DEFAULT_OBJECTS))
    with_obstacle: bool = True
    with_cameras: bool = True


def build_spec(cfg: SceneConfig | None = None) -> mujoco.MjSpec:
    """Compone robot + mondo e restituisce lo spec non compilato.

    Restituire lo spec invece del modello compilato permette di modificarlo
    ancora (aggiungere oggetti, spostare camere) prima di compilare.
    """
    cfg = cfg or SceneConfig()
    if cfg.robot not in _ROBOT_SPEC:
        raise KeyError(f"robot '{cfg.robot}' non supportato; disponibili: {sorted(_ROBOT_SPEC)}")

    info = _ROBOT_SPEC[cfg.robot]
    spec = mujoco.MjSpec.from_file(str(robot_dir(cfg.robot) / info["xml"]))

    _add_world(spec)
    _add_table(spec)
    if cfg.with_obstacle:
        _add_obstacle(spec)
    for obj in cfg.objects:
        _add_object(spec, obj)
    if cfg.with_cameras:
        _add_cameras(spec, info["hand"])
    _add_tcp_site(spec, info["hand"], info["tcp_offset"])

    return spec


def build(cfg: SceneConfig | None = None) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Costruisce e compila la scena, con i dati gia' portati alla posa iniziale."""
    cfg = cfg or SceneConfig()
    model = build_spec(cfg).compile()
    data = mujoco.MjData(model)
    reset(model, data, cfg)
    return model, data


def reset(model: mujoco.MjModel, data: mujoco.MjData, cfg: SceneConfig | None = None) -> None:
    """Riporta robot e oggetti alla posa iniziale.

    La keyframe del menagerie copre solo i giunti del robot; gli oggetti liberi
    vanno ricollocati a mano, altrimenti partono dall'origine del mondo.
    """
    cfg = cfg or SceneConfig()
    mujoco.mj_resetData(model, data)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    for obj in cfg.objects:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{obj.name}_free")
        if jid < 0:
            continue
        adr = model.jnt_qposadr[jid]
        x, y = obj.pos
        data.qpos[adr : adr + 7] = [x, y, TABLE_HEIGHT + obj.half_size, 1.0, 0.0, 0.0, 0.0]

    mujoco.mj_forward(model, data)


# --- pezzi del mondo ---------------------------------------------------------


def _add_world(spec: mujoco.MjSpec) -> None:
    """Pavimento, luci e cielo. Nulla di funzionale, serve al rendering."""
    spec.add_texture(
        name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.5, 0.6, 0.7], rgb2=[0.1, 0.1, 0.15], width=256, height=256,
    )
    spec.add_texture(
        name="grid_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.25, 0.25, 0.28], rgb2=[0.3, 0.3, 0.33], width=300, height=300,
    )
    mat = spec.add_material(name="grid_mat", texrepeat=[6, 6], reflectance=0.1)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid_tex"

    world = spec.worldbody
    # Illuminazione contenuta: col rendering software le luci si sommano in
    # fretta e la scena si sovraespone, rendendo i colori indistinguibili.
    world.add_light(
        pos=[0.5, 0, 2.0], dir=[0, 0, -1],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        diffuse=[0.35] * 3, specular=[0.1] * 3,
    )
    world.add_light(pos=[0.2, -0.6, 1.5], dir=[0.2, 0.5, -1],
                    diffuse=[0.2] * 3, specular=[0.05] * 3)
    world.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[3, 3, 0.05], material="grid_mat",
    )


def _add_table(spec: mujoco.MjSpec) -> None:
    """Piano di lavoro. La superficie sta a z = TABLE_HEIGHT."""
    spec.add_material(name="table_mat", rgba=[0.75, 0.7, 0.62, 1], reflectance=0.05)
    cx, cy = TABLE_CENTER
    body = spec.worldbody.add_body(name="table", pos=[cx, cy, TABLE_HEIGHT - 0.02])
    body.add_geom(
        name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.35, 0.45, 0.02], material="table_mat", friction=[0.9, 0.01, 0.001],
    )


def _add_obstacle(spec: mujoco.MjSpec) -> None:
    """Ostacolo fisso fra i due cubi: obbliga il planner a girarci attorno."""
    body = spec.worldbody.add_body(name="obstacle", pos=[0.50, -0.02, TABLE_HEIGHT + 0.07])
    body.add_geom(
        name="obstacle_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.10, 0.07], rgba=[0.55, 0.55, 0.6, 1],
    )


def _add_object(spec: mujoco.MjSpec, obj: SceneObject) -> None:
    """Un cubo libero di muoversi, appoggiato sul tavolo."""
    x, y = obj.pos
    body = spec.worldbody.add_body(
        name=obj.name, pos=[x, y, TABLE_HEIGHT + obj.half_size]
    )
    body.add_freejoint(name=f"{obj.name}_free")
    body.add_geom(
        name=f"{obj.name}_geom", type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[obj.half_size] * 3, rgba=list(obj.rgba),
        mass=obj.mass, friction=[1.0, 0.02, 0.001],
    )


def _add_cameras(spec: mujoco.MjSpec, hand_body: str) -> None:
    """Due camere, i due montaggi standard in robotica.

    - overhead (eye-to-hand): fissa sopra il tavolo. Vede tutta la scena ma
      non sa dove sia il robot finche' non si calibra.
    - wrist (eye-in-hand): solidale alla mano. Si muove col braccio, quindi la
      sua pose si ricava sempre dalla cinematica.
    """
    cx, cy = TABLE_CENTER
    mount = spec.worldbody.add_body(name="overhead_mount", pos=[cx, cy, 1.10])
    mount.add_geom(
        name="cam_body", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.03, 0.02],
        rgba=[0.2, 0.2, 0.2, 1], contype=0, conaffinity=0,
    )
    # Una camera MuJoCo guarda lungo il proprio -Z. Con quaternione identita'
    # il suo -Z coincide col -Z del mondo: guarda verso il basso, cioe' il tavolo.
    mount.add_camera(name="overhead", fovy=50, quat=[1.0, 0.0, 0.0, 0.0])

    # La camera al polso eredita l'orientamento della mano, che durante una
    # presa punta gia' verso il basso: nessuna rotazione aggiuntiva.
    hand = spec.body(hand_body)
    if hand is not None:
        hand.add_camera(name="wrist", pos=[0.06, 0.0, 0.02], fovy=60,
                        quat=[1.0, 0.0, 0.0, 0.0])


def _add_tcp_site(spec: mujoco.MjSpec, hand_body: str, offset: tuple) -> None:
    """Site TCP sulla pinza: il frame che l'IK deve portare sul bersaglio.

    I modelli menagerie non lo definiscono, perche' dipende dall'utensile
    montato. Su hardware reale corrisponde al TCP configurato nel controller.
    """
    hand = spec.body(hand_body)
    if hand is None:
        raise KeyError(f"body '{hand_body}' non trovato nel modello")
    hand.add_site(name="tcp", pos=list(offset), size=[0.006] * 3,
                  rgba=[1.0, 0.85, 0.1, 0.6])


def table_surface_z() -> float:
    """Quota della superficie del tavolo, in metri."""
    return TABLE_HEIGHT


def object_names(model: mujoco.MjModel) -> list[str]:
    """Nomi dei body che hanno un freejoint: gli oggetti manipolabili."""
    names = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and model.body_jntnum[i] == 1:
            jnt = model.body_jntadr[i]
            if model.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE:
                names.append(name)
    return names


def body_pose(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    """Pose 4x4 di un body nel frame mondo."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(f"body '{name}' non trovato")
    T = np.eye(4)
    T[:3, :3] = data.xmat[bid].reshape(3, 3)
    T[:3, 3] = data.xpos[bid]
    return T
