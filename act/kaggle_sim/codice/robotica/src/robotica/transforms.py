"""Pose e trasformazioni rigide.

Una pose e' una matrice 4x4 omogenea che risponde a: "dove si trova il frame B
visto dal frame A?". Si scrive ``T_a_b``. Componendo, gli indici si annullano:

    T_world_gripper = T_world_base @ T_base_gripper

Quando gli indici non si annullano, c'e' un bug. E' la fonte numero uno di
errori in robotica, per questo il nome della variabile porta i frame.

Convenzione quaternione: **wxyz** (come MuJoCo), non xyzw (come scipy).
Le funzioni di conversione qui sotto fanno da ponte.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# Tipo di comodo: matrice omogenea 4x4.
Pose = np.ndarray


def identity() -> Pose:
    """Pose identita': nessuna traslazione, nessuna rotazione."""
    return np.eye(4)


def from_pos_quat(pos: np.ndarray, quat_wxyz: np.ndarray) -> Pose:
    """Costruisce una 4x4 da posizione xyz e quaternione wxyz."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]]).as_matrix()
    T[:3, 3] = pos
    return T


def to_pos_quat(T: Pose) -> tuple[np.ndarray, np.ndarray]:
    """Scompone una 4x4 in (posizione xyz, quaternione wxyz)."""
    quat_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    return T[:3, 3].copy(), quat_xyzw[[3, 0, 1, 2]]


def from_pos_rpy(pos: np.ndarray, rpy: np.ndarray) -> Pose:
    """Costruisce una 4x4 da posizione e angoli roll-pitch-yaw in radianti."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = pos
    return T


def inverse(T: Pose) -> Pose:
    """Inversa di una trasformazione rigida.

    Non usa ``np.linalg.inv``: per una matrice rigida l'inversa e' analitica
    (R trasposta, traslazione ruotata), piu' veloce e numericamente stabile.
    """
    R = T[:3, :3]
    inv = np.eye(4)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ T[:3, 3]
    return inv


def translation(T: Pose) -> np.ndarray:
    """Solo la parte di posizione."""
    return T[:3, 3].copy()


def position_error(T_a: Pose, T_b: Pose) -> float:
    """Distanza in metri fra le origini di due pose."""
    return float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))


def rotation_error(T_a: Pose, T_b: Pose) -> float:
    """Angolo in radianti fra due orientamenti (rotazione minima che li allinea)."""
    R_rel = T_a[:3, :3].T @ T_b[:3, :3]
    return float(np.linalg.norm(Rotation.from_matrix(R_rel).as_rotvec()))


def look_down(pos: np.ndarray, yaw: float = 0.0) -> Pose:
    """Pose con la pinza puntata verso il basso (-Z del mondo).

    E' l'orientamento piu' usato per prendere oggetti da un tavolo: la pinza
    scende verticalmente. ``yaw`` ruota attorno all'asse verticale, per
    allinearsi a un oggetto orientato diversamente.
    """
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [np.pi, 0.0, yaw]).as_matrix()
    T[:3, 3] = pos
    return T


def camera_xyaxes(pos: np.ndarray, look_at: np.ndarray) -> np.ndarray:
    """I due assi (destra, alto) di una camera che da ``pos`` guarda ``look_at``.

    In MuJoCo una camera guarda lungo il proprio **-Z** e ha **+Y** in alto.
    ``xyaxes`` sono i primi due assi espressi nel mondo; il terzo segue dal
    prodotto vettoriale, quindi l'asse ottico non va dichiarato: e'
    **implicato** dalla geometria, e non puo' contraddirla.

    Non Eulero: dipende dall'ordine delle rotazioni, e il primo tentativo in
    D18 guardava il soffitto. Qui si dichiara **dove puntano gli assi**.
    Non vale per una camera che guarda in verticale: "destra" sarebbe indefinita.
    """
    p = np.asarray(pos, dtype=float)
    fwd = np.asarray(look_at, dtype=float) - p
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return np.concatenate([right, up])


def approach_from(T_grasp: Pose, distance: float = 0.10) -> Pose:
    """Pose di avvicinamento: arretrata di ``distance`` lungo l'asse della pinza.

    Non si va mai diretti sull'oggetto: ci si posiziona sopra e si scende. Cosi'
    l'avvicinamento e' prevedibile e non si urta l'oggetto di lato.
    """
    T = T_grasp.copy()
    T[:3, 3] -= T_grasp[:3, 2] * distance
    return T
