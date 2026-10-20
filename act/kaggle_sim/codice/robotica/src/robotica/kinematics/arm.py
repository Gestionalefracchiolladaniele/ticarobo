"""Cinematica del braccio: FK, IK e limiti di giunto.

**FK** (diretta): dati gli angoli dei giunti, dove finisce la pinza? Calcolo
diretto, una sola risposta, sempre esatta.

**IK** (inversa): data una pose voluta per la pinza, quali angoli servono? Puo'
avere infinite soluzioni, una sola, o nessuna (bersaglio fuori portata). Qui si
risolve numericamente con ``mink``: a ogni passo si calcola di quanto muovere
ciascun giunto per ridurre l'errore, rispettando i limiti. Se converge, la
soluzione e' valida; se no, il bersaglio non e' raggiungibile da quella posa.
"""

from __future__ import annotations

from dataclasses import dataclass

import mink
import mujoco
import numpy as np

from robotica import transforms as tf

# Tolleranze di convergenza dell'IK. 1 mm e ~0.6 gradi: sotto la ripetibilita'
# di qualunque braccio reale, quindi non ha senso chiedere di piu'.
POS_TOL = 1e-3
ROT_TOL = 1e-2
MAX_ITERS = 200
DT = 0.02


@dataclass
class IKResult:
    """Esito di una risoluzione IK."""

    success: bool
    qpos: np.ndarray  # configurazione dei giunti trovata
    pos_error: float  # metri
    rot_error: float  # radianti
    iterations: int
    reason: str = ""

    def __bool__(self) -> bool:
        return self.success


class ArmKinematics:
    """FK e IK per un braccio, riferite al site TCP.

    Opera su una copia interna dei dati: risolvere l'IK non muove il robot
    nella scena. Per muoverlo davvero si applica il risultato col controller.
    """

    def __init__(self, model: mujoco.MjModel, tcp_site: str = "tcp") -> None:
        self.model = model
        self.tcp_site = tcp_site
        self._data = mujoco.MjData(model)

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tcp_site)
        if site_id < 0:
            raise KeyError(f"site '{tcp_site}' non trovato nel modello")
        self.tcp_site_id = site_id

        self.arm_joints = self._find_arm_joints()
        self.configuration = mink.Configuration(model)

    def _find_arm_joints(self) -> list[int]:
        """Indici dei giunti del braccio: quelli a cerniera, esclusi dita e oggetti."""
        joints = []
        for i in range(self.model.njnt):
            if self.model.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
            if "finger" in name or "gripper" in name or "jaw" in name.lower():
                continue
            joints.append(i)
        return joints

    def full_qpos(self, q: np.ndarray | None) -> np.ndarray:
        """Porta ``q`` alla lunghezza ``model.nq``.

        Chi chiama passa a volte solo gli angoli del braccio (7 valori), a
        volte il qpos completo che include dita e oggetti liberi (23). Qui si
        accettano entrambi: i valori mancanti vengono dalla keyframe iniziale.
        """
        full = np.zeros(self.model.nq)
        if self.model.nkey > 0:
            full[:] = self.model.key_qpos[0]
        if q is None:
            return full
        q = np.asarray(q, dtype=float).ravel()
        if len(q) >= self.model.nq:
            return q[: self.model.nq].copy()
        full[: len(q)] = q
        return full

    # --- FK ------------------------------------------------------------------

    def forward(self, qpos: np.ndarray) -> np.ndarray:
        """FK: dalla configurazione dei giunti alla pose 4x4 del TCP nel mondo."""
        self._data.qpos[:] = self.full_qpos(qpos)
        mujoco.mj_kinematics(self.model, self._data)
        T = np.eye(4)
        T[:3, :3] = self._data.site_xmat[self.tcp_site_id].reshape(3, 3)
        T[:3, 3] = self._data.site_xpos[self.tcp_site_id]
        return T

    def tcp_pose(self, data: mujoco.MjData) -> np.ndarray:
        """Pose corrente del TCP, letta da dati gia' aggiornati."""
        T = np.eye(4)
        T[:3, :3] = data.site_xmat[self.tcp_site_id].reshape(3, 3)
        T[:3, 3] = data.site_xpos[self.tcp_site_id]
        return T

    # --- IK ------------------------------------------------------------------

    def inverse(
        self,
        T_world_target: np.ndarray,
        q_init: np.ndarray | None = None,
        pos_tol: float = POS_TOL,
        rot_tol: float = ROT_TOL,
        max_iters: int = MAX_ITERS,
    ) -> IKResult:
        """IK: trova la configurazione che porta il TCP sulla pose voluta.

        ``q_init`` e' il punto di partenza. Partire dalla posa corrente del
        robot fa convergere verso una soluzione vicina, cioe' un movimento
        breve invece di un ribaltamento del braccio.
        """
        self.configuration.update(self.full_qpos(q_init))

        pos, quat = tf.to_pos_quat(T_world_target)
        target = mink.SE3.from_rotation_and_translation(
            mink.SO3(np.asarray(quat, dtype=float)), np.asarray(pos, dtype=float)
        )

        task = mink.FrameTask(
            frame_name=self.tcp_site,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        task.set_target(target)

        # Penalizza spostamenti ampi: fra soluzioni equivalenti sceglie la piu'
        # vicina alla posa attuale, cioe' il movimento piu' piccolo.
        posture = mink.PostureTask(self.model, cost=1e-3)
        posture.set_target(self.configuration.q.copy())

        limits = [mink.ConfigurationLimit(self.model)]
        tasks = [task, posture]

        for it in range(max_iters):
            vel = mink.solve_ik(self.configuration, tasks, DT, "quadprog", 1e-3, limits=limits)
            self.configuration.integrate_inplace(vel, DT)

            err = task.compute_error(self.configuration)
            pos_err = float(np.linalg.norm(err[:3]))
            rot_err = float(np.linalg.norm(err[3:]))

            if pos_err < pos_tol and rot_err < rot_tol:
                return IKResult(True, self.configuration.q.copy(), pos_err, rot_err, it + 1)

        return IKResult(
            False, self.configuration.q.copy(), pos_err, rot_err, max_iters,
            reason=f"non converge: errore {pos_err * 1000:.1f} mm, {np.degrees(rot_err):.1f} deg",
        )

    def reachable(self, T_world_target: np.ndarray, q_init: np.ndarray | None = None) -> bool:
        """Il TCP puo' raggiungere questa pose? Scorciatoia sopra ``inverse``."""
        return bool(self.inverse(T_world_target, q_init))

    # --- limiti --------------------------------------------------------------

    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Limiti (min, max) dei giunti del braccio, in radianti."""
        lo = np.array([self.model.jnt_range[j][0] for j in self.arm_joints])
        hi = np.array([self.model.jnt_range[j][1] for j in self.arm_joints])
        return lo, hi

    def violated_limits(self, qpos: np.ndarray) -> list[str]:
        """Nomi dei giunti fuori dai limiti, con margine 1e-6 per il rumore numerico.

        Legge ciascun giunto dal suo indirizzo in qpos, non per posizione: cosi'
        funziona sia con il vettore completo sia con i soli angoli del braccio.
        """
        full = self.full_qpos(qpos)
        bad = []
        for j in self.arm_joints:
            lo, hi = self.model.jnt_range[j]
            if lo >= hi:
                continue
            val = float(full[self.model.jnt_qposadr[j]])
            if not (lo - 1e-6 <= val <= hi + 1e-6):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                bad.append(f"{name}={np.degrees(val):.1f}deg fuori "
                           f"[{np.degrees(lo):.1f}, {np.degrees(hi):.1f}]")
        return bad
