"""Verifica di sicurezza: cosa il robot non deve fare.

Ogni comando passa da qui **prima** di essere eseguito. E' il principio su cui
poggia il progetto: un LLM che genera comandi robot senza verifica non e'
utilizzabile in produzione, perche' un errore rompe cose o ferisce persone.

Le verifiche sono statiche (geometria e limiti, nessuna simulazione) e quindi
istantanee. Trasferiscono al ~95% su hardware reale, perche' le distanze e i
limiti di giunto sono gli stessi del robot vero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import mujoco
import numpy as np

from robotica import transforms as tf
from robotica.kinematics.arm import ArmKinematics

# Quanto lontano dalla propria base arriva il braccio, in metri.
#
# **E' una proprieta' del robot, non della scena**, e tenerla qui e' la
# correzione di un bug che non dava errori. Il workspace si ricavava da mezza
# dimensione del tavolo: giusto finche' il tavolo e' rimasto piccolo, sbagliato
# appena e' stato allungato a 1,9 m per il nastro — ``safety`` approvava
# bersagli a 95 cm dalla base, dove il Panda non arriva.
#
# Il sintomo non parlava di portata: il braccio partiva, si fermava a meta' e
# riportava un errore **sempre uguale**, 184 mm. Un errore costante non e'
# sfortuna, e' una distanza massima.
#
# Il valore e' conservativo: il Panda dichiara 855 mm di raggio, ma la portata
# utile con la pinza rivolta in basso e un margine di sicurezza e' minore.
ARM_REACH = 0.62


class Severity(StrEnum):
    """Gravita' di una violazione."""

    ERROR = "error"  # blocca l'esecuzione
    WARNING = "warning"  # esegue ma segnala


@dataclass
class Violation:
    """Una regola infranta, con abbastanza contesto per capire cosa correggere."""

    rule: str
    severity: Severity
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity.value}] {self.rule}: {self.detail}"


@dataclass
class SafetyReport:
    """Esito della verifica di un comando."""

    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Vero se non ci sono errori bloccanti (i warning non bloccano)."""
        return not any(v.severity is Severity.ERROR for v in self.violations)

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity is Severity.ERROR]

    def add(self, rule: str, detail: str, severity: Severity = Severity.ERROR) -> None:
        self.violations.append(Violation(rule, severity, detail))

    def __bool__(self) -> bool:
        return self.ok

    def explain(self) -> str:
        """Testo leggibile, da restituire all'agent perche' possa correggersi."""
        if not self.violations:
            return "OK: nessuna violazione."
        return "\n".join(str(v) for v in self.violations)


@dataclass
class Workspace:
    """Volume entro cui il TCP puo' muoversi.

    Su hardware reale corrisponde alle zone di sicurezza configurate nel
    controller. Qui e' un parallelepipedo: semplice e sufficiente.
    """

    x: tuple[float, float] = (0.15, 0.80)
    y: tuple[float, float] = (-0.50, 0.50)
    z: tuple[float, float] = (0.36, 1.00)

    def contains(self, p: np.ndarray) -> bool:
        return (
            self.x[0] <= p[0] <= self.x[1]
            and self.y[0] <= p[1] <= self.y[1]
            and self.z[0] <= p[2] <= self.z[1]
        )

    def describe_violation(self, p: np.ndarray) -> str:
        parts = []
        for axis, (lo, hi), val in zip("xyz", (self.x, self.y, self.z), p, strict=True):
            if not lo <= val <= hi:
                parts.append(f"{axis}={val:.3f} fuori [{lo:.2f}, {hi:.2f}]")
        return "; ".join(parts) or "dentro i limiti"


class SafetyChecker:
    """Applica le regole di sicurezza a pose e configurazioni."""

    def __init__(
        self,
        model: mujoco.MjModel,
        kinematics: ArmKinematics | None = None,
        workspace: Workspace | None = None,
        table_z: float = 0.40,
        min_clearance: float = 0.005,
    ) -> None:
        self.model = model
        self.kin = kinematics
        self.workspace = workspace or Workspace()
        self.table_z = table_z
        self.min_clearance = min_clearance
        self._data = mujoco.MjData(model)

    @classmethod
    def for_env(cls, env) -> SafetyChecker:
        """Checker tarato su un ambiente robosuite.

        Le quote vanno lette dall'env, non fissate qui: robosuite mette il
        piano del tavolo a z=0.825, mentre la vecchia scena MjSpec lo aveva a
        0.40. Un workspace con le quote sbagliate rifiuterebbe ogni comando
        valido — o peggio, ne ammetterebbe di pericolosi.

        Il workspace e' costruito attorno al tavolo con un margine, e limitato
        in alto dove il braccio perde portata.
        """
        # In robosuite ``table_offset`` e' gia' la quota della superficie, non
        # il centro del piano: non va sommato mezzo spessore.
        table_z = float(env.table_offset[2])
        cx, cy = float(env.table_offset[0]), float(env.table_offset[1])

        # **Il workspace e' limitato dal braccio, non dal tavolo**, e i due non
        # coincidono. La prima versione prendeva mezza dimensione del tavolo, e
        # ha funzionato finche' il tavolo e' stato piccolo; con il tavolo del
        # nastro (1,9 m in y) ``safety`` ha cominciato ad **approvare bersagli
        # a 95 cm dalla base**, dove il Panda non arriva.
        #
        # Il sintomo non diceva niente di tutto questo: il braccio partiva, si
        # fermava a meta' strada e riportava un errore di posizione **sempre
        # uguale** — 184 mm, misurato due volte. Un errore costante non e'
        # sfortuna: e' una distanza massima.
        #
        # Allungare il tavolo aveva spostato in silenzio un limite che non
        # c'entrava col tavolo. Ora la portata e' dichiarata qui, dove si puo'
        # leggere, e il tavolo limita solo cio' che il tavolo limita davvero.
        half_x = min(env.table_full_size[0] / 2, ARM_REACH)
        half_y = min(env.table_full_size[1] / 2, ARM_REACH)

        workspace = Workspace(
            x=(cx - half_x, cx + half_x),
            y=(cy - half_y, cy + half_y),
            z=(table_z, table_z + 0.45),
        )
        return cls(
            env.sim.model._model,
            kinematics=None,
            workspace=workspace,
            table_z=table_z,
        )

    def verify_position(self, p: np.ndarray, report: SafetyReport | None = None) -> SafetyReport:
        """Verifica una posizione del TCP, senza risolvere l'IK.

        Con OSC la raggiungibilita' non si verifica per via cinematica: il
        controller e' in coppia e non passa da una soluzione IK. Restano le
        verifiche geometriche, che sono anche quelle che trasferiscono meglio
        sul reale (~95%): workspace e collisione col piano di lavoro.
        """
        rep = report or SafetyReport()
        p = np.asarray(p, dtype=float)

        if not self.workspace.contains(p):
            rep.add("workspace", self.workspace.describe_violation(p))

        if p[2] < self.table_z:
            rep.add(
                "table_collision",
                f"TCP a z={p[2]:.3f} sotto il piano del tavolo (z={self.table_z:.3f})",
            )
        elif p[2] < self.table_z + 0.01:
            rep.add(
                "table_clearance",
                f"TCP a soli {(p[2] - self.table_z) * 1000:.0f} mm dal tavolo",
                Severity.WARNING,
            )
        return rep

    # --- regole singole ------------------------------------------------------

    def check_pose(self, T_world_target: np.ndarray, report: SafetyReport | None = None
                   ) -> SafetyReport:
        """La pose bersaglio e' dentro il workspace e sopra il tavolo?"""
        rep = report or SafetyReport()
        p = tf.translation(T_world_target)

        if not self.workspace.contains(p):
            rep.add("workspace", self.workspace.describe_violation(p))

        if p[2] < self.table_z:
            rep.add(
                "table_collision",
                f"TCP a z={p[2]:.3f} sotto il piano del tavolo (z={self.table_z:.3f})",
            )
        elif p[2] < self.table_z + 0.01:
            rep.add(
                "table_clearance",
                f"TCP a soli {(p[2] - self.table_z) * 1000:.0f} mm dal tavolo",
                Severity.WARNING,
            )
        return rep

    def check_reachable(
        self, T_world_target: np.ndarray, q_init: np.ndarray | None = None,
        report: SafetyReport | None = None,
    ) -> SafetyReport:
        """Esiste una configurazione che porta il TCP li'?

        E' la verifica che risparmia piu' tempo: un bersaglio irraggiungibile
        fallisce comunque, tanto vale saperlo prima di muovere.
        """
        rep = report or SafetyReport()
        if self.kin is None:
            raise RuntimeError(
                "check_reachable richiede la cinematica; questo checker e' stato "
                "costruito con for_env, dove si usa verify_position"
            )
        res = self.kin.inverse(T_world_target, q_init=q_init)
        if not res.success:
            rep.add("unreachable", res.reason)
        else:
            bad = self.kin.violated_limits(res.qpos)
            for b in bad:
                rep.add("joint_limits", b)
        return rep

    def check_config(self, qpos: np.ndarray, report: SafetyReport | None = None
                     ) -> SafetyReport:
        """La configurazione rispetta i limiti e non ha collisioni?"""
        rep = report or SafetyReport()

        for b in self.kin.violated_limits(qpos):
            rep.add("joint_limits", b)

        for pair, depth in self.collisions(qpos):
            rep.add("collision", f"{pair[0]} <-> {pair[1]} (penetrazione {depth * 1000:.1f} mm)")
        return rep

    def collisions(self, qpos: np.ndarray) -> list[tuple[tuple[str, str], float]]:
        """Coppie di geom in contatto in questa configurazione.

        Esclude i contatti con gli oggetti afferrabili: toccarli e' lo scopo,
        non un errore. Sono collisioni solo robot-robot e robot-ambiente.
        """
        self._data.qpos[:] = self.kin.full_qpos(qpos)
        mujoco.mj_forward(self.model, self._data)

        found = []
        for i in range(self._data.ncon):
            con = self._data.contact[i]
            if con.dist > -self.min_clearance:
                continue
            n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or "?"
            n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or "?"
            if _is_graspable(n1) or _is_graspable(n2):
                continue
            found.append(((n1, n2), -con.dist))
        return found

    # --- verifica completa ---------------------------------------------------

    def verify_move(
        self, T_world_target: np.ndarray, q_init: np.ndarray | None = None
    ) -> SafetyReport:
        """Verifica completa di un comando di movimento.

        E' il punto di ingresso unico: ogni comando generato dall'agent passa
        di qui. Ferma alla prima categoria che fallisce, perche' verificare la
        raggiungibilita' di una pose gia' fuori workspace non serve.
        """
        rep = SafetyReport()

        self.check_pose(T_world_target, rep)
        if not rep.ok:
            return rep

        if self.kin is None:
            raise RuntimeError(
                "verify_move richiede la cinematica; questo checker e' stato "
                "costruito con for_env, dove si usa verify_position"
            )
        res = self.kin.inverse(T_world_target, q_init=q_init)
        if not res.success:
            rep.add("unreachable", res.reason)
            return rep

        self.check_config(res.qpos, rep)
        return rep


def _is_graspable(geom_name: str) -> bool:
    """Un geom appartiene a un oggetto che il robot puo' toccare di proposito?"""
    return geom_name.startswith("cube_") or geom_name.startswith("obj_")
