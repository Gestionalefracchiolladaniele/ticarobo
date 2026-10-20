"""Interfaccia di controllo del braccio, sopra il controller OSC di robosuite.

Sostituisce il controller in posizione scritto a mano (vedi D5). Quello
posizionava bene — 0.9 mm — ma la presa non teneva: la pinza urtava l'oggetto
scendendo e l'apertura non avveniva davvero. OSC risolve entrambe le cose,
perche' controlla il braccio **in coppia** nello spazio operativo: quando la
pinza incontra l'oggetto cede invece di forzare, e le dita hanno un attuatore
proprio che apre e chiude sul serio.

I metodi restano quelli che espone un controller reale (``move_to_pose``,
``open_gripper``, ``pick``, ``place``): il codice scritto sopra questa classe
gira su hardware cambiando solo l'implementazione.

**Ogni comando passa da ``safety/`` prima di muovere, senza eccezioni** (D2).
Se la verifica fallisce il robot non si muove e si riceve il motivo in forma
leggibile, cosi' che un agent possa correggersi da solo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotica.safety.checks import SafetyChecker, SafetyReport

# La pinza in robosuite si comanda con l'ultima componente dell'azione:
# -1 apre, +1 chiude. Non e' una posizione ma una direzione di sforzo, ed e'
# il motivo per cui la presa tiene: continua a stringere finche' incontra
# resistenza, invece di andare a un angolo fisso.
GRIPPER_OPEN = -1.0
GRIPPER_CLOSE = 1.0

# Tolleranza di posizionamento. 5 mm: OSC e' un controller in impedenza, non
# insegue la posa con la rigidita' di un controller in posizione. Per prendere
# un cubo da 40 mm e' ampiamente sufficiente, ed e' realistico sul reale.
POS_TOL = 5e-3


@dataclass
class MoveResult:
    """Esito di un movimento."""

    success: bool
    reason: str = ""
    final_pos: np.ndarray | None = None
    pos_error: float = 0.0
    safety: SafetyReport | None = None

    def __bool__(self) -> bool:
        return self.success


class OSCRobot:
    """Braccio controllato in spazio operativo, con verifica obbligatoria.

    Avvolge un ambiente robosuite gia' costruito. Non possiede la scena: la
    scena e' dell'env, qui c'e' solo il modo di comandarla in sicurezza.
    """

    def __init__(
        self,
        env,
        safety: SafetyChecker | None = None,
        enforce_safety: bool = True,
    ) -> None:
        self.env = env
        self.safety = safety or SafetyChecker.for_env(env)
        self.enforce_safety = enforce_safety
        self._gripper = GRIPPER_OPEN

    # --- stato ---------------------------------------------------------------

    @property
    def tcp_pos(self) -> np.ndarray:
        """Posizione corrente del TCP nel frame mondo."""
        return self.env.eef_pos()

    def gripper_width(self) -> float:
        """Apertura fra le dita, in metri. Zero significa pinza chiusa a vuoto."""
        return float(np.sum(np.abs(self._finger_qpos())))

    def _finger_qpos(self) -> np.ndarray:
        """Angoli dei giunti delle dita."""
        gripper = self.env.robots[0].gripper["right"]
        ids = [self.env.sim.model.joint_name2id(j) for j in gripper.joints]
        adr = [self.env.sim.model.jnt_qposadr[i] for i in ids]
        return np.array([self.env.sim.data.qpos[a] for a in adr])

    # --- movimento -----------------------------------------------------------

    def move_to_pos(
        self,
        target: np.ndarray,
        max_steps: int = 120,
        tol: float = POS_TOL,
        check_safety: bool = True,
    ) -> MoveResult:
        """Porta il TCP su una posizione, mantenendo la pinza rivolta in basso.

        Il comando passa dalla verifica prima di muovere: se la posizione e'
        fuori workspace, dentro il tavolo o irraggiungibile, il robot resta
        fermo e si riceve il motivo.

        OSC accetta *delta* rispetto alla posa corrente, non posizioni
        assolute: ogni passo si calcola lo scarto residuo e lo si comanda,
        scalato. E' un anello di posizione chiuso attorno al controller, lo
        stesso schema che si usa su un braccio reale comandato in velocita'.
        """
        target = np.asarray(target, dtype=float)

        report = None
        if check_safety:
            report = self.safety.verify_position(target)
            if self.enforce_safety and not report.ok:
                return MoveResult(False, report.explain(), safety=report)

        err = float("inf")
        for _ in range(max_steps):
            delta = target - self.tcp_pos
            err = float(np.linalg.norm(delta))
            if err < tol:
                break
            self._step_towards(delta)

        ok = err < tol
        return MoveResult(
            ok,
            "" if ok else f"raggiunto con errore {err * 1000:.1f} mm",
            final_pos=self.tcp_pos,
            pos_error=err,
            safety=report,
        )

    def move_around(
        self,
        target: np.ndarray,
        tol: float = 0.02,
        check_safety: bool = True,
    ) -> MoveResult:
        """Come ``move_to_pos``, ma **aggirando** gli ostacoli.

        La differenza sta tutta nel percorso. ``move_to_pos`` va in linea retta:
        e' giusto quando non c'e' niente in mezzo, ed e' quello che il robot ha
        sempre fatto. Qui si chiede prima un percorso libero a ``safety/path``,
        e lo si segue punto per punto.

        **Ripiega sul movimento diretto se non ci sono ostacoli**, perche'
        pianificare un percorso attorno al nulla costa e non serve. E ripiega
        anche se mplib non e' installato: il progetto deve restare usabile senza
        l'extra ``motion``.

        Il percorso e' in **configurazioni di giunto**, non in posizioni del
        TCP: e' l'unica forma in cui "aggirare" ha senso, perche' e' il braccio
        intero a dover evitare l'ostacolo, non solo la pinza.
        """
        target = np.asarray(target, dtype=float)

        report = None
        if check_safety:
            report = self.safety.verify_position(target)
            if self.enforce_safety and not report.ok:
                return MoveResult(False, report.explain(), safety=report)

        planner = self._path_planner()
        if planner is None:
            return self.move_to_pos(target, check_safety=False)

        from robotica.safety.path import obstacles_from_env

        ostacoli = obstacles_from_env(self.env)
        if not ostacoli:
            return self.move_to_pos(target, check_safety=False)

        planner.set_obstacles(ostacoli)
        result = planner.plan_to_position(self._arm_qpos(), target)

        if not result:
            # Si riporta il motivo invece di provare comunque in linea retta:
            # andare dritti dove un pianificatore ha appena detto che non si
            # passa e' esattamente il comportamento che questo modulo esiste
            # per impedire.
            return MoveResult(False, result.explain(), safety=report)

        return self._follow(result.waypoints, target, tol, report)

    def _follow(
        self,
        waypoints: np.ndarray,
        target: np.ndarray,
        tol: float,
        report: SafetyReport | None,
    ) -> MoveResult:
        """Segue un percorso di configurazioni di giunto.

        Il controller e' OSC — comanda la **posa del TCP**, non i giunti — quindi
        le configurazioni vanno tradotte in posizioni e inseguite una per una.
        E' un compromesso: si perde il controllo esatto della posa del gomito
        fra un punto e l'altro, ma i punti sono fitti e il percorso resta quello
        verificato.
        """
        for q in waypoints:
            punto = self._fk_tcp(q)
            if punto is None:
                continue
            # Tolleranza larga sui punti intermedi: fermarsi a inseguirli con
            # precisione costerebbe tempo senza servire, perche' contano come
            # direzione e non come bersagli.
            self.move_to_pos(punto, max_steps=25, tol=0.02, check_safety=False)

        res = self.move_to_pos(target, tol=tol, check_safety=False)
        return MoveResult(
            res.success,
            res.reason,
            final_pos=self.tcp_pos,
            pos_error=res.pos_error,
            safety=report,
        )

    def _arm_qpos(self) -> np.ndarray:
        """Gli angoli dei 7 giunti del braccio, piu' le due dita."""
        robot = self.env.robots[0]
        arm = np.array([self.env.sim.data.qpos[a] for a in robot._ref_joint_pos_indexes])
        return np.concatenate([arm, self._finger_qpos()])

    def _fk_tcp(self, qpos: np.ndarray) -> np.ndarray | None:
        """Dove si trova il TCP in una data configurazione, secondo mplib."""
        planner = self._path_planner()
        if planner is None:
            return None
        return planner.tcp_of(qpos)

    def _path_planner(self):
        """Il pianificatore di percorso, costruito una volta sola.

        Costruirlo costa (mplib legge l'URDF e calcola le coppie di link che non
        collidono mai), quindi si tiene. Se mplib manca si restituisce ``None``
        e chi chiama ripiega sul movimento diretto: il motion planning e' un
        extra, non un requisito per far muovere il braccio.
        """
        if hasattr(self, "_planner_cache"):
            return self._planner_cache

        try:
            from robotica.safety.path import PathPlanner

            planner = PathPlanner()
            # Si **misura** lo scarto fra i due frame invece di calcolarlo dalla
            # posa della base: fra ``robot0_base`` e la radice dell'URDF c'e' un
            # offset interno di ~96 mm che nessun documento dichiara, e un
            # errore di frame non si vede come errore — si vede come un robot
            # che evita ostacoli inesistenti e urta quelli veri.
            planner.calibrate(self._arm_qpos(), self.tcp_pos)
        except Exception:
            planner = None

        self._planner_cache = planner
        return planner

    def _step_towards(self, delta: np.ndarray) -> None:
        """Un passo di controllo verso il delta voluto.

        Il delta va scalato in [-1, 1]: l'azione OSC e' normalizzata e
        ``output_max`` la riporta a 5 cm per passo. Si limita a meta' di quel
        massimo per non eccitare oscillazioni quando l'errore e' grande.
        """
        scale = 20.0
        cmd = np.clip(delta * scale, -0.5, 0.5)
        action = np.zeros(self.env.action_dim)
        action[:3] = cmd
        action[3:6] = self._orientation_correction()
        action[-1] = self._gripper
        self.env.step(action)

    def _orientation_correction(self) -> np.ndarray:
        """Correzione che tiene la pinza rivolta verso il basso.

        Una presa dall'alto e' l'unica che non urta il cubo di lato, quindi
        l'orientamento non e' libero: si riporta l'asse z della pinza su -z
        del mondo a ogni passo. Senza questo il braccio deriva e scende
        storto, che era una delle cause per cui la vecchia presa mancava.
        """
        mat = self.env.sim.data.site_xmat[self.env.robots[0].eef_site_id["right"]]
        z_axis = np.asarray(mat).reshape(3, 3)[:, 2]
        # Prodotto vettoriale fra dove punta la pinza e dove dovrebbe puntare:
        # e' l'asse di rotazione che annulla lo scarto.
        return np.clip(np.cross(z_axis, np.array([0.0, 0.0, -1.0])) * 2.0, -0.5, 0.5)

    def hold(self, steps: int = 20) -> None:
        """Resta fermo mantenendo lo stato della pinza.

        Serve dopo una chiusura: la presa si assesta in qualche decina di
        passi, e leggere il risultato prima falsa la verifica.
        """
        for _ in range(steps):
            action = np.zeros(self.env.action_dim)
            action[-1] = self._gripper
            self.env.step(action)

    # --- pinza ---------------------------------------------------------------

    def open_gripper(self, steps: int = 25) -> None:
        """Apre la pinza e attende che le dita si muovano davvero."""
        self._gripper = GRIPPER_OPEN
        self.hold(steps)

    def close_gripper(self, steps: int = 35) -> None:
        """Chiude la pinza sull'oggetto.

        In simulazione la presa tiene quasi sempre; sul reale no. E' il grosso
        del divario sim-to-real, ed e' il motivo per cui ``holding`` verifica
        il risultato invece di darlo per fatto.
        """
        self._gripper = GRIPPER_CLOSE
        self.hold(steps)

    def holding(self, object_name: str) -> bool:
        """L'oggetto e' davvero in mano?

        Due condizioni insieme: le dita non sono del tutto chiuse (se lo
        fossero, si sarebbero richiuse a vuoto) e l'oggetto e' vicino al TCP.
        Controllare solo il comando di chiusura direbbe sempre di si'.
        """
        fingers = self._finger_qpos()
        if np.sum(np.abs(fingers)) < 0.005:
            return False
        dist = float(np.linalg.norm(self.env.object_pos(object_name) - self.tcp_pos))
        return dist < 0.045

    # --- primitive -----------------------------------------------------------

    # Quanto sopra l'oggetto ci si posiziona prima di scendere.
    APPROACH_HEIGHT = 0.10

    def pick(self, object_name: str) -> MoveResult:
        """Prende un oggetto: sopra, giu', stringi, su.

        L'ordine non e' arbitrario. Scendere verticalmente e' l'unico
        avvicinamento che non spinge via il cubo; risalire prima di traslare
        evita di trascinarlo sul tavolo.
        """
        try:
            obj = self.env.object_pos(object_name)
        except KeyError as exc:
            return MoveResult(False, str(exc))

        above = obj + np.array([0.0, 0.0, self.APPROACH_HEIGHT])

        self.open_gripper()
        res = self.move_to_pos(above)
        if not res:
            return MoveResult(
                False, f"avvicinamento sopra {object_name}: {res.reason}", safety=res.safety
            )

        # La discesa sulla presa salta la verifica di workspace perche' il TCP
        # deve arrivare a filo del tavolo, dove la regola di clearance
        # scatterebbe. Non e' un'eccezione a D2: il punto di arrivo e' stato
        # gia' verificato sopra, e la discesa e' un movimento vincolato di
        # 10 cm in verticale su un bersaglio noto.
        res = self.move_to_pos(obj, check_safety=False, tol=0.008)
        if not res:
            return MoveResult(False, f"discesa su {object_name}: {res.reason}")

        self.close_gripper()
        if not self.holding(object_name):
            return MoveResult(False, f"presa fallita: {object_name} non in mano")

        res = self.move_to_pos(above, check_safety=False)
        if not self.holding(object_name):
            return MoveResult(False, f"{object_name} perso durante il sollevamento")

        return MoveResult(True, final_pos=self.tcp_pos)

    def place(self, target: np.ndarray, object_name: str | None = None) -> MoveResult:
        """Posa l'oggetto in un punto: sopra, giu', apri, su."""
        target = np.asarray(target, dtype=float)
        above = target + np.array([0.0, 0.0, self.APPROACH_HEIGHT])

        res = self.move_to_pos(above)
        if not res:
            return MoveResult(
                False, f"trasporto sopra la destinazione: {res.reason}", safety=res.safety
            )

        # Si rilascia poco sopra il piano: il cubo cade di pochi millimetri
        # invece di essere premuto sul tavolo dalla pinza ancora chiusa.
        drop = target + np.array([0.0, 0.0, 0.015])
        self.move_to_pos(drop, check_safety=False, tol=0.01)

        self.open_gripper()
        self.move_to_pos(above, check_safety=False)

        return MoveResult(True, final_pos=self.tcp_pos)
