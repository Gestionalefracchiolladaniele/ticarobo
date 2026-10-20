"""L'orchestratore: mette insieme mappa, planner, controller, batteria.

E' cio' che in Nav2 e' il ``bt_navigator``, il behavior tree che decide
*quando* pianificare, *quando* ripianificare, e cosa fare quando qualcosa va
storto. Qui e' una macchina a stati esplicita invece di un albero: con cinque
stati l'albero sarebbe cerimoniale, e ``py_trees`` -- che il progetto ha
installato e non ancora usato -- merita di entrare su un problema che lo
giustifichi, non su questo.

**Il ciclo, e perche' e' fatto cosi':**

1. il dispatcher sceglie un ordine (o rinuncia)
2. il planner globale traccia il percorso sulla costmap statica
3. il controller lo insegue a 20 Hz, guardando il LiDAR
4. se il controller dichiara ``BLOCKED``, si **ripianifica** su una costmap
   che include cio' che il LiDAR vede adesso
5. la batteria cala; quando serve, si va a caricare

Il punto 4 e' il cuore: la costmap globale non sa dei cassoni, perche' sono
comparsi dopo. Il LiDAR li vede, la costmap locale li incorpora, e il planner
trova un'altra strada. E' D16 per l'AMR -- con la differenza che qui
l'ostacolo **non c'era** quando il piano e' stato fatto.

**La posa viene letta dalla scena, ed e' esatta.** Sul robot vero non
esisterebbe: sarebbe una stima che deriva, ricorretta da AMCL confrontando il
LiDAR con la mappa. E' il limite dichiarato di questo layer (40-60% di
transfer), e sta scritto qui perche' e' qui che si assume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..sim.amr import warehouse
from ..sim.amr.base import HolonomicBase
from ..sim.amr.collision import CollisionMonitor
from ..sim.amr.lidar import Lidar2D
from .controller import Outcome, PurePursuit
from .costmap import DEFAULT_INFLATION, Costmap, Footprint, build_static, footprint_from_model
from .mission import Battery, Dispatcher, MissionStats, Order, OrderState
from .planner import NoPathFound, astar, simplify


class Phase(Enum):
    """Cosa sta facendo il robot, adesso."""

    FERMO = "fermo"
    AL_PRELIEVO = "va a prelevare"
    ALLA_CONSEGNA = "va a consegnare"
    ALLA_RICARICA = "va a caricarsi"
    IN_CARICA = "in carica"


@dataclass
class Event:
    """Qualcosa e' successo. Serve a raccontarlo mentre accade."""

    t: float
    kind: str
    text: str

    def __str__(self) -> str:
        return f"[{self.t:6.1f}s] {self.text}"


@dataclass
class Runner:
    """Fa girare il robot sugli ordini, e tiene i conti.

    Non apre finestre e non stampa: **emette eventi**. Cosi' lo stesso motore
    serve la modalita' automatica, quella interattiva e i test, che e' la
    scelta di ``nastro.ps1`` con ``-Misura`` e ``-Guarda``.
    """

    model: object
    data: object
    base: HolonomicBase
    lidar: Lidar2D
    monitor: CollisionMonitor
    costmap: Costmap
    footprint: Footprint
    dispatcher: Dispatcher
    battery: Battery = field(default_factory=Battery)
    stats: MissionStats = field(default_factory=MissionStats)

    phase: Phase = Phase.FERMO
    current: Order | None = None
    events: list[Event] = field(default_factory=list)
    t: float = 0.0

    _ctrl: PurePursuit | None = field(default=None, init=False)
    _target_name: str = field(default="", init=False)
    _last_xy: np.ndarray | None = field(default=None, init=False)
    _replans_here: int = field(default=0, init=False)
    _path_shown: list = field(default_factory=list, init=False)

    # Quante volte ripianificare sullo stesso tratto prima di dichiarare
    # fallito l'ordine. Non infinito: un robot che ripianifica all'infinito
    # e' un sistema che si e' arreso senza dirlo (D14).
    max_replans: int = 4

    # ------------------------------------------------------------------
    # costruzione
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        policy: str = "prudente",
        with_obstacles: bool = True,
        n_rays: int = 360,
        resolution: float = 0.05,
        start: str = "ricarica",
    ) -> Runner:
        """La scena, il robot e la mappa, pronti a girare."""
        model, data = warehouse.build(with_obstacles=with_obstacles)
        base = HolonomicBase(model, data)
        xy = warehouse.station_xy(start)
        base.teleport(float(xy[0]), float(xy[1]), 0.0)

        fp = footprint_from_model(model, data)
        hx, hy = warehouse.HALL_SIZE[0] / 2 + 0.3, warehouse.HALL_SIZE[1] / 2 + 0.3
        cm = build_static(
            (-hx, -hy, hx, hy),
            warehouse.static_obstacles(),
            fp,
            resolution=resolution,
        )
        return cls(
            model=model,
            data=data,
            base=base,
            lidar=Lidar2D(model, data, n_rays=n_rays),
            monitor=CollisionMonitor(model, data),
            costmap=cm,
            footprint=fp,
            dispatcher=Dispatcher(policy),
        )

    # ------------------------------------------------------------------
    # utilita'
    # ------------------------------------------------------------------

    def _emit(self, kind: str, text: str) -> None:
        self.events.append(Event(self.t, kind, text))

    def _xy(self, where) -> np.ndarray:
        """Accetta un nome di stazione o coordinate. Comodita' voluta."""
        if isinstance(where, str):
            return warehouse.station_xy(where)
        return np.asarray(where, dtype=float)

    def path_length(self, a, b) -> float:
        """Lunghezza del percorso fra due punti. Solleva se non esiste.

        E' la funzione che il dispatcher usa per decidere: **la lunghezza
        vera**, non la distanza in linea d'aria. In un magazzino con quattro
        scaffali la differenza e' grande, e usare la linea d'aria farebbe
        accettare ordini che non si possono concludere.
        """
        _, info = astar(self.costmap, tuple(self._xy(a)), tuple(self._xy(b)))
        return float(info["length_m"])

    def spawn_obstacle(self, name: str, x: float, y: float) -> None:
        """Porta un cassone dentro il magazzino. L'ostacolo che compare dopo."""
        import mujoco

        bid = self.model.body(name).id
        adr = self.model.jnt_qposadr[self.model.body_jntadr[bid]]
        self.data.qpos[adr : adr + 3] = [x, y, 0.3]
        self.data.qpos[adr + 3 : adr + 7] = [1, 0, 0, 0]
        dof = self.model.jnt_dofadr[self.model.body_jntadr[bid]]
        self.data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._emit("ostacolo", f"un cassone compare in ({x:.1f}, {y:.1f})")

    # ------------------------------------------------------------------
    # pianificazione
    # ------------------------------------------------------------------

    def _plan_to(self, target, *, use_lidar: bool = False) -> bool:
        """Traccia il percorso verso ``target``. Ritorna False se non esiste.

        Con ``use_lidar`` la costmap include cio' che il sensore vede adesso:
        e' la ripianificazione, e serve solo quando qualcosa si e' messo di
        mezzo. Farlo sempre costerebbe (la costmap va ricostruita) e
        renderebbe il percorso dipendente da dove il robot guardava -- che e'
        un modo elegante per rendere un esperimento irripetibile.
        """
        st = self.base.state()
        cm = self.costmap
        if use_lidar:
            cm = self.costmap.copy()
            cm.add_points(self.lidar.scan().points_world)
            cm.inflate(self.footprint, DEFAULT_INFLATION)

        try:
            path, info = astar(cm, tuple(st.xy_world), tuple(self._xy(target)))
        except NoPathFound as e:
            self._emit("errore", f"nessun percorso: {e.args[0].split(':')[0]}")
            return False

        # **Il percorso denso, non quello semplificato.** Pure pursuit cerca
        # il punto del percorso a distanza ``lookahead``: su un percorso
        # ridotto a 3 waypoint distanti metri, quel punto non esiste e il
        # controller sbanda avanti e indietro fra due vertici -- misurato,
        # oscillava all'infinito con l'indice fermo.
        #
        # ``simplify`` resta utile per **mostrare** un percorso (3 numeri si
        # leggono, 200 no), non per inseguirlo. Sono due usi diversi dello
        # stesso dato, e confonderli e' costato un pomeriggio.
        self._ctrl = PurePursuit(path)
        self._path_shown = simplify(path, cm)
        if info.get("goal_moved_m", 0) > 0.01:
            self._emit(
                "nota",
                f"bersaglio spostato di {info['goal_moved_m'] * 100:.0f} cm "
                "(cadeva in una zona vietata)",
            )
        return True

    # ------------------------------------------------------------------
    # il ciclo
    # ------------------------------------------------------------------

    def _begin(self, phase: Phase, target, label: str) -> bool:
        if not self._plan_to(target, use_lidar=self._replans_here > 0):
            return False
        self.phase = phase
        self._target_name = label
        return True

    def _next_job(self) -> None:
        """Sceglie cosa fare: un ordine, la ricarica, o niente."""
        st = self.base.state()

        if self.battery.low and self.phase is not Phase.IN_CARICA:
            n = self.dispatcher.decline_all(f"carica {self.battery} sotto la riserva")
            if n:
                self.stats.declined += n
                self._emit("rinuncia", f"rinuncia a {n} ordini: carica {self.battery}")
            if self._begin(Phase.ALLA_RICARICA, warehouse.CHARGER, "ricarica"):
                self._emit("ricarica", f"torna a caricarsi (carica {self.battery})")
            return

        order, why = self.dispatcher.choose(
            st.xy_world, self.battery, self.path_length, self._xy(warehouse.CHARGER)
        )

        if order is None:
            if "insufficiente" in why:
                n = self.dispatcher.decline_all(why)
                self.stats.declined += n
                self._emit("rinuncia", f"rinuncia a {n} ordini -- {why}")
                if self._begin(Phase.ALLA_RICARICA, warehouse.CHARGER, "ricarica"):
                    self._emit("ricarica", "va a caricarsi")
            else:
                self.phase = Phase.FERMO
            return

        order.state = OrderState.IN_CORSO
        self.current = order
        self._replans_here = 0
        if self._begin(Phase.AL_PRELIEVO, order.pickup, order.pickup):
            self._emit("ordine", f"accetta {order} -- {why}")
        else:
            order.state = OrderState.FALLITO
            order.reason = "nessun percorso al prelievo"
            self.stats.failed += 1
            self.current = None

    def _arrived(self) -> None:
        """Il controller dice ARRIVED: cosa succede dopo dipende dalla fase."""
        if self.phase is Phase.AL_PRELIEVO and self.current:
            warehouse.set_cargo(self.model, self.data, True)
            self._emit("prelievo", f"preleva a {self.current.pickup} -- ora e' carico")
            self._replans_here = 0
            if not self._begin(Phase.ALLA_CONSEGNA, self.current.dropoff, self.current.dropoff):
                self.current.state = OrderState.FALLITO
                self.current.reason = "nessun percorso alla consegna"
                self.stats.failed += 1
                self.current = None
                self.phase = Phase.FERMO

        elif self.phase is Phase.ALLA_CONSEGNA and self.current:
            warehouse.set_cargo(self.model, self.data, False)
            self.current.state = OrderState.SERVITO
            self.stats.served += 1
            self._emit(
                "consegna",
                f"consegna {self.current} -- servito ({self.stats.served}), "
                f"carica {self.battery}",
            )
            self.current = None
            self.phase = Phase.FERMO

        elif self.phase is Phase.ALLA_RICARICA:
            self.phase = Phase.IN_CARICA
            self._emit("carica", f"in stazione, carica {self.battery}")

        else:
            self.phase = Phase.FERMO

    def _blocked(self, res) -> None:
        """Il controller si e' fermato. Ripianificare o arrendersi?"""
        self._replans_here += 1
        self.stats.replans += 1
        if self._replans_here > self.max_replans:
            self._emit(
                "errore",
                f"rinuncia dopo {self.max_replans} ripianificazioni: {res.reason}",
            )
            if self.current:
                self.current.state = OrderState.FALLITO
                self.current.reason = res.reason
                self.stats.failed += 1
                self.current = None
            self.phase = Phase.FERMO
            return

        self._emit(
            "ostacolo",
            f"bloccato ({res.reason}); ripianifica col LiDAR "
            f"[{self._replans_here}/{self.max_replans}]",
        )
        self.stats.obstacles_avoided += 1
        if not self._plan_to(self._target_name, use_lidar=True):
            if self.current:
                self.current.state = OrderState.FALLITO
                self.current.reason = "nessuna via alternativa"
                self.stats.failed += 1
                self.current = None
            self.phase = Phase.FERMO

    def tick(self) -> None:
        """Un passo del mondo: 50 ms. E' l'unica funzione che fa avanzare il tempo."""
        dt = self.base.control_dt
        self.t += dt

        if self.phase is Phase.FERMO:
            self.battery.consume_idle(dt)
            self._next_job()
            return

        if self.phase is Phase.IN_CARICA:
            self.battery.charge(dt)
            self.base.stop()
            self.base.step()
            if self.battery.level >= 0.99:
                self.stats.recharges += 1
                self._emit("carica", "carica completa")
                self.phase = Phase.FERMO
            return

        # --- in marcia -----------------------------------------------------
        assert self._ctrl is not None
        st = self.base.state()
        scan = self.lidar.scan()
        res = self._ctrl.step(st.pose_world, scan)

        if res.outcome is Outcome.ARRIVED:
            self.base.stop()
            self.base.step()
            self._arrived()
            return

        if res.outcome in (Outcome.BLOCKED, Outcome.NO_PROGRESS):
            self.base.stop()
            self.base.step()
            self._blocked(res)
            return

        self.base.command_velocity(*res.velocity)
        self.base.step()

        # --- conti ----------------------------------------------------------
        now = self.base.state().xy_world
        if self._last_xy is not None:
            step = float(np.linalg.norm(now - self._last_xy))
            self.stats.distance_m += step
            self.battery.consume_distance(step)
        self._last_xy = now.copy()

        before = self.monitor.report.contacts
        self.monitor.check()
        if self.monitor.report.contacts > before:
            self.stats.collisions += 1

        # --- scarico lontano dalla base: il fallimento peggiore -------------
        if self.battery.level <= 0.0 and not self.stats.stranded:
            self.stats.stranded = True
            d = float(np.linalg.norm(now - self._xy(warehouse.CHARGER)))
            self._emit(
                "guasto",
                f"BATTERIA A ZERO a {d:.1f} m dalla stazione di ricarica. "
                "Il robot e' fermo dove si trova.",
            )
            if self.current:
                self.current.state = OrderState.FALLITO
                self.current.reason = "batteria esaurita"
                self.stats.failed += 1
                self.current = None
            self.phase = Phase.FERMO

    def run(self, max_seconds: float = 300.0) -> MissionStats:
        """Gira finche' gli ordini sono chiusi o scade il tempo."""
        while self.t < max_seconds:
            self.tick()
            if self.stats.stranded:
                break
            if self.phase is Phase.FERMO and not self.dispatcher.pending:
                break
        return self.stats
