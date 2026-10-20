"""Verificare il **percorso**, non solo il punto di arrivo.

``checks.py`` risponde a *"questo punto e' sicuro?"*. E' la domanda giusta per
un bersaglio e quella sbagliata per un movimento: con un ostacolo in mezzo il
punto di arrivo resta perfettamente legittimo, il robot ci va in linea retta e
ci sbatte contro. La verifica approva, e il braccio spinge.

Misurato sulla scena ``ObstacleEnv``: ``safety`` **approva** la destinazione, il
``pick`` riesce, il ``place`` si arena con 135 mm di errore e il cubo resta
appoggiato al muro. Nessun controllo sul punto finale poteva accorgersene.

## Perche' non si scrive a mano

Il problema sembra geometria elementare e non lo e', per due motivi:

- **Il braccio non e' un punto.** La pinza puo' passare sopra l'ostacolo mentre
  il *gomito* ci sbatte. Va verificato tutto il corpo, in ogni istante del
  movimento.
- **Lo spazio in cui cercare non e' quello che si vede.** Il Panda si muove in
  7 dimensioni, una per giunto. Un percorso "che gira intorno" e' una curva in
  uno spazio a 7 dimensioni, che non si puo' disegnare ne' percorrere a
  intuito.

Da qui D5: si usa ``mplib`` (MIT), che si appoggia a OMPL per gli algoritmi di
campionamento e a hpp-fcl per le collisioni. Vedi D16.

## Cosa fa questo modulo, e cosa no

Traduce fra il nostro mondo e mplib: costruisce il modello del braccio, gli
descrive gli ostacoli, e restituisce un percorso come **sequenza di
configurazioni di giunto**. Non muove nulla: eseguire e' compito di ``sim/``.

La separazione e' voluta. Un percorso calcolato e' un dato che si puo'
ispezionare, contare, verificare passo per passo — esattamente come un piano
simbolico (D13). Se il calcolo e l'esecuzione fossero la stessa funzione, non ci
sarebbe niente da guardare prima che il braccio si muova.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robotica.paths import CACHE

# Quanto si allargano gli ostacoli rispetto alla loro dimensione vera.
#
# Non e' pignoleria: il modello di collisione del braccio e' approssimato, il
# controller ha un errore di posizione di qualche millimetro, e un percorso che
# sfiora l'ostacolo di 1 mm nel modello lo tocca nella realta'. Si preferisce
# un percorso leggermente piu' largo a uno che passa per un pelo.
OBSTACLE_MARGIN = 0.015

# Quanti tentativi di campionamento prima di dichiarare che non c'e' strada.
#
# La ricerca e' randomizzata: un fallimento singolo non significa impossibile.
# Misurato sulla scena col muro, un tentativo riesce circa 8 volte su 10, e una
# parte dei percorsi trovati viene poi scartata dalla verifica perche' urta.
# Sei tentativi rendono il fallimento raro, e costano poco: ogni tentativo e'
# circa 0,2 secondi, contro i ~10 secondi del movimento che ne segue.
PLANNING_ATTEMPTS = 6


@dataclass
class PathResult:
    """Un percorso trovato, o il motivo per cui non c'e'.

    Si valuta come vero/falso come tutti gli esiti del progetto: ``safety``,
    ``Plan``, ``Outcome``. La convenzione e' la stessa ovunque — un risultato
    che si interroga, col motivo dentro.
    """

    waypoints: np.ndarray | None = None
    reason: str = ""
    attempts: int = 0
    rejected: int = 0
    """Quanti percorsi sono stati **scartati perche' urtavano**, pur essendo
    stati dichiarati riusciti dal pianificatore. Non e' un dettaglio da
    nascondere: e' la misura di quanto la verifica stia lavorando."""

    def __bool__(self) -> bool:
        return self.reason == "" and self.waypoints is not None

    def __len__(self) -> int:
        return 0 if self.waypoints is None else len(self.waypoints)

    def explain(self) -> str:
        if not self:
            return f"NESSUN PERCORSO: {self.reason}"
        coda = f", {self.rejected} scartati perche' urtavano" if self.rejected else ""
        return f"Percorso in {len(self)} configurazioni di giunto{coda}."


@dataclass
class Obstacle:
    """Un ostacolo come scatola: centro, mezze dimensioni, nome.

    Si descrivono come scatole e non come mesh per scelta: il modello grossolano
    e' piu' veloce da controllare e piu' **conservativo** — una scatola che
    contiene l'oggetto vero non lascia passare percorsi che l'oggetto vero
    bloccherebbe. Sbagliare per eccesso di prudenza, su un ostacolo, e' il verso
    giusto in cui sbagliare.
    """

    name: str
    center: np.ndarray
    half_sizes: np.ndarray

    def padded(self, margin: float = OBSTACLE_MARGIN) -> np.ndarray:
        """Le mezze dimensioni allargate del margine di sicurezza."""
        return np.asarray(self.half_sizes, dtype=float) + margin


class PathPlanner:
    """Calcola percorsi senza collisioni per il braccio.

    Costruirlo costa: mplib deve leggere l'URDF e calcolare quali coppie di link
    non collidono mai. Il risultato viene messo in cache su disco, quindi il
    prezzo si paga una volta e non a ogni avvio.
    """

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        move_group: str = "panda_hand",
        base_pose: np.ndarray | None = None,
    ) -> None:
        try:
            import mplib
        except ImportError as exc:  # pragma: no cover - dipende dall'ambiente
            raise ImportError(
                "mplib non installato: il motion planning con ostacoli non e' "
                "disponibile. Installalo con `uv sync --extra motion`."
            ) from exc

        self._mplib = mplib
        urdf = Path(urdf_path) if urdf_path else prepare_panda_urdf()

        self.planner = mplib.Planner(urdf=str(urdf), move_group=move_group)
        self.obstacles: list[Obstacle] = []
        self._registrati: list[str] = []

        # Traslazione dal frame di mplib al frame mondo. Si misura con
        # ``calibrate``: dedurla dalla posa della base sbaglia di ~96 mm, per
        # via di un offset interno fra ``robot0_base`` e la radice dell'URDF.
        self._offset = np.zeros(3)

        if base_pose is not None:
            self.set_base(base_pose)

    # --- la scena -------------------------------------------------------------

    def set_base(self, position: np.ndarray, quat_wxyz: np.ndarray | None = None) -> None:
        """Dove sta la base del robot nel frame mondo.

        Conta: mplib ragiona nel frame della base, il resto del progetto nel
        frame mondo. Sbagliare questo significa pianificare percorsi corretti
        attorno a ostacoli che stanno da un'altra parte — e il piano
        sembrerebbe perfetto.
        """
        q = np.array([1.0, 0.0, 0.0, 0.0]) if quat_wxyz is None else np.asarray(quat_wxyz)
        self.planner.set_base_pose(self._mplib.Pose(np.asarray(position, dtype=float), q))

    def set_obstacles(self, obstacles: list[Obstacle]) -> None:
        """Sostituisce gli ostacoli noti.

        Si sostituiscono invece di aggiungerli perche' vale la regola di sempre:
        lo stato si rilegge, non si accumula. Un ostacolo rimosso dalla scena e
        rimasto nel pianificatore fa rifiutare percorsi che sono liberi, ed e'
        un errore che si manifesta come "il robot non trova mai strada".
        """
        mondo = self.planner.planning_world
        fcl = self._mplib.collision_detection.fcl

        # Via i precedenti, per nome. Togliere prima di aggiungere e' cio' che
        # rende questa funzione una *sostituzione* e non un accumulo.
        for vecchio in self._registrati:
            mondo.remove_object(vecchio)
        self._registrati = []

        self.obstacles = list(obstacles)

        for ob in obstacles:
            # fcl vuole i lati interi, noi ragioniamo in mezze dimensioni come
            # MuJoCo: il fattore 2 sta qui e da nessun'altra parte.
            geom = fcl.Box(ob.padded() * 2.0)
            # Gli ostacoli arrivano nel frame mondo (li legge MuJoCo) e mplib
            # lavora nel suo: senza questa sottrazione il pianificatore
            # eviterebbe ostacoli fantasma e passerebbe dentro quelli veri.
            pose = self._mplib.Pose(
                np.asarray(ob.center, dtype=float) - self._offset,
                np.array([1.0, 0.0, 0.0, 0.0]),
            )
            mondo.add_object(ob.name, fcl.CollisionObject(geom, pose))
            self._registrati.append(ob.name)

    # --- la domanda ----------------------------------------------------------

    def plan_to_position(
        self,
        qpos: np.ndarray,
        target_xyz: np.ndarray,
        quat_wxyz: np.ndarray | None = None,
        attempts: int = PLANNING_ATTEMPTS,
    ) -> PathResult:
        """Un percorso dalla configurazione attuale a una posizione del TCP.

        ``qpos`` e' la configurazione di partenza (7 giunti piu' le dita).
        L'orientamento predefinito e' la pinza rivolta in basso, che e' l'unica
        presa che non urta il cubo di lato.

        Si ritenta piu' volte perche' la ricerca e' **randomizzata**: campiona
        configurazioni a caso e le collega. Un fallimento singolo non dimostra
        che non ci sia strada, e distinguere "non trovato" da "non esiste"
        conta — sono due messaggi diversi per chi legge.
        """
        # Pinza in basso: l'asse z della flangia verso -z del mondo.
        q = np.array([0.0, 1.0, 0.0, 0.0]) if quat_wxyz is None else np.asarray(quat_wxyz)
        # Anche il bersaglio arriva nel frame mondo: stessa conversione degli
        # ostacoli, nella stessa direzione.
        pose = self._mplib.Pose(np.asarray(target_xyz, dtype=float) - self._offset, q)

        ultimo = ""
        scartati = 0

        for tentativo in range(1, attempts + 1):
            try:
                res = self.planner.plan_pose(pose, np.asarray(qpos, dtype=float))
            except Exception as exc:  # mplib solleva su input incoerenti
                return PathResult(reason=f"errore del pianificatore: {exc}", attempts=tentativo)

            if res.get("status") != "Success":
                ultimo = str(res.get("status", "sconosciuto"))
                continue

            percorso = np.asarray(res["position"])

            # **"Success" non basta.** Misurato: 2 percorsi su 10 dichiarati
            # riusciti contengono configurazioni in collisione. La causa e' la
            # fase di parametrizzazione temporale, che interpola il percorso
            # grezzo e nel levigarlo puo' spostare punti dentro l'ostacolo.
            #
            # E' D2 applicato alla libreria: non ci si fida di cio' che un
            # componente dichiara, si verifica. Consegnare un percorso che urta
            # sarebbe peggio che non trovarne uno, perche' il braccio partirebbe.
            urti = self._collisions_along(percorso)
            if urti:
                scartati += 1
                ultimo = f"percorso restituito con {urti} configurazioni in collisione"
                continue

            return PathResult(waypoints=percorso, attempts=tentativo, rejected=scartati)

        return PathResult(
            reason=(
                f"nessun percorso senza collisioni dopo {attempts} tentativi "
                f"(ultimo esito: {ultimo}).\n"
                f"Ostacoli noti: {', '.join(o.name for o in self.obstacles) or 'nessuno'}.\n"
                + (
                    f"{scartati} percorsi erano stati trovati ma urtavano, e sono "
                    "stati scartati.\n"
                    if scartati
                    else ""
                )
                + "La ricerca e' randomizzata: 'non trovato' non dimostra 'non esiste', "
                "ma con un ostacolo che chiude il passaggio il risultato e' lo stesso."
            ),
            attempts=attempts,
            rejected=scartati,
        )

    def calibrate(self, qpos: np.ndarray, tcp_world: np.ndarray) -> np.ndarray:
        """Impara lo scarto fra il frame di mplib e il frame mondo, misurandolo.

        **Il perche' e' la regola piu' cara del progetto: le quote si leggono,
        non si derivano.** La cinematica di pinocchio lavora nel frame della
        radice dell'URDF, MuJoCo nel frame mondo, e fra i due c'e' la posa della
        base — ma non solo quella: fra ``robot0_base`` e la radice cinematica
        c'e' un offset interno di ~96 mm che nessun documento dichiara.

        Dedurlo dalla posa della base darebbe un errore di quasi dieci
        centimetri, e un errore di frame non si manifesta come un errore: si
        manifesta come un robot che aggira ostacoli che non ci sono e sbatte
        contro quelli veri. Misurarlo costa una chiamata.

        Si passa una configurazione nota e la posizione vera del TCP letta dalla
        scena; lo scarto e' la traslazione da sommare.
        """
        self._offset = np.zeros(3)
        self._offset = np.asarray(tcp_world, dtype=float) - self.tcp_of(qpos)
        return self._offset

    def tcp_of(self, qpos: np.ndarray) -> np.ndarray:
        """Dove si trova il TCP in una data configurazione, nel frame mondo.

        Serve a tradurre un percorso — che e' fatto di **angoli di giunto** — in
        punti che il controller OSC sa inseguire, perche' OSC comanda la posa
        della pinza e non i giunti.

        La traduzione perde qualcosa: due configurazioni diverse possono avere
        lo stesso TCP con il gomito da due parti opposte. E' il motivo per cui i
        punti intermedi si seguono fitti — inseguire solo il primo e l'ultimo
        rimetterebbe il braccio in linea retta, che e' cio' che si voleva
        evitare.
        """
        art = self.planner.planning_world.get_planned_articulations()[0]
        vecchio = art.get_qpos()
        art.set_qpos(self.planner.pad_move_group_qpos(np.asarray(qpos, dtype=float), art), True)
        pos = np.array(
            art.get_pinocchio_model().get_link_pose(self.planner.move_group_link_id).p
        )
        art.set_qpos(vecchio, True)
        return pos + self._offset

    def _collisions_along(self, waypoints: np.ndarray) -> int:
        """Quante configurazioni del percorso sono in collisione.

        Si controllano i punti consegnati, uno per uno. Non e' una prova
        assoluta — fra due punti consecutivi ci puo' stare un urto — ma i punti
        sono fitti (decine su un movimento) e questo intercetta il caso reale
        misurato, dove le configurazioni sporche sono una decina di fila.
        """
        return sum(1 for q in waypoints if self.in_collision(q))

    def in_collision(self, qpos: np.ndarray) -> list:
        """Le collisioni della configurazione con l'ambiente, se ce ne sono.

        Si restituisce la **lista** invece di un booleano perche' contiene quali
        link toccano cosa: e' la differenza fra "no" e "no, perche' il gomito
        urta il muro". Un rifiuto che non dice il motivo costringe a indovinare,
        ed e' il contrario di come rifiuta ``checks.py`` (D2).
        """
        return self.planner.check_for_env_collision(np.asarray(qpos, dtype=float))

    def is_path_clear(self, qpos_from: np.ndarray, qpos_to: np.ndarray, steps: int = 20) -> bool:
        """Il segmento diretto fra due configurazioni e' libero?

        E' la domanda che ``checks.py`` **non** sa porsi. Si campiona lungo il
        segmento e si controlla ogni configurazione: e' grossolano, ma dice
        subito se il percorso in linea retta basta — e nella maggior parte dei
        casi basta, quindi vale la pena chiederlo prima di pianificare.

        Campionare non e' una prova: fra due campioni ci puo' stare un urto. Con
        venti punti su un movimento da mezzo metro si controlla ogni 25 mm, che
        e' fitto abbastanza per un ostacolo di quelle dimensioni e resta
        **un'approssimazione**, non una garanzia.
        """
        a = np.asarray(qpos_from, dtype=float)
        b = np.asarray(qpos_to, dtype=float)

        for t in np.linspace(0.0, 1.0, steps):
            if self.in_collision(a + (b - a) * t):
                return False
        return True


def prepare_panda_urdf(force: bool = False) -> Path:
    """L'URDF del Panda in una forma che mplib sa leggere.

    Tre inciampi risolti qui, e vale la pena scriverli perche' ciascuno e'
    costato un tentativo:

    1. **L'URDF di robosuite usa ``package://panda_description/``**, che mplib
       non sa risolvere.
    2. **mplib risolve i percorsi mesh relativamente alla cartella dell'URDF**,
       sempre — anche se li si riscrive assoluti. Quindi non basta sostituire
       il prefisso: le mesh devono essere *accanto* all'URDF.
    3. **mplib scrive file accanto all'URDF che legge** (l'SRDF generato). Con
       l'originale, scriverebbe dentro ``site-packages``.

    Da qui la cartella in ``out/cache``: l'URDF ripulito piu' un collegamento
    alle mesh. Si costruisce una volta; il calcolo dell'SRDF che ne segue costa
    ~40 secondi e viene riusato.
    """
    import robosuite

    dest = CACHE / "mplib_panda"
    urdf_out = dest / "panda.urdf"

    if urdf_out.is_file() and not force:
        return urdf_out

    base = Path(robosuite.__file__).parent / "models/assets/bullet_data/panda_description"
    src = base / "urdf/panda_arm_hand.urdf"
    if not src.is_file():
        raise FileNotFoundError(
            f"URDF del Panda non trovato in {src}. "
            "Serve robosuite installato (`uv sync`)."
        )

    dest.mkdir(parents=True, exist_ok=True)

    # I percorsi diventano relativi alla cartella dell'URDF, dove mettiamo le
    # mesh: e' l'unica forma che mplib risolve correttamente.
    testo = src.read_text().replace("package://panda_description/", "")
    urdf_out.write_text(testo)

    meshes = dest / "meshes"
    if not meshes.exists():
        try:
            meshes.symlink_to(base / "meshes", target_is_directory=True)
        except (OSError, NotImplementedError):
            # Su filesystem che non supportano i link (NTFS via WSL senza
            # privilegi) si copia: sono ~10 MB, si paga una volta.
            shutil.copytree(base / "meshes", meshes)

    return urdf_out


def obstacles_from_env(env) -> list[Obstacle]:
    """Gli ostacoli di una scena robosuite, come scatole.

    Si riconoscono dal prefisso ``obstacle_``, la stessa convenzione con cui
    ``checks.py`` riconosce i geom afferrabili (``cube_``, ``obj_``). Un nome che
    dichiara il ruolo evita una tabella da tenere aggiornata a parte — e una
    tabella da aggiornare a mano e' una tabella che prima o poi diverge.
    """
    import mujoco

    model = env.sim.model._model
    data = env.sim.data._data
    trovati: list[Obstacle] = []

    for gid in range(model.ngeom):
        nome = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not nome or not nome.startswith("obstacle_"):
            continue
        trovati.append(
            Obstacle(
                name=nome,
                center=np.array(data.geom_xpos[gid]),
                half_sizes=np.array(model.geom_size[gid][:3]),
            )
        )

    return trovati
