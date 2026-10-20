"""Controller locale: dal percorso alla velocita' di adesso.

**La divisione del lavoro, che e' quella di Nav2 e non e' arbitraria.**

Il *planner globale* conosce tutta la mappa, non vede niente in tempo reale, e
risponde una volta sola a *"esiste una strada?"*. Il *controller locale* non
conosce la mappa, vede il LiDAR adesso, e risponde venti volte al secondo a
*"che velocita' do in questo istante?"*.

Sono problemi diversi e il confine e' netto: un ostacolo comparso dopo la
pianificazione lo risolve il locale, perche' il globale non sa che esiste. Se
il locale non ce la fa -- il corridoio e' chiuso, non solo ingombro -- **lo
dichiara**, e la missione ripianifica. Un controller che gira in tondo
provando all'infinito e' un sistema che si e' arreso senza dirlo (D14).

**L'algoritmo: pure pursuit con evitamento reattivo.** Pure pursuit e' il
controller da inseguimento di percorso piu' semplice che funziona davvero, ed
e' lo scheletro di quello che in Nav2 e' il ``RegulatedPurePursuitController``:
si guarda un punto del percorso a una certa distanza davanti (il *lookahead*),
e ci si punta. Sopra ci va l'evitamento: se il LiDAR vede qualcosa nella
direzione di marcia, la velocita' voluta viene deviata e ridotta.

**Perche' pure pursuit e non DWB/MPPI.** Quelli campionano traiettorie e le
valutano: sono migliori, e costano. Su 2 core, a 20 Hz, con la finestra 3D
aperta, non e' il posto dove spendere. E soprattutto: il pezzo che dimostra
qualcosa qui e' *che l'architettura regge*, non che il controller sia ottimo.
Sostituirlo tocca questo file solo -- ed e' il punto della scomposizione.

**Cosa la base olonoma cambia.** Un robot differenziale deve puntare il muso
dove va: la direzione di marcia e l'orientamento sono la stessa cosa. Qui no.
Il robot puo' andare a sinistra guardando avanti, il che rende l'inseguimento
molto piu' semplice (niente manovre) e permette di **scegliere l'orientamento
per un altro motivo** -- per esempio tenere il muso verso la stazione di
arrivo, o verso l'ostacolo da sorvegliare.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..sim.amr.base import MAX_ANG_SPEED, MAX_LIN_SPEED, _wrap
from ..sim.amr.lidar import Scan

# Quanto guardare avanti sul percorso. Troppo corto e il robot oscilla
# inseguendo ogni curva; troppo lungo e taglia gli angoli, che con gli
# scaffali a mezzo metro significa toccarli. 0.6 m e' circa un corpo di robot.
LOOKAHEAD = 0.6

# Quanto vicino bisogna arrivare perche' l'obiettivo sia raggiunto.
#
# **0.45 m, e il valore viene da un conflitto misurato.** Le stazioni stanno
# davanti agli scaffali, quindi dentro la fascia gonfiata della costmap: il
# planner sposta il bersaglio alla cella libera piu' vicina -- 35-42 cm, lo
# dichiara -- e il robot arriva li'. Con tolleranza 0.15 m si fermava a
# 0.23-0.40 m dal bersaglio *nominale* e si dichiarava bloccato, tre volte su
# tre, pur essendo praticamente arrivato.
#
# Non e' un fallimento mascherato: il robot **e'** alla stazione. Una
# stazione e' un disco di 0.9 m di diametro, e fermarsi a 45 cm dal centro
# significa starci dentro. La tolleranza va commisurata al bersaglio vero,
# non alla precisione che il controller saprebbe raggiungere.
#
# Il rischio opposto -- dichiarare vittoria lontano, il flag di D2 -- resta
# limitato: 0.45 m e' meno del raggio della stazione.
GOAL_TOLERANCE = 0.45
GOAL_ANGLE_TOLERANCE = 0.15  # rad, ~8.6 gradi

# Sotto questa distanza da un ostacolo il robot si ferma del tutto. E' il
# margine di sicurezza, e va **oltre** il raggio del robot: un AMR industriale
# ha una zona di arresto protettiva, non rasenta.
STOP_DISTANCE = 0.45

# Entro questa distanza il robot rallenta proporzionalmente. E' cio' che
# rende l'evitamento morbido invece che un freno di emergenza.
SLOW_DISTANCE = 1.2

# Mezza larghezza del corridoio di marcia: un ostacolo conta solo se cade
# **dentro la striscia** che il robot sta per percorrere.
#
# **E' il corpo del robot, non il corpo piu' un margine.** Raggio circoscritto
# misurato dalle mesh: 0.3404 m. Il primo valore era 0.44 -- corpo piu' 10 cm
# "per sicurezza" -- e quel margine in piu' ha rotto l'aggiramento: dovendo
# passare accanto a un cassone su un percorso che la costmap aveva **gia'**
# verificato sicuro, il controller vedeva lo spigolo dentro la propria
# striscia e si fermava. Robot immobile, percorso valido, quattro
# ripianificazioni identiche.
#
# La regola generale, ed e' la seconda volta che si presenta oggi: **il
# margine di sicurezza sta in un posto solo.** Lo mette la costmap, che
# gonfia gli ostacoli di ``DEFAULT_INFLATION``. Se anche il controller ne
# aggiunge uno, i due livelli si contraddicono e il piu' prudente vince
# sempre -- cioe' il robot non si muove.
#
# **Non e' un settore angolare**, ed e' la differenza che ha fatto funzionare
# la cosa: vedi la nota in ``_regulate``.
CORRIDOR_HALF_WIDTH = 0.34

# Guadagno proporzionale sull'orientamento. Basso: girare non e' urgente,
# perche' una base olonoma non ha bisogno di essere girata per andare.
ANGLE_GAIN = 1.2


class Outcome(Enum):
    """Come e' finito un tick di controllo.

    Ogni valore e' un **evento distinto**, non una sfumatura di "non ancora".
    Mescolarli renderebbe un fallimento non attribuibile -- la lezione dei tre
    contatori separati del nastro (D17).
    """

    RUNNING = "in marcia"
    ARRIVED = "arrivato"
    BLOCKED = "bloccato"  # ostacolo fermo davanti: serve ripianificare
    NO_PROGRESS = "fermo"  # si muove ma non avanza: oscillazione o incastro


@dataclass
class ControlResult:
    """L'uscita di un tick: la velocita' e il perche'."""

    velocity: tuple[float, float, float]
    outcome: Outcome
    reason: str = ""
    distance_to_goal: float = 0.0
    nearest_obstacle: float = np.inf


@dataclass
class PurePursuit:
    """Insegue un percorso, rallentando per cio' che il LiDAR vede.

    Tiene poco stato, e quello che tiene e' **del controller**, non del mondo:
    a che punto del percorso e' arrivato, e da quanti tick non fa progressi.
    La posa si rilegge sempre dalla scena.
    """

    path: list[tuple[float, float]]
    goal_theta: float | None = None
    lookahead: float = LOOKAHEAD
    goal_tolerance: float = GOAL_TOLERANCE
    stop_distance: float = STOP_DISTANCE

    _index: int = field(default=0, init=False)
    _stuck_ticks: int = field(default=0, init=False)
    _last_pos: np.ndarray | None = field(default=None, init=False)
    _blocked_ticks: int = field(default=0, init=False)

    # Quanti tick di fila senza progresso prima di dichiararlo. A 20 Hz, 40
    # tick = 2 secondi. Non uno: un singolo tick fermo e' normale (si sta
    # girando). E' la stessa asimmetria di D19 -- nascere e' immediato,
    # morire richiede pazienza.
    stuck_limit: int = 40
    # **Bloccato si dichiara in fretta: mezzo secondo.** Il primo valore era
    # 30 tick (1,5 s), e misurato era troppo: il robot continuava ad avanzare
    # rallentando mentre il cassone gli si parava davanti, e quando finalmente
    # chiedeva un percorso nuovo era gia' a 30 cm dall'ostacolo -- dove
    # nessun percorso aiuta, perche' e' dentro la fascia gonfiata.
    #
    # Aspettare ha senso per NO_PROGRESS (puo' essere una rotazione in corso,
    # vedi ``stuck_limit``); non ne ha per BLOCKED, dove il LiDAR sta gia'
    # dicendo che c'e' un oggetto fermo. **Le due attese sono diverse perche'
    # le due cause sono diverse.**
    blocked_limit: int = 10

    def __post_init__(self) -> None:
        if len(self.path) < 1:
            raise ValueError("il percorso e' vuoto: non c'e' niente da inseguire")
        self._path = np.asarray(self.path, dtype=float)

    @property
    def goal(self) -> np.ndarray:
        return self._path[-1]

    @property
    def progress(self) -> float:
        """Frazione del percorso completata, 0..1. Solo per mostrarla."""
        return self._index / max(1, len(self._path) - 1)

    def _lookahead_point(self, pos: np.ndarray) -> np.ndarray:
        """Il punto da inseguire: il primo oltre ``lookahead`` davanti a noi.

        L'indice non torna mai indietro. Senza quel vincolo, un percorso che
        si ripiega (un'inversione a U attorno a uno scaffale) farebbe
        agganciare il robot al ramo di ritorno, che gli passa accanto: il
        robot taglierebbe la U e andrebbe dritto nello scaffale.
        """
        # Il punto piu' vicino **fra quelli non ancora superati**. Cercarlo su
        # tutto il percorso farebbe agganciare il robot al ramo di ritorno di
        # un'inversione a U, che gli passa accanto: taglierebbe la U e
        # andrebbe dritto nello scaffale.
        tail = self._path[self._index :]
        self._index += int(np.argmin(np.linalg.norm(tail - pos, axis=1)))

        # Il primo punto oltre il lookahead. Se non ce n'e' -- siamo
        # nell'ultimo tratto -- si punta al traguardo: e' l'unico modo di
        # arrivarci, perche' un lookahead che resta indietro lascerebbe il
        # robot a orbitare intorno alla fine del percorso.
        for i in range(self._index, len(self._path)):
            if np.linalg.norm(self._path[i] - pos) >= self.lookahead:
                return self._path[i]
        return self._path[-1]

    def _regulate(
        self, scan: Scan | None, desired: np.ndarray, theta: float
    ) -> tuple[np.ndarray, float]:
        """Rallenta la velocita' voluta secondo cio' che il LiDAR vede.

        **Rallenta soltanto: non devia mai.** E' la regola di
        ``RegulatedPurePursuitController`` di Nav2, e la sua documentazione la
        dichiara senza mezzi termini: *"un pure pursuit controller e' quello:
        insegue **puramente** il percorso, senza interesse o preoccupazione
        per gli ostacoli dinamici"*. Se la strada e' bloccata, il controller
        **si ferma** e lascia ripianificare al planner.

        **Il primo tentativo faceva l'opposto, ed era sbagliato due volte.**

        *Sbagliato tecnicamente:* era un campo di potenziale -- l'ostacolo
        spinge via il robot -- e i campi di potenziale hanno un difetto noto
        da decenni, i **minimi locali**. Misurato mentre succedeva: davanti
        allo spigolo di ``scaffale_c`` il robot avanzava di **4 cm in 30
        tick**, respinto e riattratto. Prima ancora, la spinta era abbastanza
        forte da **invertire** la marcia e mandarlo contro il muro opposto.

        *Sbagliato architetturalmente, ed e' la lezione vera:* dare al
        controller il diritto di scegliere da che parte passare significa
        mettere due autorita' sulla stessa decisione. Il percorso lo decide
        il planner, che ha la mappa intera; il controller vede solo l'adesso.
        Il robot combatteva contro il piano che aveva appena ricevuto.

        La divisione giusta -- e quella che il ``runner`` gia' implementava:

        - **planner**: decide *dove* passare
        - **controller**: decide *quanto veloce*, e se non passa **lo dice**
        - **orchestratore**: quando sente ``BLOCKED``, ripianifica

        Ritorna ``(velocita' regolata, distanza dell'ostacolo davanti)``.
        """
        if scan is None:
            return desired, np.inf

        speed = float(np.linalg.norm(desired))
        if speed < 1e-6:
            return desired, np.inf

        # --- quanto lontano posso arrivare senza toccare? -------------------
        #
        # **La domanda giusta non e' "cosa c'e' vicino" ma "se proseguo,
        # sbatto?".** Sembra la stessa cosa e non lo e', ed e' il terzo
        # tentativo su questo pezzo -- vale scrivere anche i due sbagliati,
        # perche' entrambi erano plausibili:
        #
        # 1. *settore angolare di +/-60 gradi*: il robot non partiva. Nella
        #    stazione di ricarica, in un angolo, vedeva i muri a 83 cm e
        #    frenava per ostacoli che stava **costeggiando**. Un AMR di
        #    magazzino passa la vita a 80 cm dai muri.
        # 2. *corridoio di marcia + distanza euclidea*: il robot si fermava
        #    a meta' corsia. Misurato: passando accanto all'angolo di
        #    ``scaffale_c``, il punto piu' vicino nel corridoio distava 0.336
        #    m -- ma era **di lato** (``lat=0.432`` su un corridoio di 0.44).
        #    La distanza euclidea a un ostacolo laterale e' piccola mentre la
        #    strada davanti e' libera: il numero era giusto e rispondeva a
        #    un'altra domanda.
        #
        # Quello che fa Nav2 e' proiettare la traiettoria in avanti e
        # verificare le collisioni lungo di essa
        # (``max_allowed_time_to_collision_up_to_carrot``). Qui, con una base
        # olonoma che va dritta, la traiettoria e' un segmento: per ogni
        # ritorno si calcola **quanto si puo' avanzare prima che entri nel
        # corpo del robot**, che e' la proiezione lungo la marcia meno la
        # correzione per lo scostamento laterale.
        #
        # Un ostacolo a ``lat`` vicino al bordo del corpo non ferma: ci si
        # passa accanto. Uno in mezzo alla strada ferma.
        heading = np.arctan2(desired[1], desired[0])
        idx_all = np.flatnonzero(scan.hits)
        if len(idx_all) == 0:
            return desired, np.inf
        a_world = scan.angles[idx_all] + theta
        rng = scan.ranges[idx_all]
        rel = np.array([_wrap(a - heading) for a in a_world])
        ahead = rng * np.cos(rel)
        lateral = np.abs(rng * np.sin(rel))

        # Solo cio' che sta davanti e dentro la larghezza del corpo puo'
        # essere colpito andando dritti.
        blocking = (ahead > 0) & (lateral <= CORRIDOR_HALF_WIDTH)
        if not blocking.any():
            return desired, np.inf

        # Quanto si avanza prima del contatto: la distanza lungo la marcia,
        # scalata dallo spazio laterale che resta. Un punto proprio sul bordo
        # (``lateral == CORRIDOR_HALF_WIDTH``) non e' un ostacolo: gli si
        # passa a fianco, e la sua distanza utile e' infinita.
        margin = 1.0 - lateral[blocking] / CORRIDOR_HALF_WIDTH
        # sotto il 15% di margine laterale si sfiora appena: non frena
        usable = margin > 0.15
        if not usable.any():
            return desired, np.inf

        d_near = float(ahead[blocking][usable].min())
        if d_near >= SLOW_DISTANCE:
            return desired, d_near

        if d_near <= self.stop_distance:
            # fermo: la direzione resta quella giusta, la velocita' e' zero.
            # Il tick dopo, se nulla cambia, si dichiara BLOCKED e tocca al
            # planner. **Non si cerca una via d'uscita qui.**
            return np.zeros(2), d_near

        # Rallenta proporzionalmente fra stop_distance e SLOW_DISTANCE.
        # E' l'euristica di prossimita' di Nav2
        # (``use_cost_regulated_linear_velocity_scaling``): la **direzione
        # non si tocca**, cambia solo il modulo.
        scale = (d_near - self.stop_distance) / (SLOW_DISTANCE - self.stop_distance)
        return desired * scale, d_near

    def step(self, pose: np.ndarray, scan: Scan | None = None) -> ControlResult:
        """Un tick: da posa (+ LiDAR) alla velocita' nel frame mondo.

        ``pose`` e' ``(x, y, theta)``. Che arrivi da fuori e non venga letta
        qui dentro e' voluto: questo modulo non sa cosa sia MuJoCo, quindi il
        giorno che la posa venisse da un sistema di localizzazione vero non
        cambierebbe una riga. E' lo stesso motivo per cui ``perception/``
        riceve la maschera invece di chiederla (D18).
        """
        pos = np.asarray(pose[:2], dtype=float)
        theta = float(pose[2])
        to_goal = self.goal - pos
        dist = float(np.linalg.norm(to_goal))

        # --- arrivato? -----------------------------------------------------
        if dist <= self.goal_tolerance:
            if self.goal_theta is None:
                return ControlResult((0.0, 0.0, 0.0), Outcome.ARRIVED, "in posizione", dist)
            dth = _wrap(self.goal_theta - theta)
            if abs(dth) <= GOAL_ANGLE_TOLERANCE:
                return ControlResult((0.0, 0.0, 0.0), Outcome.ARRIVED, "in posa", dist)
            # ferma la traslazione, gira soltanto
            w = float(np.clip(ANGLE_GAIN * dth, -MAX_ANG_SPEED, MAX_ANG_SPEED))
            return ControlResult((0.0, 0.0, w), Outcome.RUNNING, "orientamento finale", dist)

        # --- la velocita' voluta: verso il punto di lookahead ---------------
        target = self._lookahead_point(pos)
        d = target - pos
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            d, n = to_goal, max(dist, 1e-9)

        # rallenta avvicinandosi al traguardo, cosi' non lo supera
        speed = min(MAX_LIN_SPEED, max(0.15, dist * 1.5))
        desired = d / n * speed

        # --- evitamento ----------------------------------------------------
        v_xy, d_obs = self._regulate(scan, desired, theta)

        # --- orientamento ---------------------------------------------------
        #
        # **La traslazione non aspetta la rotazione.** E' il punto della base
        # olonoma, ed e' l'errore piu' facile da fare qui: far girare il robot
        # verso la direzione di marcia *come se* fosse differenziale.
        #
        # Misurato mentre succedeva: il robot restava fermo a oscillare, con
        # ``v_theta`` saturata fra +1.5 e -1.5 e l'indice del percorso fermo a
        # 0 per 600 tick. L'anello era questo -- la direzione voluta si
        # calcola dalla velocita', la velocita' e' nel frame mondo,
        # l'orientamento la insegue, e girando cambia cio' che insegue.
        #
        # Qui il muso punta **verso il bersaglio finale**, che e' un
        # riferimento fisso e non dipende da come il robot e' girato adesso.
        # Serve a due cose concrete: mettere LiDAR e camere nella direzione
        # utile, e rendere leggibile a occhio dove il robot "sta andando".
        # Ma **non condiziona la traslazione**: se il muso e' storto, il robot
        # va lo stesso, di lato.
        if dist > self.goal_tolerance * 2:
            want = float(np.arctan2(to_goal[1], to_goal[0]))
        else:
            want = theta if self.goal_theta is None else self.goal_theta
        w = float(np.clip(ANGLE_GAIN * _wrap(want - theta), -MAX_ANG_SPEED, MAX_ANG_SPEED))

        # --- diagnosi: fermo, o bloccato? -----------------------------------
        # Le due cose hanno cause diverse e rimedi diversi, quindi sono
        # contatori separati. "Bloccato" = c'e' un ostacolo, serve un altro
        # percorso. "Fermo" = nessun ostacolo eppure non si avanza, quindi il
        # problema e' il controller (oscillazione) o la geometria.
        #
        # **Si chiede aiuto mentre si rallenta, non dopo essersi fermati.**
        # La prima versione richiedeva ``v_xy ~ 0``, cioe' il robot doveva
        # essere gia' fermo: quando finalmente dichiarava BLOCKED era a 30 cm
        # dall'ostacolo, **dentro la fascia gonfiata della costmap**, dove
        # nessun percorso nuovo puo' aiutarlo perche' non esiste una cella
        # libera da cui ripartire.
        #
        # La soglia e' ``stop_distance * 1.5``: il robot sta ancora
        # avanzando, ha spazio davanti, e un percorso alternativo lo puo'
        # ancora usare. Misurato che funziona: col cassone in mezzo al
        # corridoio, il percorso aggirante costa 62 cm in piu' ed esiste --
        # ma solo se lo si chiede in tempo.
        if d_obs <= self.stop_distance * 1.5:
            self._blocked_ticks += 1
            if self._blocked_ticks >= self.blocked_limit:
                return ControlResult(
                    (0.0, 0.0, 0.0),
                    Outcome.BLOCKED,
                    f"ostacolo fermo a {d_obs:.2f} m da {self._blocked_ticks} tick: "
                    "serve un percorso diverso",
                    dist,
                    d_obs,
                )
        else:
            self._blocked_ticks = 0

        if self._last_pos is not None and np.linalg.norm(pos - self._last_pos) < 0.005:
            self._stuck_ticks += 1
        else:
            self._stuck_ticks = 0
        self._last_pos = pos.copy()

        if self._stuck_ticks >= self.stuck_limit:
            return ControlResult(
                (0.0, 0.0, 0.0),
                Outcome.NO_PROGRESS,
                f"fermo da {self._stuck_ticks} tick a {dist:.2f} m dal bersaglio "
                f"(ostacolo piu' vicino: {d_obs:.2f} m)",
                dist,
                d_obs,
            )

        return ControlResult(
            (float(v_xy[0]), float(v_xy[1]), w),
            Outcome.RUNNING,
            "",
            dist,
            d_obs,
        )
