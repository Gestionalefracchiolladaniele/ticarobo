"""Scena di lavoro come ambiente robosuite.

Sostituisce la costruzione manuale di ``scene.py`` per tutto cio' che riguarda
**l'esecuzione**. Il motivo e' D5: il controllo del braccio e della pinza lo
fa robosuite, che ha OSC e presa gia' risolti e testati. Qui si descrive solo
*cosa* c'e' nella scena, non *come* si muove il robot.

La scena e' la stessa di prima nei contenuti — un tavolo, un cubo rosso, un
cubo blu, una zona verde di destinazione, un ostacolo fisso — ma costruita con
i mattoni di robosuite (``TableArena``, ``BoxObject``), cosi' che il robot sia
un ``Robot`` robosuite completo di controller OSC.

**Frame:** robosuite mette il piano del tavolo a z=0.8 nel frame mondo e la
base del robot a terra davanti al tavolo. Sono quote diverse da quelle della
vecchia scena MjSpec (tavolo a 0.40): il workspace di ``safety/`` va costruito
su queste, non su quelle. ``table_surface_z()`` e' l'unica fonte della quota.
"""

from __future__ import annotations

import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial, new_body, new_element, new_geom
from robosuite.utils.placement_samplers import UniformRandomSampler

from robotica.transforms import camera_xyaxes

# Dimensioni del piano di lavoro e quota della sua **superficie**.
# Attenzione al frame: in robosuite ``table_offset`` e' gia' la superficie del
# tavolo, non il suo centro. Sommarci mezzo spessore sposta tutto 25 mm in
# alto e fa nascere i cubi a mezz'aria. Verificato leggendo ``geom_xpos`` del
# collision geom: 0.775 + 0.025 di mezzo spessore = 0.8.
TABLE_FULL_SIZE = (0.8, 0.8, 0.05)
TABLE_FRICTION = (1.0, 5e-3, 1e-4)
TABLE_OFFSET = (0.0, 0.0, 0.8)

# Lato dei cubi (mezzo lato, come in MuJoCo).
CUBE_HALF = 0.02

# Posizioni iniziali, relative al centro del tavolo. Sono le stesse proporzioni
# della vecchia scena: un cubo a sinistra, uno a destra, zona di posa in fondo.
CUBE_RED_XY = (-0.08, -0.15)
CUBE_BLUE_XY = (0.06, 0.12)
GREEN_ZONE_XY = (0.14, -0.22)
GREEN_ZONE_HALF = 0.06

# --- scena di smistamento: quattro colori, due zone laterali -----------------
#
# I colori sono una **proprieta'** del cubo, non parte del nome. La differenza
# conta per la Fase 3: quando sara' una camera a dire di che colore e' un cubo,
# si sostituisce chi produce il fatto ``color(x, rosso)`` e il resto non cambia.
# Se il colore stesse nel nome (``cube_red_1``), la percezione dovrebbe
# *rinominare* gli oggetti, che e' assurdo.
# Sei cubi, non quattro. La ragione e' misurata, non estetica: con 4 cubi il
# pianificatore trova il piano intero in 2.782 stati, con 5 in 50.360, con 6 si
# arrende dopo 200.000 dicendo "probabilmente irraggiungibile" — su un problema
# che ha una soluzione da 12 passi. Sei cubi e' il punto in cui la
# decomposizione smette di essere un'ottimizzazione e diventa necessaria, e una
# scena che sta sotto la soglia non dimostrerebbe niente.
SORT_COLORS = {
    "cube_a": "rosso",
    "cube_b": "blu",
    "cube_c": "verde",
    "cube_d": "giallo",
    "cube_e": "rosso",
    "cube_f": "verde",
}

# Che colore va in che zona. E' la regola di classificazione, e sta qui perche'
# e' una proprieta' del **compito**, non del pianificatore: cambiare quali
# colori sono buoni non deve richiedere di toccare il planning.
SORT_RULE = {
    "rosso": "buoni",
    "blu": "buoni",
    "verde": "scarti",
    "giallo": "scarti",
}

_COLOR_RGBA = {
    "rosso": (0.85, 0.2, 0.2, 1.0),
    "blu": (0.2, 0.35, 0.85, 1.0),
    "verde": (0.2, 0.7, 0.3, 1.0),
    "giallo": (0.9, 0.75, 0.15, 1.0),
}

# I cubi nascono al **centro**, in fila; le zone stanno ai lati.
#
# Le quote non sono a occhio. Il workspace di ``safety/`` e' il tavolo, quindi
# y resta dentro [-0.40, 0.40]; le zone a y=+-0.26 con semilato 0.07 occupano
# da 0.19 a 0.33, e i cubi stanno fra -0.12 e +0.12. Fra la zona e il cubo piu'
# vicino restano 70 mm: abbastanza perche' nessun cubo nasca gia' dentro una
# zona — altrimenti ``in_zone`` sarebbe vero a scena appena aperta e
# l'obiettivo risulterebbe raggiunto senza muovere nulla.
SORT_CUBE_XY = {
    "cube_a": (-0.18, -0.08),
    "cube_b": (-0.18, 0.08),
    "cube_c": (-0.02, -0.08),
    "cube_d": (-0.02, 0.08),
    "cube_e": (0.14, -0.08),
    "cube_f": (0.14, 0.08),
}

SORT_ZONE_XY = {
    "buoni": (0.0, -0.26),
    "scarti": (0.0, 0.26),
}
SORT_ZONE_HALF = 0.07
SORT_ZONE_RGBA = {
    "buoni": (0.2, 0.75, 0.3, 1.0),
    "scarti": (0.75, 0.45, 0.15, 1.0),
}

# --- la camera fissa (Fase 3, D18) -------------------------------------------
#
# Montaggio **eye-to-hand**: ferma nel mondo, non sul polso. I compiti sono di
# classificazione — servono tutti i cubi insieme per decidere — e una camera
# ferma isola i guasti: un punto 3D sbagliato e' la calibrazione, non anche la
# cinematica o il braccio che si e' mosso durante lo scatto.
#
# Sta a x positiva, **dal lato opposto al robot** (base a x=-0.56): il braccio
# entra nell'inquadratura di lato invece di coprire il bersaglio.
# Inclinata, non a picco: da perpendicolare un cubo e' un quadrato piatto e la
# depth non ne distingue lo spessore.
#
# L'orientamento **non si scrive, si calcola** da dove sta la camera e da cosa
# deve guardare (``_camera_xyaxes``). Il primo tentativo usava ``euler`` scelto
# a occhio e la camera guardava il **soffitto**: asse ottico z=+0.589 invece che
# negativo, e il 100% di pixel non neri non lo rivelava perche' lo sfondo del
# cielo non e' nero. Le quote si leggono, non si derivano — e un orientamento
# indovinato e' una quota derivata.
CAM_POS = (0.75, 0.0, 1.45)
CAM_LOOK_AT = (0.0, 0.0, 0.80)
CAM_FOVY = 45.0

# --- l'ostacolo: un muro fra il cubo e la zona -------------------------------
#
# Le quote non sono decorative: il muro sta **a meta' strada** fra il cubo rosso
# (-0.08, -0.15) e la zona verde (0.14, -0.22), cioe' a (0.03, -0.185). Se
# stesse altrove il robot lo aggirerebbe per caso e la scena non dimostrerebbe
# niente — il punto e' proprio che il percorso diretto ci passa in mezzo.
#
# Alto 12 cm: piu' del cubo sollevato durante il trasporto (il robot alza di
# ~10 cm), cosi' non si scavalca per inerzia. Sottile in x perche' deve essere
# un ostacolo da aggirare, non un recinto.
WALL_XY = (0.03, -0.185)
WALL_HALF = (0.015, 0.11, 0.06)
WALL_RGBA = (0.45, 0.45, 0.5, 1.0)

# --- il nastro trasportatore -------------------------------------------------
#
# **Il nastro non e' un meccanismo: e' una velocita' imposta ai cubi.**
#
# MuJoCo 3.3.7 non ha ``geom/surfacevel``, l'attributo che nelle versioni piu'
# recenti fa trascinare per attrito gli oggetti appoggiati a una superficie in
# movimento. Verificato sulla macchina, non dedotto: l'attributo non esiste in
# questa versione, e salire non si puo' (D6 fissa MuJoCo <3.4, robosuite si
# rompe con la 3.13).
#
# Si impone allora la **velocita'** (``qvel``) e non la posizione (``qpos``), e
# la differenza decide se l'esperimento e' valido. Scrivendo ``qpos`` il cubo
# non si muove: viene *messo* altrove a ogni passo. Quando la pinza lo afferra,
# il nastro continua a spostarlo e la pinza a trattenerlo — due autorita' sulla
# stessa posizione, e il cubo vibra o schizza via. Il fallimento non sarebbe
# piu' attribuibile: colpa del braccio o della simulazione?
#
# Con ``qvel`` la fisica resta l'unica autorita' su dove il cubo finisce. Se la
# pinza stringe, l'attrito vince e il cubo si ferma in mano, come su un nastro
# vero. Vedi ``ConveyorEnv.advance_belt``.
#
# Il nastro corre lungo **y**, perpendicolare al robot, e i cubi entrano dal
# lato +y. Le zone di posa stanno in x, ai lati del nastro, cosi' che posare un
# cubo non richieda di attraversare il nastro stesso.
#
# **Lungo 1,9 m perche' i venti cubi ci devono stare tutti appoggiati.** Con
# 12 cm di distanza l'uno dall'altro, venti cubi occupano 2,3 m di fila: su un
# nastro corto nascerebbero a mezz'aria oltre il bordo e cadrebbero sul tavolo,
# arrivando in disordine. La lunghezza non cambia pero' la **finestra di
# presa**, che resta quella del braccio: allungare il nastro fa vedere la coda
# che arriva, non aumenta il tempo per prendere.
BELT_Y_START = 2.60
"""Dove comincia il nastro, e **da dove entra la coda dei cubi**.

Il numero si ricava dal compito, non a occhio: venti cubi a 12 cm di distanza
occupano 2,28 m di fila, e la fila deve stare **tutta** sul nastro piu' il
tratto di lavoro davanti al braccio.

Con un nastro piu' corto (era 1,25) diciannove cubi su venti nascevano oltre il
bordo e cadevano subito: sul nastro restava solo il primo, il robot lo prendeva
correttamente e poi si fermava perche' non arrivava piu' nulla. Il sintomo
sembrava un blocco del robot, ed era una scena costruita male."""

# **Due costanti, non una**, e la separazione e' costata un bug che non dava
# errori. Prima ``BELT_Y_END`` faceva due lavori: dove finisce il *piano* del
# nastro, e fino a dove il nastro *spinge*. Finche' i due valori coincidevano
# sembrava un'economia elegante. Poi ho esteso la spinta fin dentro la scatola
# — e ho **spostato fisicamente il nastro di 12 cm**, perche' la sua posizione
# si calcola come punto medio fra inizio e fine. Il nastro e' finito sopra la
# rampa, i cubi non ci salivano piu' e cadevano di lato.
#
# Una costante che serve a due scopi e' una trappola: il giorno in cui i due
# scopi divergono, cambiarne uno rompe l'altro **in silenzio**.
BELT_Y_END = -1.02
"""Dove finisce il **piano** del nastro.

**Sporge oltre il bordo del tavolo** (che arriva a -1,00), ed e' voluto: i cubi
non raccolti devono uscire di scena da soli, e un nastro che finisce sul tavolo
li lascerebbe appoggiati li' a ingombrare."""

BELT_PUSH_UNTIL = -1.06
"""Fin dove il nastro **spinge** i cubi.

Si smette **oltre il bordo del tavolo**, non prima. Misurato con la soglia a
-1,00: due cubi su tre cadevano e il terzo restava in bilico a y=-1,001, cioe'
un millimetro oltre la soglia — la spinta era cessata nell'istante in cui il
cubo aveva ancora appoggio, e li' e' rimasto.

Il margine di 6 cm serve a questo: il cubo deve essere **gia' oltre il vuoto**
quando la spinta finisce, non sul ciglio. Una soglia esatta e' una soglia che
qualche cubo colpira' nell'istante sbagliato.

Il punto non e' ovvio, ed e' costato cinque tentativi di costruire un
contenitore di raccolta. Imporre ``qvel`` a ogni passo non *spinge* il cubo: lo
**trasporta a velocita' costante**, senza dargli quantita' di moto e senza
lasciare che la gravita' agisca lungo il tragitto. Misurato su un cubo fermo a
mezz'aria: ``vz`` esattamente 0,0000.

Quindi finche' il nastro spinge, il cubo **non puo' cadere** — e nessuno
scivolo, rampa o cestino puo' funzionare, perche' il cubo arriva in fondo senza
energia da spendere. Smettere di spingere non e' un dettaglio di pulizia: e'
l'unica cosa che restituisce il cubo alla fisica."""
BELT_X = -0.20
"""Dove corre il nastro in x.

**Vicino al robot, non al centro del tavolo**, e il numero viene da una misura.
La base del Panda sta a x=-0,56 e la portata utile e' ~0,62 m: col nastro a
x=0,00 il braccio lavorava a **0,56 m, il 90% della propria estensione**, e per
ogni presa doveva **distendersi quasi del tutto**.

Un braccio steso e' in **configurazione singolare**: ha perso la capacita' di
muoversi in certe direzioni, perche' per spostare la pinza di un centimetro
dovrebbe ruotare enormemente i giunti. OSC comanda forze, non angoli, e da li'
non riesce piu' a muoverlo — il braccio resta fermo e non esegue piu' nulla.

Il sintomo sembra un blocco logico ed e' geometria: si e' visto solo
**guardando la scena**, in una posa che nessun numero di esito raccontava.
A x=-0,20 il nastro dista 0,36 m: il braccio lavora piegato, con margine di
manovra in tutte le direzioni."""
BELT_HALF = (0.09, 0.95, 0.005)
"""Semi-dimensioni del nastro: larghezza, **lunghezza indicativa**, spessore.

La lunghezza qui non viene usata per costruire il nastro — quella si ricava da
``BELT_Y_START`` e ``BELT_Y_END``, che sono l'unica fonte. Resta come
riferimento per chi legge, e per i controlli in x e z."""

BELT_RGBA = (0.25, 0.25, 0.28, 1.0)

# Il tavolo sotto il nastro, lungo quanto il nastro piu' un margine.
#
# **Allungare il tavolo non allunga la portata del braccio**, che resta ~85 cm
# dalla base. Serve solo a reggere il nastro: senza, i cubi in coda
# galleggerebbero. La larghezza resta 80 cm perche' e' quella che il braccio
# copre davvero — allargarla darebbe zone irraggiungibili.
BELT_TABLE_SIZE = (0.8, 1.9, 0.05)
"""Il tavolo sotto il nastro.

**Piu' corto del nastro, ed e' voluto.** Il nastro sporge oltre il bordo da
entrambi i lati: in fondo perche' i cubi non raccolti devono cadere, in cima
perche' la coda dei venti cubi e' lunga 2,28 m e il tavolo non deve seguirla —
sarebbe un tavolo da quattro metri per reggere una fila di cubi.

La larghezza resta 80 cm: e' quella che il braccio copre davvero, e allargarla
darebbe zone irraggiungibili."""

# Il cestino in fondo al nastro: dove finisce cio' che non viene raccolto.
#
# **Non e' decorazione.** Senza, un cubo che supera la finestra di presa cade
# dal bordo e rotola via: non si vede dove finisce, e guardando la scena un
# giallo lasciato passare di proposito e un rosso perso per lentezza sono
# identici. Un cubo che rotola puo' anche tornare sotto il braccio e disturbare
# la presa successiva.
#
# Col cestino i non raccolti si **accumulano in un punto** e si contano a
# occhio: la verifica visiva di cio' che i contatori dichiarano, che e' la
# regola del progetto — il sintomo si guarda dove si manifesta.
#
# E cambia anche il concetto: lo scarto diventa **un posto** invece di
# un'assenza. Un cubo giallo li' dentro e' arrivato a destinazione.
# --- lo scarico: i cubi non raccolti cadono dal bordo ------------------------
#
# **Niente rampa, niente cestino, e il motivo e' una misura — non una rinuncia.**
#
# Ci sono stati cinque tentativi di costruire un contenitore di raccolta, e i
# primi quattro hanno prodotto quattro diagnosi diverse, tutte plausibili e
# tutte incomplete: fondo piu' alto del nastro, bordo sollevato dalla rotazione,
# spinta interrotta troppo presto, una costante che spostava il nastro. Ognuna
# era un vero difetto, e correggerla non bastava.
#
# Il quinto tentativo ha misurato le **velocita'** invece della geometria:
#
#     y=-0.901  z=0.8198  vy=+0.000  vz=+0.0000
#
# ``vz`` esattamente zero. Un oggetto che cade ha sempre velocita' verticale —
# la gravita' la produce a ogni passo. Zero significa che qualcuno la cancella.
#
# **Imporre ``qvel`` a ogni passo non spinge il cubo: lo trasporta a velocita'
# costante.** Non accumula quantita' di moto e non risente della gravita' lungo
# il tragitto, perche' la sua velocita' viene riscritta prima che la fisica
# possa cambiarla. Quando la spinta cessa, il cubo si ferma dove si trova — in
# aria, se li' non c'e' appoggio.
#
# Da qui la conseguenza che nessuna geometria poteva aggirare: **con una
# velocita' imposta, nessuno scivolo puo' funzionare**, perche' il cubo non ha
# inerzia da spendere in discesa. Il contenitore andava costruito prima di
# capire come si muovono i cubi, e questa e' la lezione vera dei cinque
# tentativi.
#
# Cosi' invece funziona, ed e' anche piu' semplice: superata la finestra di
# presa il nastro smette di spingere, **la fisica riprende il controllo** e il
# cubo cade dal bordo. I cubi non raccolti si vedono a terra e si contano; il
# conteggio vero lo fa comunque ``passed_by()``, che non ha mai sbagliato in
# nessuna delle prove.
BIN_WALL = 0.008

# Dove il braccio puo' prendere: una finestra centrata davanti a lui.
#
# Non e' tutto il nastro. Il workspace di ``safety/`` e la portata del braccio
# limitano dove si arriva, e un cubo che esce da questa finestra e' **perso**:
# e' il terzo contatore, quello che dice se il nastro va troppo veloce.
BELT_REACH_Y = (-0.16, 0.16)

# Le due zone di posa, ai lati del nastro. Stessa forma delle zone di
# smistamento, spostate in x perche' il nastro occupa il centro.
# Le due zone, **entrambe a portata comoda del braccio**.
#
# La prima versione le metteva simmetriche rispetto al *nastro*: la zona
# "buoni" finiva a x=+0,22, cioe' a **0,79 m dalla base** — oltre i 0,62 m di
# portata utile. Il robot ci arrivava lo stesso, stendendosi tutto, e restava
# in quella posa da cui non riusciva piu' a muoversi.
#
# Simmetriche rispetto al **robot**, non al nastro: una davanti (piu' vicina),
# una di lato. Distano 0,33 e 0,39 m dalla base, ben dentro la portata.
BELT_ZONE_XY = {
    "buoni": (-0.02, -0.22),
    "scarti": (-0.02, 0.22),
}
BELT_ZONE_HALF = 0.08
BELT_ZONE_RGBA = {
    "buoni": (0.2, 0.75, 0.3, 1.0),
    "scarti": (0.75, 0.45, 0.15, 1.0),
}

# La regola di smistamento sul nastro, e **la novita' rispetto a SORT_RULE**:
# un colore che non va da nessuna parte.
#
# Il giallo si lascia passare. Concettualmente non e' un dettaglio: finora ogni
# cubo che compariva era un compito, e il robot non aveva modo di **decidere di
# non agire**. Non agire non e' fallire — un cubo giallo che esce dal nastro e'
# un successo, un cubo rosso che fa lo stesso e' un errore. I contatori devono
# distinguerli, altrimenti misurano il contrario di cio' che serve.
#
# I tre colori noti tengono il ruolo che avevano in ``SORT_RULE``: cambiare la
# scena *e* la regola insieme renderebbe un fallimento non attribuibile.
BELT_RULE = {
    "rosso": "buoni",
    "blu": "buoni",
    "verde": "scarti",
    "giallo": None,  # si lascia passare
}


def table_surface_z() -> float:
    """Quota della superficie del tavolo nel frame mondo, in metri."""
    return TABLE_OFFSET[2]



class WorkspaceEnv(ManipulationEnv):
    """Tavolo con due cubi, una zona di posa e un ostacolo.

    Non definisce una ricompensa utile: non si allena nulla qui (D7). Serve
    come *motore di esecuzione* verificabile — il robot con OSC, gli oggetti,
    e le quote note per la verifica di sicurezza.

    ``_check_success`` e' comunque implementato, perche' e' il criterio con cui
    il pick-and-place si dichiara riuscito: il cubo rosso appoggiato dentro la
    zona verde, non semplicemente "la pinza si e' chiusa".
    """

    def __init__(
        self,
        robots: str = "Panda",
        randomize_objects: bool = False,
        **kwargs,
    ) -> None:
        self.table_full_size = TABLE_FULL_SIZE
        self.table_friction = TABLE_FRICTION
        self.table_offset = np.array(TABLE_OFFSET)
        self.randomize_objects = randomize_objects
        self.placement_initializer = None

        kwargs.setdefault("use_camera_obs", False)
        kwargs.setdefault("has_offscreen_renderer", False)
        kwargs.setdefault("control_freq", 20)
        kwargs.setdefault("horizon", 2000)

        super().__init__(robots=robots, **kwargs)

    # --- costruzione della scena --------------------------------------------

    def _load_model(self) -> None:
        super()._load_model()

        # La base del robot va posizionata rispetto al tavolo, altrimenti il
        # braccio nasce dentro il piano.
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])

        self.cube_red = BoxObject(
            name="cube_red",
            size_min=[CUBE_HALF] * 3,
            size_max=[CUBE_HALF] * 3,
            rgba=[0.85, 0.2, 0.2, 1.0],
            material=_solid_material("redwood", "WoodRed"),
            friction=[1.0, 0.02, 0.001],
        )
        self.cube_blue = BoxObject(
            name="cube_blue",
            size_min=[CUBE_HALF] * 3,
            size_max=[CUBE_HALF] * 3,
            rgba=[0.2, 0.35, 0.85, 1.0],
            material=_solid_material("bluewood", "WoodBlue"),
            friction=[1.0, 0.02, 0.001],
        )
        self.cubes = [self.cube_red, self.cube_blue]

        self._build_placement_initializer()

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=self.cubes,
        )

        _add_green_zone(self.model, self.table_offset)

    def _build_placement_initializer(self) -> None:
        """Dove nascono i cubi a ogni reset.

        Con ``randomize_objects=False`` i cubi partono sempre nello stesso
        punto: e' cio' che rende il pick-and-place *ripetibile* e quindi
        confrontabile fra un'esecuzione e l'altra. La randomizzazione serve
        piu' avanti, per misurare la robustezza (Fase 4).
        """
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.cubes,
            x_range=[-0.02, 0.02] if self.randomize_objects else [0.0, 0.0],
            y_range=[-0.02, 0.02] if self.randomize_objects else [0.0, 0.0],
            rotation=None if self.randomize_objects else 0.0,
            rotation_axis="z",
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.001,
        )
        # Ogni cubo ha il proprio riferimento: il sampler somma il range a
        # questo centro, quindi con range nullo il cubo cade esattamente qui.
        self.placement_initializer.reset()

    def _setup_references(self) -> None:
        super()._setup_references()
        self.cube_body_ids = {
            obj.name: self.sim.model.body_name2id(obj.root_body) for obj in self.cubes
        }

    def _reset_internal(self) -> None:
        super()._reset_internal()

        # Posa deterministica: si scrive direttamente il freejoint di ciascun
        # cubo. Passare dal sampler per due oggetti fissi aggiungerebbe solo
        # indirezione, e il punto qui e' la ripetibilita'.
        for obj, (dx, dy) in (
            (self.cube_red, CUBE_RED_XY),
            (self.cube_blue, CUBE_BLUE_XY),
        ):
            pos = self.table_offset + np.array([dx, dy, CUBE_HALF])
            self.sim.data.set_joint_qpos(
                obj.joints[0], np.concatenate([pos, [1.0, 0.0, 0.0, 0.0]])
            )

    # --- stato ---------------------------------------------------------------

    def object_pos(self, name: str) -> np.ndarray:
        """Posizione (x, y, z) di un cubo nel frame mondo."""
        if name not in self.cube_body_ids:
            raise KeyError(
                f"oggetto '{name}' non nella scena; presenti: {sorted(self.cube_body_ids)}"
            )
        return np.array(self.sim.data.body_xpos[self.cube_body_ids[name]])

    def object_names(self) -> list[str]:
        """Nomi degli oggetti manipolabili."""
        return sorted(self.cube_body_ids)

    def green_zone_center(self) -> np.ndarray:
        """Centro della zona di posa, sulla superficie del tavolo."""
        return self.table_offset + np.array([GREEN_ZONE_XY[0], GREEN_ZONE_XY[1], 0.0])

    def eef_pos(self) -> np.ndarray:
        """Posizione corrente del TCP (centro delle dita) nel frame mondo."""
        return np.array(self.sim.data.site_xpos[self.robots[0].eef_site_id["right"]])

    def _check_success(self) -> bool:
        """Il cubo rosso e' appoggiato dentro la zona verde?"""
        return self.object_in_zone("cube_red")

    def object_in_zone(self, name: str, tol: float = GREEN_ZONE_HALF) -> bool:
        """L'oggetto e' dentro la zona di posa e appoggiato sul tavolo?

        Si verificano entrambe le cose: un cubo ancora in mano sopra la zona
        non e' "posato". La quota attesa e' il piano piu' il mezzo lato.
        """
        pos = self.object_pos(name)
        zone = self.green_zone_center()
        in_xy = bool(np.all(np.abs(pos[:2] - zone[:2]) <= tol))
        # Appoggiato significa centro a un mezzo lato dal piano. La tolleranza
        # assorbe l'affondamento di contatto del solutore.
        on_table = abs(pos[2] - (table_surface_z() + CUBE_HALF)) < 0.02
        return in_xy and on_table

    def reward(self, action=None) -> float:
        """Ricompensa binaria. Non si allena nulla (D7): serve solo come esito."""
        return 1.0 if self._check_success() else 0.0



class SortingEnv(WorkspaceEnv):
    """Quattro cubi di quattro colori al centro, due zone di smistamento ai lati.

    E' la stessa scena di ``WorkspaceEnv`` in tutto cio' che riguarda il robot —
    stesso tavolo, stesso controller OSC, stessa sicurezza — e cambia solo
    **cosa c'e' sopra**. Ereditare invece di copiare non e' pigrizia: garantisce
    che le quote (``table_offset``, il mezzo spessore che in Fase 0 e' costato
    ogni presa) restino una sola definizione.

    La differenza di sostanza e' che qui il compito e' **classificazione**: non
    "porta questo cubo li'", ma "ogni cubo nella zona che gli spetta in base a
    una proprieta'". Il colore e' quella proprieta', ed e' interrogabile con
    ``color_of``. In Fase 3 l'informazione arrivera' da una camera invece che da
    un dizionario, e sara' l'unica cosa da sostituire.
    """

    def __init__(self, robots: str = "Panda", **kwargs) -> None:
        self.colors = dict(SORT_COLORS)
        super().__init__(robots=robots, **kwargs)

    # --- costruzione ---------------------------------------------------------

    def _load_model(self) -> None:
        # Non si chiama ``WorkspaceEnv._load_model``: costruirebbe i due cubi
        # della scena base. Si risale a ``ManipulationEnv``, che prepara il
        # robot e nient'altro.
        ManipulationEnv._load_model(self)

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])

        self.cubes = [
            BoxObject(
                name=name,
                size_min=[CUBE_HALF] * 3,
                size_max=[CUBE_HALF] * 3,
                rgba=list(_COLOR_RGBA[colore]),
                friction=[1.0, 0.02, 0.001],
            )
            for name, colore in SORT_COLORS.items()
        ]

        self._build_placement_initializer()

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=self.cubes,
        )

        for zona, (zx, zy) in SORT_ZONE_XY.items():
            _add_zone_marker(
                self.model,
                name=zona,
                xy=(zx, zy),
                half=SORT_ZONE_HALF,
                rgba=SORT_ZONE_RGBA[zona],
                z=self.table_offset[2] + 0.001,
            )

        _add_overhead_camera(self.model, self.table_offset)

    def _reset_internal(self) -> None:
        # Stesso motivo di sopra: si salta la posa dei due cubi della scena base.
        ManipulationEnv._reset_internal(self)

        for obj in self.cubes:
            dx, dy = SORT_CUBE_XY[obj.name]
            pos = self.table_offset + np.array([dx, dy, CUBE_HALF])
            self.sim.data.set_joint_qpos(
                obj.joints[0], np.concatenate([pos, [1.0, 0.0, 0.0, 0.0]])
            )

    # --- stato ---------------------------------------------------------------

    def color_of(self, name: str) -> str:
        """Il colore di un cubo.

        Oggi lo dice un dizionario, perche' la scena la costruiamo noi e il
        colore lo sappiamo. In Fase 3 lo dira' una camera: **questo metodo e' il
        punto di sostituzione**, e per questo esiste invece di leggere il
        dizionario dove serve.
        """
        if name not in self.colors:
            raise KeyError(f"oggetto '{name}' non nella scena; presenti: {sorted(self.colors)}")
        return self.colors[name]

    def zone_centers(self) -> dict[str, np.ndarray]:
        """Le zone di smistamento, col centro sulla superficie del tavolo."""
        return {
            nome: self.table_offset + np.array([xy[0], xy[1], 0.0])
            for nome, xy in SORT_ZONE_XY.items()
        }

    def zone_for(self, name: str) -> str:
        """In che zona deve finire questo cubo, secondo la regola di colore."""
        return SORT_RULE[self.color_of(name)]

    def green_zone_center(self) -> np.ndarray:
        """Qui non esiste una "zona verde": il concetto e' sostituito dalle zone.

        Si solleva invece di restituire qualcosa di plausibile, perche' un
        valore inventato si propagherebbe in silenzio fin dentro un movimento.
        """
        raise AttributeError(
            "SortingEnv non ha una zona verde unica: usa zone_centers() "
            f"(zone disponibili: {', '.join(sorted(SORT_ZONE_XY))})"
        )

    def object_in_zone(self, name: str, zone: str | None = None, tol: float | None = None) -> bool:
        """Il cubo e' posato dentro una zona?

        Senza ``zone`` si intende **la zona che gli spetta**, che e' la domanda
        che conta per lo smistamento: non "e' in una zona", ma "e' in quella
        giusta".
        """
        zone = zone or self.zone_for(name)
        tol = SORT_ZONE_HALF if tol is None else tol

        pos = self.object_pos(name)
        center = self.zone_centers()[zone]
        in_xy = bool(np.all(np.abs(pos[:2] - center[:2]) <= tol))
        on_table = abs(pos[2] - (table_surface_z() + CUBE_HALF)) < 0.02
        return in_xy and on_table

    def _check_success(self) -> bool:
        """Riuscito quando **ogni** cubo e' nella zona del suo colore."""
        return all(self.object_in_zone(n) for n in self.object_names())



class ObstacleEnv(WorkspaceEnv):
    """La scena base piu' un **muro** fra il cubo e la zona di posa.

    Esiste per mostrare un buco, non per esercitare una funzione. Oggi
    ``safety/`` verifica il **punto di arrivo** e il robot ci va in linea retta:
    con un muro in mezzo la verifica approva e il braccio ci sbatte contro. E'
    il tipo di fallimento che non si vede finche' la scena e' vuota, e che
    nessun test sul punto finale puo' scoprire.

    Il muro e' **solido** (collide davvero), a differenza dei riquadri delle
    zone che sono solo disegnati. Un ostacolo che non collide non proverebbe
    niente: il braccio ci passerebbe attraverso e sembrerebbe tutto a posto.
    """

    def __init__(self, robots: str = "Panda", **kwargs) -> None:
        super().__init__(robots=robots, **kwargs)

    def _load_model(self) -> None:
        super()._load_model()
        _add_wall(self.model, self.table_offset)

    def wall_center(self) -> np.ndarray:
        """Centro del muro nel frame mondo."""
        return self.table_offset + np.array(
            [WALL_XY[0], WALL_XY[1], WALL_HALF[2]]
        )

    def wall_half_sizes(self) -> np.ndarray:
        """Mezze dimensioni del muro: quanto spazio occupa davvero."""
        return np.array(WALL_HALF)



class ConveyorEnv(WorkspaceEnv):
    """Un nastro che porta cubi colorati davanti al robot, due zone ai lati.

    **Cosa cambia rispetto a ``SortingEnv``, e perche' serve py_trees.** Nello
    smistamento i cubi stanno fermi: si legge lo stato, si calcola un piano, lo
    si esegue. Qui il mondo si muove mentre il piano viene eseguito, e **un
    piano calcolato all'inizio scade prima di finire**. Non e' una differenza di
    difficolta': e' una differenza di natura. Pianificare dodici passi su una
    scena che fra dodici passi non esiste piu' e' pianificare su una finzione.

    **Il terzo esito.** ``BELT_RULE`` ha un colore che non va in nessuna zona:
    il giallo si lascia passare. Finora ogni cubo era un compito; qui il robot
    deve poter **decidere di non agire**, e non agire non e' fallire. Un cubo
    giallo che esce dal nastro e' un successo.

    **Le due misure restano separate** (vedi ``ConveyorStats``): decidere in
    tempo e afferrare sono capacita' diverse, e trasferiscono al reale in modo
    diverso — la decisione all'85-98%, la presa no. Un numero unico non direbbe
    di chi e' la colpa.
    """

    def __init__(
        self,
        robots: str = "Panda",
        colors: list[str] | None = None,
        belt_speed: float = 0.03,
        spacing: float = 0.12,
        **kwargs,
    ) -> None:
        """``colors`` e' la sequenza che il nastro portera', in ordine.

        Si passa dall'esterno invece di fissarla nella scena perche' la
        sequenza **e' l'esperimento**: un cubo solo per misurare la velocita',
        venti per vedere come si comporta il robot. Una scena che sa gia' quali
        cubi avra' costringerebbe a scriverne una per prova.
        """
        self._sequenza = list(colors) if colors else ["rosso"]
        self.belt_speed = float(belt_speed)
        self.spacing = float(spacing)
        self.colors = {f"cube_{i:02d}": c for i, c in enumerate(self._sequenza)}
        # Prima di ``super().__init__``: la costruzione fa gia' dei passi di
        # simulazione, e ``step`` legge questi due attributi.
        self._held: str | None = None
        self._usciti: set[str] = set()

        # **L'orizzonte va alzato, e il motivo e' misurato.** robosuite chiude
        # l'episodio dopo ``horizon`` passi e poi solleva su ogni ``step``. Il
        # default di ``WorkspaceEnv`` e' 2000, che basta per uno smistamento a
        # scena ferma e non per il nastro: un ciclo di presa misurato costa 19
        # secondi, cioe' 380 passi a 20 Hz, e venti cubi ne chiedono almeno
        # 7600. Con 2000 la scena si chiude a meta' esperimento — e il sintomo
        # e' un'eccezione a meta' corsa, non un numero sbagliato.
        kwargs.setdefault("horizon", 30000)

        super().__init__(robots=robots, **kwargs)

    # --- costruzione ---------------------------------------------------------

    def _load_model(self) -> None:
        # Come ``SortingEnv``: si risale a ``ManipulationEnv`` per non costruire
        # i due cubi della scena base.
        ManipulationEnv._load_model(self)

        # Il tavolo lungo si impone **qui**, prima di costruire l'arena:
        # ``WorkspaceEnv.__init__`` fissa ``table_full_size`` e l'arena nasce in
        # questo metodo, quindi e' l'unico punto in cui il cambio ha effetto.
        self.table_full_size = BELT_TABLE_SIZE

        # La base del robot resta calcolata sulla **larghezza** (x), che non e'
        # cambiata: allungare il tavolo in y non deve spostare il braccio, o si
        # perderebbero tutte le quote verificate finora.
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](TABLE_FULL_SIZE[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        arena.set_origin([0, 0, 0])

        self.cubes = [
            BoxObject(
                name=nome,
                size_min=[CUBE_HALF] * 3,
                size_max=[CUBE_HALF] * 3,
                rgba=list(_COLOR_RGBA[colore]),
                friction=[1.0, 0.02, 0.001],
            )
            for nome, colore in self.colors.items()
        ]

        self._build_placement_initializer()

        self.model = ManipulationTask(
            mujoco_arena=arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=self.cubes,
        )

        _add_belt(self.model, self.table_offset)

        for zona, (zx, zy) in BELT_ZONE_XY.items():
            _add_zone_marker(
                self.model,
                name=zona,
                xy=(zx, zy),
                half=BELT_ZONE_HALF,
                rgba=BELT_ZONE_RGBA[zona],
                z=self.table_offset[2] + 0.001,
            )

    def _reset_internal(self) -> None:
        ManipulationEnv._reset_internal(self)

        # I cubi nascono in fila sul nastro, **e ci devono stare tutti**.
        #
        # Sembra ovvio e non lo era: con venti cubi a 12 cm si occupano 2,28 m
        # di fila, il nastro e' lungo 2,27, e partendo dall'inizio (y=+1,25)
        # l'ultimo finiva a y=+3,53 — due metri oltre il bordo. **Diciannove
        # cubi su venti nascevano nel vuoto e cadevano subito**, lasciando sul
        # nastro solo il primo: il robot lo prendeva, faceva la cosa giusta, e
        # poi restava fermo perche' non arrivava piu' niente.
        #
        # Il sintomo sembrava un blocco del robot. Era una scena costruita male.
        #
        # La spaziatura si **ricava** dallo spazio disponibile invece di essere
        # un numero fisso: cosi' il vincolo "ci stanno tutti" e' garantito per
        # costruzione, e non e' una promessa da mantenere a mano ogni volta che
        # si cambia il numero di cubi o la lunghezza del nastro.
        z = self.table_offset[2] + BELT_HALF[2] * 2 + CUBE_HALF

        # Si parte appena sopra la finestra di presa e si va indietro: il primo
        # cubo e' gia' quasi a portata, gli altri in coda dietro di lui.
        primo = BELT_REACH_Y[1] + 0.12
        disponibile = BELT_Y_START - primo
        n = max(len(self.cubes) - 1, 1)
        passo = min(self.spacing, disponibile / n)

        for i, obj in enumerate(self.cubes):
            pos = np.array([BELT_X, primo + i * passo, z])
            self.sim.data.set_joint_qpos(
                obj.joints[0], np.concatenate([pos, [1.0, 0.0, 0.0, 0.0]])
            )

        self._usciti: set[str] = set()
        self._held = None

    # --- il nastro -----------------------------------------------------------

    def step(self, action):
        """Un passo di simulazione, **col nastro che avanza**.

        Il nastro si muove qui e non in un ciclo esterno, e la ragione e' il
        cuore dell'esperimento. ``robot.pick()`` chiama ``env.step()`` decine di
        volte al suo interno: un nastro fatto avanzare solo dal chiamante
        resterebbe **fermo per tutti i ~9 secondi di una presa**.

        Sarebbe la rovina della prova. Il punto del nastro e' che *il mondo si
        muove mentre il piano viene eseguito*; se si ferma ogni volta che il
        braccio lavora, ogni cubo aspetta educatamente il suo turno e si sta
        misurando uno smistamento normale con un nastro decorativo. I contatori
        direbbero "20 su 20" e non significherebbe nulla — un numero ottimo che
        misura la cosa sbagliata.

        Avvolgendo ``step`` il nastro avanza **chiunque** lo chiami, e
        dimenticarsene non e' piu' possibile.
        """
        self.advance_belt(held=self._held)
        return super().step(action)

    def set_held(self, name: str | None) -> None:
        """Dichiara quale cubo e' in mano, perche' il nastro non lo spinga.

        Si dichiara invece di dedurlo dalla pinza: dedurlo richiederebbe di
        interrogare la presa a ogni passo, e nell'istante fra "le dita toccano"
        e "la presa tiene" la risposta e' ambigua. Un cubo spinto dal nastro
        mentre il braccio lo solleva verrebbe strappato di mano.
        """
        self._held = name

    def advance_belt(self, held: str | None = None) -> None:
        """Un passo di nastro: impone la velocita' ai cubi che ci stanno sopra.

        **Si scrive ``qvel``, non ``qpos``,** ed e' la scelta che rende
        l'esperimento valido. Imporre la posizione significherebbe *mettere* il
        cubo altrove a ogni passo: quando la pinza lo afferra, nastro e pinza si
        contenderebbero la stessa posizione e il cubo vibrerebbe o schizzerebbe
        via — un fallimento non piu' attribuibile al braccio o alla scena.

        Imponendo la velocita', la fisica resta l'unica autorita' su dove il
        cubo finisce: se la pinza stringe, l'attrito vince e il cubo si ferma in
        mano, come su un nastro vero.

        ``held`` e' il cubo in mano, se c'e': su quello **non si interviene**.
        Continuare a spingerlo significherebbe tirare un cubo che non e' piu'
        sul nastro, e il braccio lo trascinerebbe contro la propria presa.
        """
        for obj in self.cubes:
            nome = obj.name
            if nome == held or nome in self._usciti:
                continue

            pos = self.object_pos(nome)

            # Fuori dal nastro in x, o gia' posato in una zona: non e' piu'
            # merce in transito e il nastro non lo riguarda.
            if abs(pos[0] - BELT_X) > BELT_HALF[0]:
                continue
            # Si smette di spingere a **scarico avvenuto**, non dove finisce il
            # piano: ``BELT_PUSH_UNTIL`` sta dentro la scatola, ``BELT_Y_END``
            # dove comincia la rampa. Sono due cose diverse (vedi le costanti).
            if pos[1] < BELT_PUSH_UNTIL:
                self._usciti.add(nome)
                continue

            addr = self.sim.model.get_joint_qvel_addr(obj.joints[0])
            inizio = addr[0] if isinstance(addr, tuple) else addr
            # Solo la componente y: x e z restano alla fisica, cosi' il cubo
            # puo' comunque cadere o essere spostato di lato.
            self.sim.data.qvel[inizio + 1] = -self.belt_speed

    # --- stato ---------------------------------------------------------------

    def color_of(self, name: str) -> str:
        """Il colore di un cubo. In Fase 3 lo dira' una camera (D15)."""
        if name not in self.colors:
            raise KeyError(f"oggetto '{name}' non nella scena; presenti: {sorted(self.colors)}")
        return self.colors[name]

    def zone_for(self, name: str) -> str | None:
        """In che zona va questo cubo, o ``None`` se si lascia passare.

        ``None`` non e' un errore ne' un valore mancante: e' **un esito**. Il
        colore che non va in nessuna zona e' il caso che distingue "non ho
        agito" da "ho fallito".
        """
        return BELT_RULE[self.color_of(name)]

    def zone_centers(self) -> dict[str, np.ndarray]:
        return {
            nome: self.table_offset + np.array([xy[0], xy[1], 0.0])
            for nome, xy in BELT_ZONE_XY.items()
        }

    def green_zone_center(self) -> np.ndarray:
        raise AttributeError(
            "ConveyorEnv non ha una zona verde unica: usa zone_centers() "
            f"(zone disponibili: {', '.join(sorted(BELT_ZONE_XY))})"
        )

    def on_belt(self, name: str) -> bool:
        """Il cubo e' ancora sul nastro, dentro i suoi bordi?"""
        pos = self.object_pos(name)
        return (
            abs(pos[0] - BELT_X) <= BELT_HALF[0]
            and BELT_PUSH_UNTIL <= pos[1] <= BELT_Y_START + len(self.cubes) * self.spacing
        )

    def in_reach(self, name: str) -> bool:
        """Il cubo e' nella finestra in cui il braccio puo' prenderlo?

        E' la domanda che il nastro aggiunge: non "dove sta", ma "sta ancora
        dove posso arrivare". Un cubo che esce da questa finestra e' **perso**,
        e perso e' un esito diverso da fallito.
        """
        y = float(self.object_pos(name)[1])
        return BELT_REACH_Y[0] <= y <= BELT_REACH_Y[1]

    def passed_by(self, name: str) -> bool:
        """Il cubo ha superato la finestra di presa senza essere preso."""
        return float(self.object_pos(name)[1]) < BELT_REACH_Y[0]

    def fallen_off(self, name: str) -> bool:
        """Il cubo e' caduto dal tavolo: non raccolto, uscito di scena.

        Serve a **verificare guardando** cio' che i contatori dichiarano. I
        cubi a terra si contano a occhio nella finestra 3D, e devono essere
        tanti quanti ``lasciati + persi``. Se i due numeri divergono, uno dei
        due sta mentendo — e un contatore che si autocertifica non e' una prova.

        Si guarda la **quota** e non la posizione: un cubo sotto il piano del
        tavolo e' caduto, ovunque sia finito rotolando.
        """
        return float(self.object_pos(name)[2]) < self.table_offset[2] - 0.10

    def fallen_cubes(self) -> list[str]:
        """Quali cubi sono caduti dal tavolo, adesso."""
        return [n for n in self.object_names() if self.fallen_off(n)]

    def object_in_zone(self, name: str, zone: str | None = None, tol: float | None = None) -> bool:
        """Il cubo e' posato dentro una zona?

        Senza ``zone`` si intende quella che gli spetta. **Per un colore da
        lasciar passare la risposta e' sempre falsa**: non c'e' una zona in cui
        dovrebbe stare, e rispondere "si'" da qualche parte sarebbe inventare
        un successo.

        Il controllo sul ``None`` non e' difensivo, e' necessario: senza,
        ``zone_centers()[None]`` solleva ``KeyError`` a meta' simulazione, e il
        messaggio non dice niente su quale sia il problema vero — che il
        chiamante sta chiedendo dove vada un cubo che non va da nessuna parte.
        """
        zone = zone or self.zone_for(name)
        if zone is None:
            return False

        tol = BELT_ZONE_HALF if tol is None else tol
        pos = self.object_pos(name)
        center = self.zone_centers()[zone]
        in_xy = bool(np.all(np.abs(pos[:2] - center[:2]) <= tol))
        on_table = abs(pos[2] - (table_surface_z() + CUBE_HALF)) < 0.03
        return in_xy and on_table

    def _check_success(self) -> bool:
        """Riuscito quando ogni cubo da smistare e' nella sua zona, **e** ogni
        cubo da lasciar passare e' effettivamente passato.

        Le due condizioni sono diverse e servono entrambe: un robot che prende
        tutto, gialli compresi, e li posa bene non ha fatto il compito — ha
        fatto un compito piu' facile.
        """
        for nome in self.object_names():
            if self.zone_for(nome) is None:
                if not self.passed_by(nome):
                    return False
            elif not self.object_in_zone(nome):
                return False
        return True


def _solid_material(name: str, texture: str) -> CustomMaterial:
    """Materiale a tinta piatta, leggibile anche col rendering software."""
    return CustomMaterial(
        texture=texture,
        tex_name=name,
        mat_name=f"{name}_mat",
        tex_attrib={"type": "cube"},
        mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
    )



def _add_zone_marker(
    task: ManipulationTask,
    name: str,
    xy: tuple[float, float],
    half: float,
    rgba: tuple[float, float, float, float],
    z: float,
) -> None:
    """Una zona di posa: un riquadro colorato disegnato sul tavolo.

    E' puramente visiva (``contype=0``, ``conaffinity=0``): non deve collidere
    con nulla, serve a dare un bersaglio nominabile e a renderlo visibile nelle
    immagini delle camere.
    """
    body = new_body(name=f"zone_{name}", pos=[xy[0], xy[1], z])
    body.append(
        new_geom(
            name=f"zone_{name}_geom",
            type="box",
            size=[half, half, 0.001],
            rgba=list(rgba),
            group=1,
            contype=0,
            conaffinity=0,
        )
    )
    task.worldbody.append(body)



def _camera_xyaxes(
    pos: tuple[float, float, float], look_at: tuple[float, float, float]
) -> str:
    """L'attributo ``xyaxes`` di una camera che da ``pos`` guarda ``look_at``,
    come stringa XML. Il calcolo e' ``transforms.camera_xyaxes``."""
    return " ".join(f"{v:.6f}" for v in camera_xyaxes(np.asarray(pos), np.asarray(look_at)))


def _add_overhead_camera(
    task: ManipulationTask,
    table_offset: np.ndarray,
    name: str = "overhead",
) -> None:
    """La camera fissa che guarda il tavolo. Montaggio **eye-to-hand** (D18).

    Sta in worldbody, non su un corpo del robot: la sua posa nel mondo
    resta **costante**, ed e' cio' che rende T_base_camera calibrabile una
    volta sola. Attaccata al polso sarebbe eye-in-hand, e la posa andrebbe
    ricomposta dalla cinematica a ogni scatto.

    **Sta dal lato opposto al robot** (x positiva). E' il difetto noto
    dell'eye-to-hand: il braccio si mette fra camera e oggetto proprio quando
    arriva a prenderlo. Da qui il braccio entra nell'inquadratura di lato
    invece che coprire il bersaglio.

    **Inclinata, non a picco.** Da perpendicolare un cubo appare come un
    quadrato piatto e la depth non distingue lo spessore; di sbieco si vede
    anche la faccia laterale.

    Il fovy e' 45 gradi perche' intrinsics() in render.py ricava la
    focale proprio da li': cambiarlo cambia la matrice K, quindi e' una quota
    che si **legge dal modello**, non si assume.

    Si usa ``xyaxes`` e non ``euler``: gli angoli di Eulero dipendono
    dall'ordine e dalle convenzioni, e il primo tentativo con ``euler`` guardava
    il soffitto. ``xyaxes`` dichiara **dove puntano gli assi**, e si ricava
    dalle due posizioni: non c'e' conversione, quindi non c'e' niente da
    sbagliare.
    """
    body = new_body(
        name=f"cam_{name}_mount",
        pos=[CAM_POS[0], CAM_POS[1], CAM_POS[2]],
    )
    body.append(
        new_element(
            tag="camera",
            name=name,
            mode="fixed",
            pos="0 0 0",
            xyaxes=_camera_xyaxes(CAM_POS, CAM_LOOK_AT),
            fovy=str(CAM_FOVY),
        )
    )
    task.worldbody.append(body)


def _add_wall(task: ManipulationTask, table_offset: np.ndarray) -> None:
    """Il muro: un ostacolo **solido**, non un disegno.

    A differenza dei riquadri delle zone, questo ha ``contype`` e
    ``conaffinity`` di default, quindi collide davvero. E' il punto: un ostacolo
    attraversabile non proverebbe niente, perche' il braccio ci passerebbe in
    mezzo e sembrerebbe tutto a posto.

    Il nome inizia per ``obstacle_`` perche' ``safety/`` riconosce i geom
    afferrabili dal prefisso (``cube_``, ``obj_``): un ostacolo non e' afferrabile
    e toccarlo dev'essere una collisione, non lo scopo.
    """
    body = new_body(
        name="obstacle_wall",
        pos=[WALL_XY[0], WALL_XY[1], table_offset[2] + WALL_HALF[2]],
    )
    body.append(
        new_geom(
            name="obstacle_wall_geom",
            type="box",
            size=list(WALL_HALF),
            rgba=list(WALL_RGBA),
            group=1,
        )
    )
    task.worldbody.append(body)



def _add_belt(task: ManipulationTask, table_offset: np.ndarray) -> None:
    """Il piano del nastro: una lastra sottile su cui i cubi scorrono.

    **E' solido** (i cubi ci si appoggiano davvero e non cadono attraverso), ma
    non si muove: il movimento lo fa ``advance_belt`` imponendo la velocita' ai
    cubi. Un nastro che si muove per davvero richiederebbe ``surfacevel``, che
    MuJoCo 3.3.7 non ha.

    L'attrito laterale e' basso perche' il cubo deve scorrere in y senza
    incepparsi, e quello frontale alto abbastanza da non farlo slittare quando
    la pinza lo tocca di striscio.

    **Posizione e lunghezza si ricavano entrambe dagli estremi**, invece di
    essere due numeri indipendenti. Prima la lunghezza stava in ``BELT_HALF`` e
    la posizione si calcolava dal punto medio: due fonti per la stessa
    geometria, che sono rimaste coerenti finche' non ho toccato un estremo — e
    li' il nastro si e' spostato di 12 cm finendo sopra la rampa, senza che
    niente segnalasse l'incoerenza.
    """
    # Gli estremi sono l'unica fonte: centro e semi-lunghezza ne discendono.
    centro = (BELT_Y_START + BELT_Y_END) / 2.0
    semi = (BELT_Y_START - BELT_Y_END) / 2.0

    body = new_body(
        name="belt",
        pos=[BELT_X, centro, table_offset[2] + BELT_HALF[2]],
    )
    body.append(
        new_geom(
            name="belt_geom",
            type="box",
            size=[BELT_HALF[0], semi, BELT_HALF[2]],
            rgba=list(BELT_RGBA),
            group=1,
            friction=[0.4, 0.005, 0.0001],
        )
    )
    task.worldbody.append(body)

    # Due righe chiare ai bordi della finestra di presa. Sono solo disegnate
    # (``contype=0``): servono a **vedere** dove il braccio arriva, che e' il
    # confine fra un cubo preso e un cubo perso.
    for lato, y in (("in", BELT_REACH_Y[1]), ("out", BELT_REACH_Y[0])):
        marker = new_body(name=f"reach_{lato}", pos=[BELT_X, y, table_offset[2] + 0.012])
        marker.append(
            new_geom(
                name=f"reach_{lato}_geom",
                type="box",
                size=[BELT_HALF[0], 0.004, 0.001],
                rgba=[0.9, 0.9, 0.2, 0.8],
                group=1,
                contype=0,
                conaffinity=0,
            )
        )
        task.worldbody.append(marker)



def _add_green_zone(task: ManipulationTask, table_offset: np.ndarray) -> None:
    """La zona verde della scena base. E' un caso particolare di zona."""
    _add_zone_marker(
        task,
        name="green",
        xy=GREEN_ZONE_XY,
        half=GREEN_ZONE_HALF,
        rgba=(0.2, 0.75, 0.3, 1.0),
        z=table_offset[2] + 0.001,
    )
