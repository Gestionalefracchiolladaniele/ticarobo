"""Il magazzino: pavimento, muri, scaffali, stazioni. Scena MuJoCo diretta.

**Frame.** Origine al centro del magazzino, ``x`` avanti, ``y`` a sinistra,
``z`` in alto (MuJoCo). Il pavimento sta a ``z=0`` e la base del robot ci
appoggia sopra: il modello TidyBot ha il proprio ``base_link`` gia' alla quota
giusta, quindi **non si somma niente a mano**. E' la lezione di
``table_offset`` (25 mm di errore) applicata prima di sbagliare.

**Perche' un XML costruito da stringhe** invece di un file statico: le quote
del magazzino devono essere leggibili come costanti Python, perche' la
navigazione ci ragiona sopra (la griglia, le zone, il varco stretto). Un XML
statico obbligherebbe a scriverle due volte, e due copie della stessa quota
divergono.

**Il varco stretto e' voluto.** ``AISLE_NARROW`` e' piu' largo del robot ma
piu' stretto del robot + margine: e' il caso che un planner che ragiona sul
*punto* invece che sul *corpo* attraversa allegramente, e in cui il robot vero
si incastra. Serve a dimostrare che la verifica guarda l'ingombro (D16 nella
sua forma per AMR).
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from robotica.paths import MENAGERIE

# --- il magazzino -----------------------------------------------------------
#
# Unita' SI: metri. Il capannone e' un rettangolo chiuso da quattro muri.
# 10x8 m e' abbastanza grande da avere corridoi veri (il robot e' ~0.5 m) e
# abbastanza piccolo da stare in una finestra 3D senza zoom.
HALL_SIZE = (10.0, 8.0)

# **Le quote verticali di un magazzino vero.** Un capannone di logistica ha
# scaffalature da 4-6 m e un soffitto a 7-8: le prime versioni di questa scena
# avevano muri da 1.2 m e scaffali da 1.0, e vista dall'alto sembrava una
# piantina colorata invece di un posto.
#
# **Non cambia niente per la navigazione**, ed e' il punto: il planner ragiona
# in pianta, il LiDAR sta su un piano solo. L'altezza e' cio' che rende la
# scena leggibile a un occhio umano, e un esperimento che non si guarda non
# si capisce (D18).
WALL_HEIGHT = 6.0
WALL_THICK = 0.15
CEILING_HEIGHT = 7.0

# Gli scaffali. La **collisione** e' un blocco pieno: alla navigazione
# interessa l'ingombro in pianta, non i ripiani. La **vista** e' la struttura
# vera, montanti e traverse, perche' e' cio' che fa sembrare un magazzino un
# magazzino.
#
# 4.5 m e' l'altezza di una scaffalatura porta-pallet da magazzino medio.
#
# Posizione = centro in (x, y); dimensioni = (lunghezza, profondita') piene.
SHELF_HEIGHT = 4.5


@dataclass(frozen=True)
class Shelf:
    """Uno scaffale: dove sta e quanto e' grande in pianta."""

    name: str
    center: tuple[float, float]
    size: tuple[float, float]


# Quattro scaffali in due file, con un corridoio centrale largo e due laterali.
# Il corridoio fra le due file e' la via principale; quelli contro i muri sono
# le scorciatoie, e uno dei due e' **stretto apposta** (vedi AISLE_NARROW).
SHELVES = (
    Shelf("scaffale_a", (-2.5, 2.0), (3.0, 0.8)),
    Shelf("scaffale_b", (2.5, 2.0), (3.0, 0.8)),
    Shelf("scaffale_c", (-2.5, -2.0), (3.0, 0.8)),
    Shelf("scaffale_d", (2.5, -2.0), (3.0, 0.8)),
)

# Il varco stretto: lo spazio fra il bordo di uno scaffale e il muro.
# scaffale_a va da y=1.6 a y=2.4; il muro nord sta a y=4.0.
# Quindi il passaggio libero e' 4.0 - 2.4 = 1.6 m -- comodo.
# Per avere un varco davvero stretto si mette un blocco fisso che lo chiude
# quasi del tutto: resta AISLE_NARROW di luce.
#
# **Il valore e' scelto sulla misura del robot, non a occhio.** Letto dalle
# mesh del menagerie: la base e' quadrata 0.4814 m, quindi raggio inscritto
# 0.2407 e circoscritto 0.3404 -- passa in **0.681 m** comunque orientata.
#
# 0.72 m e' il caso che separa due domande diverse:
#   - "il corpo ci entra?"      -> si', con 2 cm di luce totale
#   - "ci si passa in sicurezza?" -> no: la costmap gonfiata lo chiude
#
# **La soglia e' stata misurata, non dedotta** (ricerca binaria, 14 passi):
# con celle da 5 cm il planner accetta il varco da **0.750 m** in su. La
# teoria direbbe 0.681 (= 2 x raggio circoscritto): la differenza di 7 cm e'
# discretizzazione, e la costmap sbaglia dalla parte **conservativa**.
# Vedi ``inflate`` in ``navigation/costmap.py`` per il perche'.
#
# Primo tentativo: 0.62 m -- **sbagliato**, sotto l'ingombro del robot (lo
# avrebbe reso un muro, e un muro dimostra solo che A* evita i muri). Era
# scritto prima di leggere le quote, ed e' l'ottava volta che succede.
AISLE_NARROW = 0.66

# Quanto il varco dev'essere largo perche' il planner lo accetti. **Misurato
# per ricerca binaria sulla scena vera**, non teorico.
#
# Il valore e' cambiato una volta, ed e' istruttivo: era 0.750 con i muri a
# 0.10 m di spessore, e' diventato 0.700 portandoli a 0.15. Il gonfiaggio
# parte dai bordi degli ostacoli, e spostare un muro sposta la soglia.
#
# **Una soglia misurata va rimisurata quando la scena cambia.** Il primo
# AISLE_NARROW (0.72) era sotto 0.75 e quindi rifiutato; con la soglia a
# 0.70 sarebbe diventato percorribile, e il varco avrebbe smesso di
# dimostrare qualcosa senza che nessun test lo dicesse.
AISLE_THRESHOLD_MEASURED = 0.70
_NARROW_GAP_CENTER = (-2.5, 3.2)

# Le stazioni: dove il robot preleva, dove consegna, dove si carica.
# Sono **posizioni nominali**, non geometria: il robot ci si ferma davanti.
# Che non siano geom e' una scelta -- una stazione e' un posto, e un posto non
# deve poter far collidere niente.
STATIONS: dict[str, tuple[float, float]] = {
    "banchina": (-4.2, 0.0),
    "scaffale_a": (-2.5, 1.2),
    "scaffale_b": (2.5, 1.2),
    "scaffale_c": (-2.5, -1.2),
    "scaffale_d": (2.5, -1.2),
    "spedizione": (4.2, 0.0),
    "ricarica": (-4.2, -3.2),
}

# La stazione di ricarica ha un nome speciale perche' la logica della batteria
# la interroga: e' l'unico posto dove l'energia risale.
CHARGER = "ricarica"

# Raggio del disco di una stazione. **E' anche il volume in cui la percezione
# si aspetta di trovare qualcosa**: chi guarda una stazione cerca dentro
# questo raggio e non oltre. Senza, il volume andrebbe scelto a occhio -- e
# misurato: a +/-0.90 m entra ``muro_ovest`` (34.517 px contro i 1.734 del
# pacco), che diventa la detection piu' grande. Un numero derivato dalla
# comodita' invece che dalla scena e' ``ARM_REACH`` ricavato dal tavolo (D17).
STATION_RADIUS = 0.45

_STATION_RGBA = "0.20 0.60 0.85 0.35"
_CHARGER_RGBA = "0.20 0.85 0.35 0.45"


def _textures_dir() -> str:
    """Le texture di robosuite, riusate qui.

    **Perche' prenderle da li' invece di generarle.** Sono le stesse che
    danno la resa delle scene del braccio (``TableArena`` le usa per
    pavimento, muri e tavolo): riusarle fa si' che il magazzino **somigli**
    alle altre scene del progetto invece di sembrare un altro software.

    Sono PNG gia' su disco, e robosuite e' gia' una dipendenza: non si
    scarica niente. Se un giorno robosuite sparisse, ``_FALLBACK`` qui sotto
    tiene in piedi la scena con texture procedurali -- piu' povere ma
    funzionanti, perche' una scena che non carica e' peggio di una brutta.
    """
    import os

    import robosuite

    return os.path.join(os.path.dirname(robosuite.__file__), "models", "assets", "textures")


def _assets_xml() -> str:
    """Materiali e texture. Separati dalla geometria perche' sono estetica.

    ``reflectance``, ``shininess`` e ``specular`` sono cio' che distingue una
    superficie da un colore piatto: senza, ogni faccia rende lo stesso grigio
    da ogni angolo e la scena sembra disegnata invece che illuminata.
    """
    from pathlib import Path

    t = Path(_textures_dir())
    if not t.is_dir():
        # ripiego procedurale: nessun file, solo pattern generati da MuJoCo
        return """
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.60 0.68"
      rgb2="0.10 0.12 0.16" width="512" height="3072"/>
    <texture type="2d" name="tex_floor" builtin="checker" mark="edge"
      rgb1="0.42 0.43 0.45" rgb2="0.36 0.37 0.39" markrgb="0.5 0.5 0.5"
      width="512" height="512"/>
    <material name="mat_floor" texture="tex_floor" texuniform="true"
      texrepeat="8 8" reflectance="0.12" specular="0.2" shininess="0.1"/>
    <material name="mat_wall" rgba="0.72 0.71 0.68 1" specular="0.1" shininess="0.05"/>
    <material name="mat_shelf_frame" rgba="0.30 0.33 0.38 1" specular="0.7" shininess="0.6"/>
    <material name="mat_shelf_deck" rgba="0.55 0.40 0.24 1" specular="0.2" shininess="0.1"/>
    <material name="mat_crate" rgba="0.80 0.45 0.15 1" specular="0.2" shininess="0.1"/>
    <material name="mat_pillar" rgba="0.62 0.62 0.60 1" specular="0.1" shininess="0.05"/>
"""

    return f"""
    <texture type="skybox" builtin="gradient" rgb1="0.55 0.60 0.68"
      rgb2="0.10 0.12 0.16" width="512" height="3072"/>

    <texture type="2d" name="tex_floor" file="{(t / "light-gray-floor-tile.png").as_posix()}"/>
    <material name="mat_floor" texture="tex_floor" texuniform="true"
      texrepeat="6 5" reflectance="0.15" specular="0.25" shininess="0.1"/>

    <texture type="2d" name="tex_wall" file="{(t / "light-gray-plaster.png").as_posix()}"/>
    <material name="mat_wall" texture="tex_wall" texuniform="true"
      texrepeat="4 2" reflectance="0.0" specular="0.1" shininess="0.05"/>

    <texture type="cube" name="tex_steel" file="{(t / "steel-brushed.png").as_posix()}"/>
    <material name="mat_shelf_frame" texture="tex_steel"
      reflectance="0.45" specular="0.7" shininess="0.6"/>

    <texture type="cube" name="tex_wood" file="{(t / "light-wood.png").as_posix()}"/>
    <material name="mat_shelf_deck" texture="tex_wood"
      reflectance="0.05" specular="0.2" shininess="0.1"/>

    <texture type="cube" name="tex_crate" file="{(t / "red-wood.png").as_posix()}"/>
    <material name="mat_crate" texture="tex_crate"
      reflectance="0.05" specular="0.25" shininess="0.15"/>

    <texture type="cube" name="tex_pillar" file="{(t / "white-bricks.png").as_posix()}"/>
    <material name="mat_pillar" texture="tex_pillar" texrepeat="2 2"
      reflectance="0.0" specular="0.1" shininess="0.05"/>
"""


def _wall(name: str, pos: tuple[float, float], half: tuple[float, float]) -> str:
    """Un muro come box. ``half`` sono semi-dimensioni, convenzione MuJoCo."""
    hz = WALL_HEIGHT / 2.0
    return (
        f'    <geom name="{name}" type="box" pos="{pos[0]} {pos[1]} {hz}" '
        f'size="{half[0]} {half[1]} {hz}" material="mat_wall"/>\n'
    )


def _walls_xml() -> str:
    """I quattro muri perimetrali, spessi verso l'esterno."""
    hx, hy = HALL_SIZE[0] / 2.0, HALL_SIZE[1] / 2.0
    t = WALL_THICK / 2.0
    return (
        _wall("muro_nord", (0.0, hy + t), (hx + WALL_THICK, t))
        + _wall("muro_sud", (0.0, -hy - t), (hx + WALL_THICK, t))
        + _wall("muro_est", (hx + t, 0.0), (t, hy + WALL_THICK))
        + _wall("muro_ovest", (-hx - t, 0.0), (t, hy + WALL_THICK))
    )


def _ceiling_xml() -> str:
    """Travi e lampade sotto il soffitto.

    **Solo visivo** (``class="scena"``), e alto abbastanza da non toccare
    niente: serve a chiudere il volume. Senza soffitto una scena con muri da
    6 m sembra un recinto a cielo aperto, e la prospettiva non si legge --
    l'occhio ha bisogno di un riferimento in alto per capire l'altezza.
    """
    hx, hy = HALL_SIZE[0] / 2.0, HALL_SIZE[1] / 2.0
    z = CEILING_HEIGHT
    out = ""
    # capriate trasversali
    for k in range(5):
        x = -hx + (k + 0.5) * (2 * hx / 5)
        out += (
            f'    <geom type="box" class="scena" pos="{x:.2f} 0 {z - 0.35:.2f}" '
            f'size="0.12 {hy} 0.28" material="mat_shelf_frame"/>\n'
        )
    # lampade industriali: due file di plafoniere accese
    for k in range(4):
        x = -hx + (k + 0.5) * (2 * hx / 4)
        for y in (-hy * 0.5, hy * 0.5):
            out += (
                f'    <geom type="box" class="scena" pos="{x:.2f} {y:.2f} {z - 0.7:.2f}" '
                f'size="0.55 0.14 0.05" rgba="1 0.97 0.85 1"/>\n'
            )
    return out


# Quanti ripiani ha uno scaffale, e quanto e' spesso un montante.
# 4 livelli su 4.5 m = un ripiano ogni 1.5 m, la spaziatura di una
# scaffalatura porta-pallet vera.
SHELF_LEVELS = 4
SHELF_POST = 0.08
SHELF_DECK = 0.06


def _shelves_xml() -> str:
    """Gli scaffali: struttura vera da guardare, blocco pieno da evitare.

    **La separazione visual/collision e' voluta ed e' quella di robosuite.**
    Il geom con ``contype=0 conaffinity=0`` (gruppo 1) si vede e non collide;
    quello del gruppo 0 collide e resta invisibile.

    Perche' la collisione resta un blocco pieno: alla navigazione interessa
    l'**ingombro in pianta**, e un modello con montanti e ripiani costerebbe
    decine di coppie di contatto in piu' senza cambiare **nessuna decisione**
    del planner -- che ragiona in 2D. E' la stessa scelta del nastro: si
    modella cio' che cambia un esito, non cio' che si vede.

    Un effetto collaterale interessante, e onesto: il LiDAR sta a 35 cm e
    vedrebbe **solo i montanti** se la collisione fosse la struttura vera.
    Un vero magazzino ha proprio questo problema -- un AMR che vede solo le
    gambe di uno scaffale ricostruisce una mappa piena di buchi. Qui non si
    presenta perche' il blocco di collisione e' pieno, e vale dirlo: e' una
    **semplificazione dichiarata**, non una proprieta' della realta'.
    """
    out = ""
    hz = SHELF_HEIGHT / 2.0
    for s in SHELVES:
        cx, cy = s.center
        hx, hy = s.size[0] / 2.0, s.size[1] / 2.0

        # Cio' che il robot urta: un blocco pieno, praticamente invisibile.
        #
        # **L'alpha e' 0.004 e non 0, e non e' pignoleria.** ``mj_ray``
        # ignora i geom completamente trasparenti -- serve anche al picking
        # col mouse, quindi salta cio' che non si vede. Con ``alpha=0`` il
        # LiDAR **attraversava gli scaffali**: misurato, a 240 gradi dava
        # 4.62 m (il muro sud) invece dei 2.00 m dello scaffale.
        #
        # Trappola perfetta: il geom collide nella fisica, e sparisce dal
        # sensore. Il robot ci sbatte contro senza averlo mai visto.
        # 0.004 e' invisibile all'occhio e visibile al raggio.
        out += (
            f'    <geom name="{s.name}" type="box" pos="{cx} {cy} {hz}" '
            f'size="{hx} {hy} {hz}" group="0" rgba="0.3 0.3 0.3 0.004"/>\n'
        )

        # --- cio' che si vede -------------------------------------------
        p = SHELF_POST / 2.0
        # montanti: agli angoli e a meta' delle campate lunghe, come una
        # scaffalatura vera (una campata da 3 m senza montante intermedio
        # non starebbe in piedi)
        for fx in (-1.0, 0.0, 1.0):
            for sy in (-1, 1):
                out += (
                    f'    <geom type="box" class="scena" '
                    f'pos="{cx + fx * (hx - p):.3f} {cy + sy * (hy - p):.3f} {hz}" '
                    f'size="{p} {p} {hz}" material="mat_shelf_frame"/>\n'
                )
        # correnti: le traverse orizzontali che portano i pallet
        for k in range(SHELF_LEVELS):
            z = 0.12 + k * (SHELF_HEIGHT - 0.3) / (SHELF_LEVELS - 1)
            for sy in (-1, 1):
                out += (
                    f'    <geom type="box" class="scena" '
                    f'pos="{cx} {cy + sy * (hy - p):.3f} {z:.3f}" '
                    f'size="{hx} {p} {SHELF_DECK / 2.0}" material="mat_shelf_frame"/>\n'
                )
            # il piano di appoggio, arretrato cosi' si vedono le traverse
            out += (
                f'    <geom type="box" class="scena" pos="{cx} {cy} {z - 0.02:.3f}" '
                f'size="{hx - 0.02} {hy - 0.06} 0.012" material="mat_shelf_deck"/>\n'
            )
            # i pallet: due o tre per campata, saltandone qualcuno perche'
            # un magazzino pieno uguale sembra finto
            for j, fx in enumerate((-0.58, 0.0, 0.58)):
                if (k + j) % 4 == 3:
                    continue
                px = cx + fx * hx
                out += (
                    f'    <geom type="box" class="scena" '
                    f'pos="{px:.3f} {cy} {z + 0.08:.3f}" '
                    f'size="{hx * 0.25:.3f} {hy * 0.72:.3f} 0.07" '
                    f'material="mat_shelf_deck"/>\n'
                    f'    <geom type="box" class="scena" '
                    f'pos="{px:.3f} {cy} {z + 0.38:.3f}" '
                    f'size="{hx * 0.22:.3f} {hy * 0.62:.3f} 0.23" '
                    f'material="mat_crate"/>\n'
                )
    return out


# --- l'ostacolo che il LiDAR 2D non puo' vedere -----------------------------
#
# **Il limite del 2D, reso visibile invece che scritto in una nota.**
#
# Un LiDAR 2D scansiona **un piano solo**: qui a 35 cm da terra. Tutto cio'
# che sta sopra o sotto quel piano, per il sensore, *non esiste*. E' il limite
# che la documentazione industriale dichiara senza mezzi termini -- "il LiDAR
# 2D non puo' percepire la struttura verticale" -- ed e' la ragione per cui
# gli AMR veri fondono LiDAR, camere e IMU invece di fidarsi di un sensore.
#
# Questa e' una forca di muletto sporgente a 1.2 m: il robot ci passa sotto
# col LiDAR che dice "libero", e il rilevatore di collisioni registra l'urto.
# Congelato in un test, come la linea retta che sbatte in D16: se un giorno
# il LiDAR lo vedesse, vorrebbe dire che qualcuno ha cambiato il montaggio.
# **Le quote sono vincolate da tre numeri e non si scelgono a occhio:**
#   - il LiDAR sta a 0.35 m  -> la sporgenza deve partire **sopra**
#   - il robot e' alto 0.335 -> ma deve partire **sotto** 0.335, o ci passa
#     sotto e non urta niente
#
# Le due condizioni si escludono: *nessun oggetto puo' essere invisibile al
# LiDAR e urtabile dalla base*. Con la base sola il limite del 2D non si
# manifesta -- ed e' un'informazione, non un fallimento dell'esperimento.
#
# Diventa vero appena il robot **porta qualcosa**: un AMR con un pallet
# caricato e' alto 1.2-1.5 m, e li' la sporgenza lo colpisce. Si modella il
# carico come un'estensione dell'ingombro verticale (``CARGO_HEIGHT``), che
# e' anche cio' che fa un AMR vero: la zona di sicurezza cambia a seconda che
# sia carico o scarico.
#
# Primo tentativo: sporgenza a 1.2 m. Il test "passava" -- ma il robot
# urtava il **palo di sostegno**, che scende a terra e che il LiDAR vede
# benissimo. Giusto risultato, motivo sbagliato: il flag di D2.
CARGO_HEIGHT = 1.10
OVERHANG = (1.8, -3.0, 0.95)  # x, y, quota inferiore
OVERHANG_SIZE = (0.9, 0.35, 0.22)


def _overhang_xml() -> str:
    x, y, z0 = OVERHANG
    hx, hy, hz = (s / 2.0 for s in OVERHANG_SIZE)
    zc = z0 + hz
    return (
        # il braccio sporgente: collide, ma sta sopra il piano del LiDAR
        f'    <geom name="sporgenza" type="box" pos="{x} {y} {zc}" '
        f'size="{hx} {hy} {hz}" material="mat_shelf_frame"/>\n'
        # il palo che lo regge, **fuori** dal corridoio: se stesse sotto la
        # sporgenza il LiDAR lo vedrebbe e l'ostacolo non sarebbe invisibile
        f'    <geom name="sporgenza_palo" type="box" '
        f'pos="{x + hx + 0.06} {y} {zc / 2}" size="0.06 {hy} {zc / 2}" '
        f'material="mat_shelf_frame"/>\n'
    )


def _narrow_gap_xml() -> str:
    """Il blocco che crea il varco stretto contro il muro nord.

    Lo si dimensiona **a partire dalla luce voluta**, non a occhio: il varco e'
    fra il bordo nord dello scaffale_a e il muro nord, e il blocco ne occupa
    tutto tranne ``AISLE_NARROW``.
    """
    shelf = SHELVES[0]
    shelf_north = shelf.center[1] + shelf.size[1] / 2.0
    wall_south = HALL_SIZE[1] / 2.0
    free = wall_south - shelf_north
    blocked = free - AISLE_NARROW
    if blocked <= 0:
        raise ValueError(
            f"il varco voluto ({AISLE_NARROW} m) non entra nello spazio "
            f"disponibile ({free:.3f} m): non c'e' niente da bloccare"
        )
    # Il blocco sta attaccato allo scaffale, la luce resta verso il muro.
    cy = shelf_north + blocked / 2.0
    hz = WALL_HEIGHT / 2.0
    return (
        f'    <geom name="pilastro" type="box" '
        f'pos="{_NARROW_GAP_CENTER[0]} {cy:.4f} {hz}" '
        f'size="0.5 {blocked / 2.0:.4f} {hz}" material="mat_pillar"/>\n'
    )


def _stations_xml() -> str:
    """Le stazioni come dischi piatti sul pavimento: si vedono, non collidono.

    ``contype=0 conaffinity=0`` e' la parte che conta: sono segnaposto visivi,
    e un segnaposto che ferma il robot sarebbe un bug travestito da scenografia.
    """
    out = ""
    for name, (x, y) in STATIONS.items():
        rgba = _CHARGER_RGBA if name == CHARGER else _STATION_RGBA
        out += (
            f'    <geom name="stazione_{name}" type="cylinder" class="scena" '
            f'pos="{x} {y} 0.004" size="{STATION_RADIUS} 0.004" rgba="{rgba}"/>\n'
            f'    <geom name="bordo_{name}" type="cylinder" class="scena" '
            f'pos="{x} {y} 0.006" size="{STATION_RADIUS} 0.002" rgba="1 1 1 0.25"/>\n'
        )
    return out


def _cargo_xml() -> str:
    """Il carico sul robot: un pallet con una cassa, alto ``CARGO_HEIGHT``.

    Sta **attaccato al robot** (dentro ``base_link``), quindi si muove con
    lui senza fisica aggiuntiva. Nasce trasparente e sotto il pavimento: lo
    si "accende" spostandolo, cosi' prelievo e consegna **si vedono** invece
    di essere solo due righe nel terminale.

    **E' anche ingombro vero**, e qui sta il punto: un AMR carico e' alto
    1.2-1.5 m, e cio' che passava sotto una sporgenza da scarico ora la
    colpisce. E' il motivo per cui un AMR vero ha zone di sicurezza diverse
    a seconda che porti qualcosa -- il limite del LiDAR 2D non e' teorico.
    """
    return (
        '      <body name="carico" pos="0 0 -5">\n'
        '        <geom name="carico_pallet" type="box" pos="0 0 0.05" '
        'size="0.26 0.26 0.05" material="mat_shelf_deck" mass="0"/>\n'
        f'        <geom name="carico_cassa" type="box" '
        f'pos="0 0 {0.1 + (CARGO_HEIGHT - 0.44) / 2:.3f}" '
        f'size="0.22 0.22 {(CARGO_HEIGHT - 0.44) / 2:.3f}" '
        'material="mat_crate" mass="0"/>\n'
        "      </body>\n"
    )


def _obstacles_xml(obstacles: dict[str, tuple[float, float]]) -> str:
    """Gli ostacoli mobili: cassoni che compaiono in mezzo ai corridoi.

    Stanno in un ``body`` con un giunto libero cosi' che si possano **spostare
    a runtime** scrivendo ``qpos``: un ostacolo che compare mentre il robot
    naviga e' il punto dell'esperimento, e un geom statico non si sposta.
    """
    out = ""
    hz = 0.3
    for name, (x, y) in obstacles.items():
        out += (
            f'    <body name="{name}" pos="{x} {y} {hz}">\n'
            f'      <freejoint name="{name}_free"/>\n'
            f'      <geom name="{name}_geom" type="box" size="0.3 0.3 {hz}" '
            f'material="mat_crate" mass="30"/>\n'
            f"    </body>\n"
        )
    return out


# Dove nascono gli ostacoli quando la scena si crea. Sono parcheggiati **fuori
# dal magazzino**: l'esperimento li porta dentro quando serve, cosi' il primo
# percorso e' pulito e il secondo no, e la differenza e' attribuibile.
OBSTACLE_PARK = {
    "cassone_1": (-4.5, 3.5),
    "cassone_2": (4.5, 3.5),
}


def warehouse_xml(*, with_obstacles: bool = True) -> str:
    """L'XML completo della scena.

    Si include ``base.xml`` del menagerie **senza toccarlo**: e' il modello
    ufficiale, e modificarlo significherebbe che le quote non sono piu' quelle
    del produttore. La scena ci si costruisce intorno.
    """
    base = MENAGERIE / "stanford_tidybot" / "base.xml"
    if not base.is_file():
        raise FileNotFoundError(
            f"modello TidyBot non trovato in {base}. "
            "Clonare mujoco_menagerie in third_party/ (vedi scripts/setup.sh)"
        )
    meshdir = base.parent / "assets"
    obstacles = _obstacles_xml(OBSTACLE_PARK) if with_obstacles else ""
    hx, hy = HALL_SIZE[0] / 2.0, HALL_SIZE[1] / 2.0

    # Il carico va **dentro** ``base_link``, e MuJoCo non permette di
    # aggiungere figli a un body che arriva da ``<include>`` (rifiuta con
    # "repeated name"). Si legge il file ufficiale, gli si inserisce il
    # carico prima della chiusura del body, e si passa il risultato come
    # stringa: **il file su disco non viene toccato**, che e' la regola --
    # i modelli del menagerie restano quelli del produttore.
    base_xml = base.read_text(encoding="utf-8")
    chiusura = base_xml.rindex("</body>")
    base_xml = base_xml[:chiusura] + _cargo_xml() + base_xml[chiusura:]
    base_xml = base_xml.replace("<mujoco model=\"base\">", "").replace("</mujoco>", "")
    base_xml = base_xml.replace(
        '<compiler angle="radian" meshdir="assets"/>', ""
    )

    return f"""<mujoco model="magazzino amr">
  <compiler angle="radian" meshdir="{meshdir.as_posix()}"/>
{base_xml}

  <statistic meansize="0.1" extent="12" center="0 0 0.5"/>

  <default>
    <!-- Cio' che si vede e non si tocca: la struttura degli scaffali, i
         segnaposto. Il gruppo 1 e' la convenzione di robosuite per il
         visual, cosi' le due scene si leggono allo stesso modo. -->
    <default class="scena">
      <geom contype="0" conaffinity="0" group="1"/>
    </default>
  </default>

  <visual>
    <!-- shadowsize alto e' cio' che cambia di piu' la resa: senza ombre
         ogni oggetto sembra incollato addosso al pavimento. 4096 costa
         poco sulla GPU integrata perche' la scena ha pochi corpi. -->
    <quality shadowsize="4096" offsamples="8"/>
    <headlight diffuse="0.35 0.35 0.38" ambient="0.25 0.25 0.28" specular="0.1 0.1 0.1"/>
    <map znear="0.05" zfar="60" shadowclip="1.5" shadowscale="0.8"/>
    <rgba haze="0.62 0.66 0.72 1"/>
    <global azimuth="130" elevation="-30" offwidth="1280" offheight="960"/>
  </visual>

  <asset>{_assets_xml()}  </asset>

  <worldbody>
    <!-- Tre luci: una generale dall'alto che fa le ombre, due laterali
         morbide che riempiono. Una sola luce lascia i fianchi neri, ed e'
         il motivo per cui una scena puo' sembrare "piatta". -->
    <light name="sole" pos="1.5 2.0 7.0" dir="-0.15 -0.25 -1" directional="true"
      diffuse="0.55 0.55 0.52" specular="0.25 0.25 0.25" castshadow="true"/>
    <light name="riempi_ovest" pos="{-hx * 0.6:.2f} 0 3.5" dir="0.4 0 -1"
      directional="true" diffuse="0.18 0.18 0.20" castshadow="false"/>
    <light name="riempi_est" pos="{hx * 0.6:.2f} 0 3.5" dir="-0.4 0 -1"
      directional="true" diffuse="0.18 0.18 0.20" castshadow="false"/>

    <geom name="pavimento" type="plane" pos="0 0 0" size="{hx} {hy} 0.05"
      material="mat_floor" condim="3"/>
{_walls_xml()}{_ceiling_xml()}{_shelves_xml()}{_narrow_gap_xml()}{_overhang_xml()}{_stations_xml()}{obstacles}
    <!-- I punti di vista, come le camere di TableArena.
         ``corsia`` sta ad altezza d'uomo dentro il magazzino: e' quella che
         fa capire l'altezza degli scaffali, perche' la prospettiva si legge
         solo da vicino. Dall'alto un capannone e una piantina si somigliano. -->
    <camera name="corsia" mode="fixed" pos="-6.5 -0.2 1.7"
      xyaxes="0 -1 0 0.22 0 0.97"/>
    <camera name="angolo" mode="fixed" pos="-8.5 -7.5 4.2"
      xyaxes="0.72 -0.70 0 0.26 0.27 0.93"/>
    <camera name="dallalto" mode="fixed" pos="0 -0.5 14" quat="1 0 0 0"/>
  </worldbody>
</mujoco>
"""


def build(*, with_obstacles: bool = True) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Costruisce e carica la scena. Ritorna modello e stato, gia' avanzati.

    ``mj_forward`` alla fine non e' cerimoniale: senza, ``geom_xpos`` contiene
    zeri e chi legge le quote dal modello (cioe' tutti, per convenzione) le
    leggerebbe sbagliate.
    """
    model = mujoco.MjModel.from_xml_string(warehouse_xml(with_obstacles=with_obstacles))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def set_cargo(model: mujoco.MjModel, data: mujoco.MjData, carico: bool) -> None:
    """Mette o toglie il carico dal robot.

    Il body ``carico`` esiste sempre; quando il robot e' scarico sta sotto il
    pavimento, dove non si vede e non tocca niente. Spostarlo costa una riga
    e **rende visibile una transizione di stato** che altrimenti sarebbe solo
    testo nel terminale.

    Non e' scenografia: da carico il robot e' alto ``CARGO_HEIGHT``, e la
    sporgenza a 0.95 m -- invisibile al LiDAR 2D -- lo colpisce.
    """
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "carico")
    if bid < 0:
        return
    model.body_pos[bid] = [0.0, 0.0, 0.34 if carico else -5.0]
    mujoco.mj_forward(model, data)


def has_cargo(model: mujoco.MjModel) -> bool:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "carico")
    return bid >= 0 and float(model.body_pos[bid][2]) > 0.0


def station_xy(name: str) -> np.ndarray:
    """Le coordinate di una stazione. Solleva se il nome non esiste."""
    if name not in STATIONS:
        raise KeyError(f"stazione '{name}' sconosciuta; disponibili: {sorted(STATIONS)}")
    return np.asarray(STATIONS[name], dtype=float)


def static_obstacles() -> list[tuple[float, float, float, float]]:
    """Gli ingombri in pianta di cio' che non si muove, come ``(x, y, hx, hy)``.

    Serve alla costruzione della mappa: la navigazione ragiona in 2D, e il
    terzo asse di un magazzino non dice niente finche' il robot non vola.
    Muri inclusi, perche' un planner che non conosce i muri manda il robot
    contro il perimetro.
    """
    hx, hy = HALL_SIZE[0] / 2.0, HALL_SIZE[1] / 2.0
    t = WALL_THICK / 2.0
    out: list[tuple[float, float, float, float]] = [
        (0.0, hy + t, hx + WALL_THICK, t),
        (0.0, -hy - t, hx + WALL_THICK, t),
        (hx + t, 0.0, t, hy + WALL_THICK),
        (-hx - t, 0.0, t, hy + WALL_THICK),
    ]
    for s in SHELVES:
        out.append((s.center[0], s.center[1], s.size[0] / 2.0, s.size[1] / 2.0))

    # il pilastro che crea il varco stretto, ricalcolato dalle stesse quote
    shelf = SHELVES[0]
    shelf_north = shelf.center[1] + shelf.size[1] / 2.0
    blocked = (HALL_SIZE[1] / 2.0) - shelf_north - AISLE_NARROW
    out.append((_NARROW_GAP_CENTER[0], shelf_north + blocked / 2.0, 0.5, blocked / 2.0))
    return out
