"""La costmap: la mappa trasformata in "quanto costa stare qui".

**L'idea, e perche' non basta una mappa di ostacoli.** Un planner che cerca su
una griglia di celle libere/occupate ragiona su un **punto**. Ma il robot e' un
corpo largo mezzo metro: un varco da 60 cm ha celle libere in mezzo, quindi il
planner ci passa, e il robot ci si incastra.

La soluzione standard (e' quella di Nav2, ``nav2_costmap_2d``) e' **gonfiare**
gli ostacoli del raggio del robot: ogni cella entro ``inscribed_radius`` da un
ostacolo diventa *letale*, perche' se il centro del robot ci sta, il corpo
tocca. Dopo il gonfiaggio il planner puo' tornare a ragionare sul punto, ed e'
corretto -- la larghezza del robot e' stata messa nella mappa.

Questo e' D16 per l'AMR. Li' mplib diceva ``"Success"`` su percorsi che
urtavano, perche' verificava il punto di arrivo e non il percorso. Qui il
rischio gemello e' verificare il percorso e non **il corpo che lo percorre**.

**I tre livelli di costo**, che sono quelli di Nav2:

``LETHAL``
    il centro del robot qui = collisione certa. Vietato.
``INSCRIBED``
    entro il raggio inscritto: il robot tocca di sicuro. Vietato.
``inflation`` (1..252)
    fuori dal corpo ma vicino: si puo' passare, **costa**. Serve a far
    preferire il centro del corridoio ai muri, perche' rasentare un muro con
    una posa stimata male e' come toccarlo.

Il gradiente non e' estetica: e' cio' che rende il percorso robusto
all'incertezza di posa, che sul robot vero e' l'errore dominante.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# I valori di costo, con gli stessi nomi e la stessa semantica di Nav2.
# 255 = sconosciuto, 254 = letale, 253 = dentro il raggio inscritto.
# Tenere la scala identica significa che un giorno i due codici si leggono
# con lo stesso occhio.
UNKNOWN = 255
LETHAL = 254
INSCRIBED = 253
FREE = 0

# Risoluzione della griglia in metri per cella. 5 cm e' il valore tipico di un
# AMR di magazzino: abbastanza fine da vedere un varco da 60 cm (12 celle),
# abbastanza grossa da non far esplodere A* su 10x8 m (200x160 = 32k celle).
DEFAULT_RESOLUTION = 0.05

# Quanto lontano dagli ostacoli si estende la fascia a costo decrescente.
#
# **Misurato, e il valore risolve un conflitto fra due livelli.** Con 0.35 il
# planner rasentava gli scaffali -- luce minima 0.375 m -- e il controller
# locale, che sorveglia un corridoio largo 0.44 m, si trovava un angolo di
# scaffale dentro la striscia di marcia e frenava su un percorso che aveva
# appena ricevuto. I due livelli si contraddicevano.
#
# ============  ==========  ====
# inflation     luce min    wp
# ============  ==========  ====
# 0.35 m        0.375 m     4
# **0.50 m**    **0.475 m** 3
# 0.70 m        0.475 m     3
# ============  ==========  ====
#
# Da 0.50 in su la luce si assesta a 0.475 m -- sopra il corridoio del
# controller -- e tutte le coppie di stazioni restano raggiungibili. Il
# percorso si allunga di 9 cm. Oltre non migliora, quindi non serve.
#
# La lezione e' quella di ``ARM_REACH`` in D17 al contrario: qui il valore
# **non** si deriva dal robot ma dal punto in cui due livelli smettono di
# contraddirsi, e quel punto si trova solo misurando.
DEFAULT_INFLATION = 0.50


@dataclass(frozen=True)
class Footprint:
    """L'ingombro del robot in pianta, come due raggi.

    ``inscribed`` e' il raggio del cerchio piu' grande **contenuto** nel corpo:
    dentro quella distanza da un ostacolo, il robot tocca di sicuro.
    ``circumscribed`` e' il cerchio piu' piccolo che **contiene** il corpo:
    oltre quella distanza il robot e' sicuramente libero, comunque orientato.
    Fra i due dipende da come e' girato.

    Questa e' l'astrazione di Nav2, e vale per un corpo qualsiasi. Per una base
    quadrata di lato L: inscritto L/2, circoscritto L*sqrt(2)/2.

    **Il rischio e' quello di ``ARM_REACH`` in D17**: una costante che sembra
    ovvia, ricavata dal posto sbagliato, che fa approvare cio' che non va.
    Qui i raggi si **leggono dal modello** con ``from_model``, non si scrivono
    a occhio guardando una foto del robot.
    """

    inscribed: float
    circumscribed: float

    def __post_init__(self) -> None:
        if self.inscribed <= 0 or self.circumscribed <= 0:
            raise ValueError("i raggi del footprint devono essere positivi")
        if self.circumscribed < self.inscribed:
            raise ValueError(
                f"il raggio circoscritto ({self.circumscribed:.3f}) non puo' essere "
                f"minore dell'inscritto ({self.inscribed:.3f}): "
                "e' geometricamente impossibile, probabilmente sono scambiati"
            )

    @property
    def min_gap(self) -> float:
        """Il varco minimo in cui il robot passa **comunque orientato**.

        E' il diametro circoscritto. Un varco fra questo e il diametro
        inscritto si attraversa solo con l'orientamento giusto -- e una base
        olonoma puo' scegliere l'orientamento, quindi non e' teorico.
        """
        return 2.0 * self.circumscribed


def footprint_from_model(
    model,
    data,
    body: str = "base_link",
    *,
    escludi: tuple[str, ...] = ("carico",),
) -> Footprint:
    """Legge l'ingombro **dal modello**, non da un numero scritto a mano.

    Scorre i geom di collisione del robot e ne prende l'estensione in pianta,
    **dai vertici delle mesh**. E' la convenzione del progetto applicata dove
    e' costata di piu': le quote si leggono, non si derivano.

    **Due campi comodi che qui mentono**, entrambi provati e scartati:

    ``geom_rbound``
        il raggio della sfera che contiene il geom, usato da MuJoCo per
        scartare in fretta le collisioni. E' in **3D** e volutamente
        generoso: sulla base del TidyBot vale 0.402 contro i 0.340 veri, e
        includerebbe l'altezza nel calcolo di un ingombro **in pianta**.
    ``geom_aabb``
        dava ``0.23 x 0.567`` su un robot che e' quadrato ``0.481 x 0.481``:
        per una mesh non e' l'estensione sugli assi che ci si aspetta.

    Entrambi rispondono a una domanda diversa da quella posta, e lo fanno con
    un numero plausibile. E' la forma di ``ARM_REACH`` ricavato dal tavolo in
    D17: la costante che sembra ovvia, presa dal posto sbagliato.
    """
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise ValueError(f"body '{body}' non trovato nel modello")

    # I body da escludere, con tutto cio' che ci pende sotto.
    #
    # **Il carico non e' il robot.** Il footprint descrive l'ingombro della
    # macchina, e deve restare lo stesso che sia carica o scarica: e' con
    # quello che la costmap gonfia gli ostacoli, e un gonfiaggio che cambia
    # a meta' missione renderebbe il percorso valido all'andata e non al
    # ritorno. Misurato prima di escluderlo: col pallet dentro, l'inscritto
    # saliva da 0.2407 a 0.2600.
    #
    # L'ingombro **verticale** del carico conta eccome (vedi la sporgenza
    # invisibile al LiDAR in ``warehouse.py``), ma quello e' un altro
    # problema e un'altra misura.
    esclusi: set[int] = set()
    for nome in escludi:
        eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nome)
        if eid < 0:
            continue
        for b in range(model.nbody):
            p = b
            while p > 0:
                if p == eid:
                    esclusi.add(b)
                    break
                p = model.body_parentid[p]

    bodies = {bid}
    for b in range(model.nbody):
        if b in esclusi:
            continue
        p = b
        while p > 0:
            if p == bid:
                bodies.add(b)
                break
            p = model.body_parentid[p]
    bodies -= esclusi

    lo = np.array([np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf])
    origin = data.xpos[bid][:2]
    found = False
    for g in range(model.ngeom):
        if model.geom_bodyid[g] not in bodies:
            continue
        # solo i geom che collidono davvero: un geom puramente visivo
        # (contype=0 e conaffinity=0) non e' ingombro
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        gpos = data.geom_xpos[g][:2] - origin
        mid = int(model.geom_dataid[g])
        if mid >= 0 and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            # i vertici, ruotati nel frame mondo: l'unica fonte che dice
            # davvero quanto e' largo il robot
            v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            verts = model.mesh_vert[v0 : v0 + nv].reshape(-1, 3)
            rot = data.geom_xmat[g].reshape(3, 3)
            xy = (verts @ rot.T)[:, :2] + gpos
            lo = np.minimum(lo, xy.min(axis=0))
            hi = np.maximum(hi, xy.max(axis=0))
        else:
            # primitive (box, cylinder, sphere): geom_size e' esatto
            s = model.geom_size[g][:2]
            lo = np.minimum(lo, gpos - s)
            hi = np.maximum(hi, gpos + s)
        found = True

    if not found:
        raise ValueError(
            f"nessun geom di collisione sotto '{body}': "
            "l'ingombro sarebbe zero, e un robot senza ingombro passa ovunque"
        )

    half = np.maximum(np.abs(lo), np.abs(hi))
    return Footprint(
        inscribed=float(half.min()),
        circumscribed=float(np.hypot(half[0], half[1])),
    )


class Costmap:
    """Griglia di costi in coordinate mondo.

    Il frame: la cella ``(row, col)`` copre il quadrato di mondo centrato in
    ``world_from_cell(row, col)``. L'origine della griglia sta nell'angolo
    ``(x_min, y_min)``, e **row indicizza y, col indicizza x** -- che e' il
    contrario dell'intuizione cartesiana ed e' la prima cosa che si sbaglia.
    Le due funzioni di conversione esistono perche' nessuno debba farlo a mano.
    """

    def __init__(
        self,
        bounds: tuple[float, float, float, float],
        resolution: float = DEFAULT_RESOLUTION,
    ) -> None:
        x_min, y_min, x_max, y_max = bounds
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(f"bounds degeneri: {bounds}")
        if resolution <= 0:
            raise ValueError(f"risoluzione deve essere positiva: {resolution}")
        self.resolution = float(resolution)
        self.x_min, self.y_min = float(x_min), float(y_min)
        self.n_cols = int(np.ceil((x_max - x_min) / resolution))
        self.n_rows = int(np.ceil((y_max - y_min) / resolution))
        self.grid = np.zeros((self.n_rows, self.n_cols), dtype=np.uint8)

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.shape

    # --- conversioni di frame ----------------------------------------------

    def cell_from_world(self, x: float, y: float) -> tuple[int, int]:
        """Da coordinate mondo a ``(row, col)``. Non controlla i limiti."""
        col = int((x - self.x_min) / self.resolution)
        row = int((y - self.y_min) / self.resolution)
        return row, col

    def world_from_cell(self, row: int, col: int) -> tuple[float, float]:
        """Dal centro di una cella alle coordinate mondo."""
        return (
            self.x_min + (col + 0.5) * self.resolution,
            self.y_min + (row + 0.5) * self.resolution,
        )

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.n_rows and 0 <= col < self.n_cols

    # --- costruzione --------------------------------------------------------

    def add_boxes(self, boxes: list[tuple[float, float, float, float]]) -> None:
        """Marca come letali le celle coperte da rettangoli ``(x, y, hx, hy)``.

        Il tocco che conta: si arrotonda **verso l'esterno**. Una cella
        sfiorata da un ostacolo e' occupata, perche' l'alternativa -- una cella
        mezza libera considerata libera -- e' un ostacolo che si assottiglia di
        mezza cella per lato a ogni conversione.
        """
        for cx, cy, hx, hy in boxes:
            r0 = max(0, int(np.floor((cy - hy - self.y_min) / self.resolution)))
            r1 = min(self.n_rows - 1, int(np.ceil((cy + hy - self.y_min) / self.resolution)))
            c0 = max(0, int(np.floor((cx - hx - self.x_min) / self.resolution)))
            c1 = min(self.n_cols - 1, int(np.ceil((cx + hx - self.x_min) / self.resolution)))
            if r1 >= r0 and c1 >= c0:
                self.grid[r0 : r1 + 1, c0 : c1 + 1] = LETHAL

    def add_points(self, points: np.ndarray) -> None:
        """Marca come letali le celle che contengono punti (ritorni LiDAR).

        E' la costmap **locale**: cio' che il sensore vede adesso, senza sapere
        cosa sia. Un ostacolo comparso dopo la mappa arriva da qui.
        """
        if len(points) == 0:
            return
        cols = ((points[:, 0] - self.x_min) / self.resolution).astype(int)
        rows = ((points[:, 1] - self.y_min) / self.resolution).astype(int)
        m = (rows >= 0) & (rows < self.n_rows) & (cols >= 0) & (cols < self.n_cols)
        self.grid[rows[m], cols[m]] = LETHAL

    def inflate(self, footprint: Footprint, inflation_radius: float = 0.35) -> None:
        """Gonfia gli ostacoli. **E' il passaggio che fa passare il corpo.**

        Tre fasce, come in Nav2:

        - entro ``inscribed``: ``LETHAL``, il robot tocca comunque sia girato;
        - fino a ``circumscribed``: ``INSCRIBED``, tocca a seconda di come e'
          girato -- trattato come vietato, perche' il planner non ragiona
          sull'orientamento (sceglierlo sarebbe il lavoro del controller);
        - fino a ``inflation_radius``: costo decrescente, si puo' passare ma si
          preferisce stare larghi.

        Senza questa chiamata il planner trova percorsi che il robot non puo'
        fare. **Misurato, non supposto:** su un varco da 0.50 m -- in cui un
        robot da 0.68 non entra nemmeno fisicamente -- il planner senza
        gonfiaggio ci manda il robot dentro con un percorso da 2.20 m invece
        dei 8.85 del giro. Quattro volte piu' corto, e impossibile. E' la
        forma di mplib in D16: ``"Success"`` su un percorso che urta.

        **Il gonfiaggio sbaglia dalla parte conservativa, e di quanto dipende
        dalla risoluzione.** Soglia di accettazione di un varco, misurata per
        ricerca binaria contro i 0.681 m di teoria (2 x circoscritto):

        ==========  ============  ==============
        celle       soglia vera   errore
        ==========  ============  ==============
        10 cm       0.900 m       +22 cm
        5 cm        0.750 m       +7 cm
        2.5 cm      0.725 m       +4 cm
        ==========  ============  ==============

        Due arrotondamenti che si sommano invece di compensarsi: ``add_boxes``
        allarga gli ostacoli di mezza cella per lato (voluto), e la distanza
        si misura fra centri di cella. Rifiutare un varco percorribile e' il
        lato giusto dell'errore -- l'altro sarebbe un robot incastrato -- ma
        e' un numero da **dichiarare**, o chi legge "passa in 68 cm" e vede
        rifiutare 85 cm pensa a un bug.
        """
        from scipy.ndimage import distance_transform_edt

        occupied = self.grid >= INSCRIBED
        if not occupied.any():
            return
        # distanza di ogni cella libera dall'ostacolo piu' vicino, in metri
        dist = distance_transform_edt(~occupied, sampling=self.resolution)

        self.grid[dist <= footprint.inscribed] = LETHAL
        band = (dist > footprint.inscribed) & (dist <= footprint.circumscribed)
        self.grid[band] = INSCRIBED

        if inflation_radius > footprint.circumscribed:
            soft = (dist > footprint.circumscribed) & (dist <= inflation_radius)
            span = inflation_radius - footprint.circumscribed
            decay = 1.0 - (dist[soft] - footprint.circumscribed) / span
            # 1..252: mai 0 (sarebbe "libero come il centro del corridoio") e
            # mai 253 (sarebbe "vietato"). La fascia morbida deve restare
            # percorribile, altrimenti un corridoio stretto si chiude del tutto.
            self.grid[soft] = np.clip((decay * 252).astype(np.uint8), 1, 252)

    def is_free(self, row: int, col: int) -> bool:
        """La cella e' percorribile dal **centro** del robot?"""
        return self.in_bounds(row, col) and self.grid[row, col] < INSCRIBED

    def is_free_world(self, x: float, y: float) -> bool:
        return self.is_free(*self.cell_from_world(x, y))

    def cost(self, row: int, col: int) -> int:
        return int(self.grid[row, col])

    def copy(self) -> Costmap:
        """Una copia indipendente. La costmap locale parte da quella globale."""
        other = Costmap.__new__(Costmap)
        other.resolution = self.resolution
        other.x_min, other.y_min = self.x_min, self.y_min
        other.n_rows, other.n_cols = self.n_rows, self.n_cols
        other.grid = self.grid.copy()
        return other


def build_static(
    bounds: tuple[float, float, float, float],
    boxes: list[tuple[float, float, float, float]],
    footprint: Footprint,
    *,
    resolution: float = DEFAULT_RESOLUTION,
    inflation_radius: float = DEFAULT_INFLATION,
) -> Costmap:
    """La costmap globale: cio' che non si muove, gia' gonfiato."""
    cm = Costmap(bounds, resolution)
    cm.add_boxes(boxes)
    cm.inflate(footprint, inflation_radius)
    return cm
