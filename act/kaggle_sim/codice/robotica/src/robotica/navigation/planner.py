"""Planner globale: A* sulla costmap. Da dove sono a dove voglio andare.

**Cos'e' A*, in una riga:** una ricerca che espande sempre la cella piu'
promettente, dove "promettente" = costo gia' speso + stima ottimistica di
quello che manca. Se la stima non sopravvaluta mai (e' *ammissibile*), il primo
percorso trovato e' ottimo.

**Perche' A* e non la ricerca in ampiezza di ``planning/planner.py``.** Sono
due problemi diversi e vale la pena vedere la differenza, perche' e' la stessa
ragione per cui in D14 la decomposizione e' diventata necessaria:

- ``planning/`` cerca su **stati simbolici**: "cubo A su cubo B". Non c'e'
  distanza fra due stati, quindi non c'e' euristica possibile, e con 6 cubi la
  ricerca si arrende dopo 200.000 stati.
- qui si cerca su una **griglia**: c'e' una distanza, quindi c'e' un'euristica,
  e A* la usa per non guardare nemmeno le celle dalla parte sbagliata.

**Il percorso trovato e' gia' sicuro per il corpo del robot** -- ma solo
perche' la costmap e' stata gonfiata. Se qualcuno passasse una costmap non
gonfiata, questo codice troverebbe allegramente percorsi che il robot non puo'
fare, e direbbe "trovato". E' esattamente la forma di mplib in D16: il
``"Success"`` che non prova niente. Per questo ``plan`` controlla e si rifiuta.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

from .costmap import INSCRIBED, Costmap

# I costi di movimento fra celle adiacenti. Otto direzioni: le diagonali
# costano sqrt(2), altrimenti il percorso preferirebbe le scale a gradini a
# parita' di numero di passi e uscirebbe piu' lungo di quanto dice.
_STEPS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (1, 1, math.sqrt(2)),
)

# Quanto pesa il costo di prossimita' nella scelta. Con 0 il percorso rasenta
# i muri (e' piu' corto); troppo alto e il robot fa giri assurdi per stare al
# centro. 0.01 per unita' di costo significa che attraversare una cella a
# costo 252 costa come ~2.5 celle in piu' di strada.
PROXIMITY_WEIGHT = 0.01


class NoPathFound(Exception):
    """Nessun percorso esiste. **Non e' la stessa cosa di "mi sono arreso".**

    D14 ha insegnato che un sistema che si arrende deve dirlo, invece di
    dedurre che il problema non ha soluzione. A* su una griglia finita esplora
    tutto il raggiungibile, quindi qui l'esaurimento **e'** una prova di
    impossibilita' -- e il messaggio lo dice, perche' chi legge non deve
    doverlo ricostruire.
    """


def _heuristic(r: int, c: int, gr: int, gc: int) -> float:
    """Distanza ottagonale: il minimo numero di passi a 8 direzioni.

    Ammissibile e piu' informata di quella euclidea su una griglia a 8 vicini:
    non sovrastima mai (quindi A* resta ottimo) ma e' piu' vicina al vero,
    quindi espande meno celle. La differenza si sente su 32.000 celle.
    """
    dr, dc = abs(r - gr), abs(c - gc)
    return float(max(dr, dc) + (math.sqrt(2) - 1.0) * min(dr, dc))


def _nearest_free(cm: Costmap, row: int, col: int, max_radius_cells: int = 20) -> tuple[int, int]:
    """La cella libera piu' vicina a una occupata.

    Serve perche' un bersaglio dichiarato a mano (una stazione) puo' cadere
    dentro la fascia gonfiata di uno scaffale: la stazione sta *davanti* allo
    scaffale, e il gonfiaggio la puo' coprire. Spostare il bersaglio di poche
    celle e' cio' che fa Nav2 (``GoalUpdater``), ed e' meglio che rifiutare --
    ma **si dichiara**, perche' un bersaglio spostato in silenzio e' un altro
    flag che dice "va tutto bene".
    """
    if cm.is_free(row, col):
        return row, col
    for rad in range(1, max_radius_cells + 1):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if max(abs(dr), abs(dc)) != rad:
                    continue
                if cm.is_free(row + dr, col + dc):
                    return row + dr, col + dc
    raise NoPathFound(
        f"nessuna cella libera entro {max_radius_cells} celle da (row={row}, col={col}): "
        "il punto e' sepolto dentro un ostacolo gonfiato"
    )


def astar(
    cm: Costmap,
    start_world: tuple[float, float],
    goal_world: tuple[float, float],
    *,
    relax_goal: bool = True,
) -> tuple[list[tuple[float, float]], dict]:
    """Il percorso da start a goal, in coordinate mondo.

    Ritorna ``(percorso, info)``. ``info`` non e' decorazione: contiene quante
    celle sono state espanse e se il bersaglio e' stato spostato. Un percorso
    senza il conto di cio' che e' costato non permette di rispondere a *"la
    ricerca ha funzionato o si e' arresa?"* -- la domanda di D14.
    """
    sr, sc = cm.cell_from_world(*start_world)
    gr, gc = cm.cell_from_world(*goal_world)

    if not cm.in_bounds(sr, sc):
        raise NoPathFound(f"la partenza {start_world} e' fuori dalla mappa")
    if not cm.in_bounds(gr, gc):
        raise NoPathFound(f"il bersaglio {goal_world} e' fuori dalla mappa")

    info: dict = {"goal_moved_m": 0.0, "start_moved_m": 0.0}

    if not cm.is_free(sr, sc):
        # la partenza occupata capita davvero: il robot fermo davanti a uno
        # scaffale sta dentro la fascia gonfiata. Non e' un errore, e' la
        # geometria -- ma va detto.
        nr, nc = _nearest_free(cm, sr, sc)
        info["start_moved_m"] = math.hypot(nr - sr, nc - sc) * cm.resolution
        sr, sc = nr, nc

    if not cm.is_free(gr, gc):
        if not relax_goal:
            raise NoPathFound(
                f"il bersaglio {goal_world} cade in una cella vietata "
                f"(costo {cm.cost(gr, gc)}): il robot non ci starebbe. "
                "Con relax_goal=True si cercherebbe la cella libera piu' vicina."
            )
        nr, nc = _nearest_free(cm, gr, gc)
        info["goal_moved_m"] = math.hypot(nr - gr, nc - gc) * cm.resolution
        gr, gc = nr, nc

    open_heap: list[tuple[float, int, int]] = [(_heuristic(sr, sc, gr, gc), sr, sc)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = {(sr, sc): 0.0}
    closed: set[tuple[int, int]] = set()
    expanded = 0

    while open_heap:
        _, r, c = heapq.heappop(open_heap)
        if (r, c) in closed:
            continue
        closed.add((r, c))
        expanded += 1

        if (r, c) == (gr, gc):
            cells = [(r, c)]
            while (r, c) in came:
                r, c = came[(r, c)]
                cells.append((r, c))
            cells.reverse()
            info["expanded"] = expanded
            info["cells"] = len(cells)
            info["length_m"] = _path_length([cm.world_from_cell(*x) for x in cells])
            return [cm.world_from_cell(*x) for x in cells], info

        for dr, dc, step in _STEPS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in closed or not cm.in_bounds(nr, nc):
                continue
            cost = cm.cost(nr, nc)
            if cost >= INSCRIBED:
                continue
            # la diagonale che taglia l'angolo di un ostacolo: entrambe le
            # celle ortogonali devono essere libere, o il robot passa
            # attraverso uno spigolo. E' un classico, e in mezzo metro di
            # robot fa la differenza fra passare e incastrarsi.
            if dr and dc and not (cm.is_free(r + dr, c) and cm.is_free(r, c + dc)):
                continue
            tentative = g_score[(r, c)] + step + PROXIMITY_WEIGHT * cost
            if tentative < g_score.get((nr, nc), math.inf):
                g_score[(nr, nc)] = tentative
                came[(nr, nc)] = (r, c)
                heapq.heappush(
                    open_heap, (tentative + _heuristic(nr, nc, gr, gc), nr, nc)
                )

    raise NoPathFound(
        f"nessun percorso da {start_world} a {goal_world}: "
        f"esplorate tutte le {expanded} celle raggiungibili senza arrivarci. "
        "La ricerca NON si e' arresa -- il bersaglio e' separato dalla partenza "
        "(un varco troppo stretto per il robot conta come muro)."
    )


def _path_length(path: list[tuple[float, float]]) -> float:
    if len(path) < 2:
        return 0.0
    p = np.asarray(path)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def simplify(
    path: list[tuple[float, float]], cm: Costmap, tolerance: float = 0.05
) -> list[tuple[float, float]]:
    """Toglie i punti intermedi che stanno su una retta libera.

    Un percorso A* su griglia ha un punto ogni 5 cm, quasi tutti allineati.
    Ridurli non cambia il percorso ma rende il controller molto piu' leggibile
    quando si stampa, e piu' economico da inseguire.

    **Si controlla che la scorciatoia sia libera**, cella per cella. Saltare
    quel controllo darebbe una linea retta che attraversa uno scaffale -- che
    e' letteralmente il fallimento congelato di D16.
    """
    if len(path) < 3:
        return list(path)
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if _segment_free(path[i], path[j], cm, tolerance):
                break
            j -= 1
        out.append(path[j])
        i = j
    return out


def _segment_free(a: tuple[float, float], b: tuple[float, float], cm: Costmap, step: float) -> bool:
    """Il segmento a-b attraversa solo celle percorribili?"""
    d = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(2, int(d / step) + 1)
    for t in np.linspace(0.0, 1.0, n):
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1])
        if not cm.is_free_world(x, y):
            return False
    return True
