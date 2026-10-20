"""Dal piano ai movimenti: il ponte fra logica e robot.

Il pianificatore ragiona su fatti; il robot si muove in metri. Qui si traduce
nei due sensi:

- **lettura**: la scena diventa fatti (``planning/state.py`` fa il lavoro)
- **esecuzione**: ``stack(a, b)`` diventa una posa sopra ``b``

La parte insidiosa e' la seconda. ``stack(a, b)`` dice *dove* in termini
relativi — sopra ``b`` — e la quota va **letta dalla posizione di b**, non
calcolata dall'altezza del tavolo piu' un numero di cubi. E' la stessa regola
che in Fase 0 e' costata ogni presa: 25 mm dedotti male su ``table_offset``
bastavano a far fallire tutto. Nel dubbio, si misura.

L'esecuzione passa dai metodi del robot, che gia' verificano ogni movimento con
``safety/``. Il pianificatore non scavalca la sicurezza: le si appoggia sopra.
Un piano logicamente corretto puo' contenere un movimento fuori dal workspace,
e in quel caso il passo fallisce e il recupero fa il suo mestiere.
"""

from __future__ import annotations

import numpy as np

from robotica.planning.actions import Action
from robotica.planning.state import CUBE_SIZE, State, WorldModel

# Quanto si posa sopra la faccia superiore del cubo sotto.
#
# Non zero: si lascia un margine e si fa cadere il cubo di quel poco. Posare a
# contatto esatto significa spingere un cubo contro l'altro finche' il
# controller non si accorge del contatto, e in quel momento la pila si sposta.
STACK_CLEARANCE = 0.004


class RobotBridge:
    """Esegue le azioni del piano sul robot, e legge lo stato dalla scena.

    Tiene il minimo stato necessario: cosa la pinza ha in mano. Non si deduce
    dalle coordinate in modo affidabile — un cubo appena sollevato e un cubo in
    mano si somigliano troppo — e sbagliarlo falserebbe tutti i fatti.
    """

    def __init__(self, env, robot, zone_names: dict[str, np.ndarray] | None = None) -> None:
        self.env = env
        self.robot = robot
        self.held: str | None = None

        from robotica.sim.env import GREEN_ZONE_HALF, SORT_ZONE_HALF

        # Le zone e i colori si chiedono all'ambiente, se sa rispondere. Una
        # scena di smistamento ha piu' zone e cubi colorati; quella base ha una
        # zona verde e nessun colore. Il ponte non deve sapere quale delle due
        # ha davanti: chiede, e si adatta.
        zones = zone_names
        zone_half = GREEN_ZONE_HALF
        colors: dict[str, str] = {}

        if zones is None:
            if hasattr(env, "zone_centers"):
                zones = env.zone_centers()
                zone_half = SORT_ZONE_HALF
            else:
                zones = {"green": env.green_zone_center()}

        if hasattr(env, "color_of"):
            colors = {n: env.color_of(n) for n in env.object_names()}

        self.zone_half = zone_half
        self.model = WorldModel(
            objects=list(env.object_names()),
            zones=zones,
            zone_half=zone_half,
            colors=colors,
        )

    # --- lettura -------------------------------------------------------------

    def read_state(self) -> State:
        """Lo stato attuale, letto dalla scena.

        Si rilegge tutto ogni volta invece di aggiornare un modello interno:
        e' l'unico modo per accorgersi che un cubo e' caduto o e' stato spostato
        da fuori. Costa una lettura, e vale la pena.
        """
        positions = {name: self.env.object_pos(name) for name in self.env.object_names()}
        return self.model.read(positions, held=self.held)

    def objects(self) -> list[str]:
        return list(self.env.object_names())

    def zones(self) -> list[str]:
        return list(self.model.zones)

    def colors(self) -> dict[str, str]:
        """Colore di ogni oggetto. Vuoto se la scena non ha colori."""
        return dict(self.model.colors)

    # --- esecuzione ----------------------------------------------------------

    def execute(self, action: Action) -> tuple[bool, str]:
        """Esegue un'azione. Restituisce (riuscito, motivo se no).

        Il motivo del fallimento arriva da ``safety/`` o dalla verifica della
        presa, quindi e' gia' scritto per essere letto: contiene regola, valore
        e limite.
        """
        handler = {
            "pickup": self._pick,
            "unstack": self._pick,
            "putdown": self._putdown,
            "stack": self._stack,
            "put_in_zone": self._put_in_zone,
        }.get(action.name)

        if handler is None:
            return False, f"azione '{action.name}' non eseguibile su questo robot"

        try:
            return handler(action)
        except Exception as exc:  # un passo fallito non deve fermare il piano
            return False, f"eccezione: {exc}"

    def _pick(self, action: Action) -> tuple[bool, str]:
        """``pickup(x)`` e ``unstack(x, y)``: entrambi afferrano ``x``.

        Sono la stessa mossa per il robot e due azioni diverse per il
        pianificatore, perche' ``unstack`` libera anche il cubo sotto. La
        distinzione serve alla logica, non alla meccanica.
        """
        name = action.args[0]
        result = self.robot.pick(name)
        if not result:
            return False, str(result.reason)

        self.held = name
        return True, ""

    def _putdown(self, action: Action) -> tuple[bool, str]:
        """Posa sul tavolo, in un punto libero.

        Il pianificatore dice *posa*, non *dove*: qualsiasi punto libero va
        bene. Si sceglie qui, evitando gli altri oggetti.
        """
        name = action.args[0]
        target = self._free_spot(exclude=name)
        return self._place_at(name, target)

    def _stack(self, action: Action) -> tuple[bool, str]:
        """``stack(x, y)``: posa ``x`` sopra ``y``.

        **La quota si legge da ``y``**, non si calcola dall'altezza del tavolo:
        se ``y`` e' gia' sopra un terzo cubo, la sua posizione lo dice e la pila
        cresce correttamente.
        """
        x, y = action.args
        base = self.env.object_pos(y)
        target = np.array([base[0], base[1], base[2] + CUBE_SIZE + STACK_CLEARANCE])
        return self._place_at(x, target)

    def _put_in_zone(self, action: Action) -> tuple[bool, str]:
        """Posa ``x`` dentro una zona, in un punto **libero** della zona.

        Posare sempre al centro era il limite emerso nella prova del
        2026-09-18: col secondo cubo si costruiva una pila per caso, e il
        pianificatore la registrava come ``on(x, y)`` — corretto, ma non era
        smistamento. Per un compito di classificazione i cubi devono stare
        **accanto**, non sopra.
        """
        name, zone = action.args
        target = self._free_spot_in_zone(zone, exclude=name)
        return self._place_at(name, target)

    def _free_spot_in_zone(self, zone: str, exclude: str) -> np.ndarray:
        """Un punto libero dentro la zona, o il centro se la zona e' piena.

        Si scorre una griglia 3x3 dentro il quadrato della zona e si prende il
        primo punto abbastanza lontano dai cubi gia' posati. Il margine e' un
        lato di cubo e mezzo: abbastanza perche' due cubi non si tocchino, poco
        abbastanza da farcene stare quattro in una zona da 140 mm di lato.

        Se non c'e' posto si torna al centro e si lascia decidere alla fisica.
        E' una scelta: il caso "zona piena" e' un problema di capienza, non di
        esecuzione, e fingere che il punto esista sarebbe peggio che ammassare.
        """
        from robotica.sim.env import CUBE_HALF, table_surface_z

        center = self.model.zones[zone]
        z = table_surface_z()
        margin = CUBE_HALF * 3  # 60 mm fra i centri: i cubi non si toccano

        occupati = [
            self.env.object_pos(n)[:2]
            for n in self.env.object_names()
            if n not in (exclude, self.held)
        ]

        # Dal centro verso l'esterno: si riempie la zona ordinatamente invece
        # di partire da un angolo.
        passo = self.zone_half - CUBE_HALF
        offsets = [(0.0, 0.0)]
        offsets += [
            (dx * passo, dy * passo)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if (dx, dy) != (0, 0)
        ]

        for dx, dy in offsets:
            p = np.array([center[0] + dx, center[1] + dy])
            if all(float(np.linalg.norm(p - o)) > margin for o in occupati):
                return np.array([p[0], p[1], z])

        return np.array([center[0], center[1], z])

    def _place_at(self, name: str, target: np.ndarray) -> tuple[bool, str]:
        result = self.robot.place(target, name)
        if not result:
            return False, str(result.reason)

        self.held = None
        self._settle()
        return True, ""

    # --- utilita' ------------------------------------------------------------

    def _free_spot(self, exclude: str, margin: float = 0.10) -> np.ndarray:
        """Un punto sul tavolo lontano dagli altri oggetti.

        Si prova una griglia dentro il workspace e si prende il primo punto che
        dista abbastanza da tutto. Grezzo ma sufficiente: serve solo a
        parcheggiare un cubo mentre se ne sposta un altro.
        """
        from robotica.sim.env import table_surface_z

        occupati = [
            self.env.object_pos(n)[:2]
            for n in self.env.object_names()
            if n != exclude and n != self.held
        ]

        ws = self.robot.safety.workspace
        z = table_surface_z()

        for x in np.linspace(ws.x[0] + margin, ws.x[1] - margin, 5):
            for y in np.linspace(ws.y[0] + margin, ws.y[1] - margin, 5):
                p = np.array([x, y])
                if all(float(np.linalg.norm(p - o)) > margin for o in occupati):
                    return np.array([x, y, z])

        # Nessun punto libero: si prova il centro e si lascia decidere a safety.
        return np.array([0.0, 0.0, z])

    def _settle(self, steps: int = 40) -> None:
        """Lascia assestare gli oggetti prima di rileggere lo stato.

        Senza, i fatti verrebbero letti mentre un cubo sta ancora cadendo, e
        ``on(x, y)`` risulterebbe vero o falso a seconda del millisecondo.
        """
        action = np.zeros(self.env.action_dim)
        action[-1] = self.robot._gripper
        for _ in range(steps):
            self.env.step(action)
