"""Cosa il robot sa fare, e a quali condizioni.

Un'azione dichiara tre cose:

- **precondizioni** — cosa deve essere vero perche' si possa fare
- **effetti positivi** — cosa diventa vero dopo
- **effetti negativi** — cosa smette di essere vero

E' la stessa idea di ``safety/``, alzata di un livello. Li' si verifica che *un
comando* sia geometricamente sicuro; qui che *una sequenza* abbia senso. Sono
complementari: un piano logicamente corretto puo' contenere un movimento
impossibile, e un movimento sicuro puo' far parte di un piano assurdo.

Le azioni sono le quattro canoniche del *blocks world*. Prendere dal tavolo e
prendere da una pila sembrano la stessa cosa e non lo sono: togliendo un cubo
da una pila, quello sotto **diventa libero**. Unirle farebbe perdere
quell'effetto, e il pianificatore non saprebbe che per liberare un cubo basta
togliere quello sopra — che e' esattamente cio' che serve per riordinare una
pila.
"""

from __future__ import annotations

from dataclasses import dataclass

from robotica.planning.state import (
    Fact,
    State,
    arm_empty,
    clear,
    holding,
    in_zone,
    on,
    on_table,
)


@dataclass(frozen=True)
class Action:
    """Un'azione istanziata: ``pickup(cube_red)``, non ``pickup(X)``.

    Il pianificatore lavora su azioni gia' applicate a oggetti concreti, perche'
    deve poterle mettere in sequenza e simularne l'effetto.
    """

    name: str
    args: tuple[str, ...]
    pre: frozenset[Fact]
    add: frozenset[Fact]
    remove: frozenset[Fact]

    def applicable(self, state: State) -> bool:
        """Se l'azione si puo' fare in questo stato.

        Basta che le precondizioni siano un sottoinsieme dello stato: vale
        l'ipotesi del mondo chiuso, quindi cio' che non e' scritto e' falso.
        """
        return self.pre <= state

    def apply(self, state: State) -> State:
        """Lo stato dopo l'azione.

        Prima si tolgono gli effetti negativi, poi si aggiungono i positivi.
        L'ordine conta quando un fatto compare in entrambi: l'aggiunta vince,
        che e' la semantica standard di STRIPS.
        """
        return frozenset((state - self.remove) | self.add)

    def missing(self, state: State) -> frozenset[Fact]:
        """Le precondizioni che mancano. Serve a spiegare un rifiuto."""
        return frozenset(self.pre - state)

    def __str__(self) -> str:
        return f"{self.name}({', '.join(self.args)})"


# --- le quattro azioni del blocks world, piu' una per le zone ----------------


def pickup(x: str) -> Action:
    """Prende ``x`` dal tavolo.

    Serve che sia libero (niente sopra) e che la pinza sia vuota. Dopo, ``x`` non
    e' piu' sul tavolo ne' libero: e' in mano.
    """
    return Action(
        name="pickup",
        args=(x,),
        pre=frozenset({clear(x), on_table(x), arm_empty()}),
        add=frozenset({holding(x)}),
        remove=frozenset({clear(x), on_table(x), arm_empty()}),
    )


def putdown(x: str) -> Action:
    """Posa sul tavolo ``x`` che si ha in mano."""
    return Action(
        name="putdown",
        args=(x,),
        pre=frozenset({holding(x)}),
        add=frozenset({clear(x), on_table(x), arm_empty()}),
        remove=frozenset({holding(x)}),
    )


def stack(x: str, y: str) -> Action:
    """Posa ``x`` sopra ``y``.

    Serve che ``y`` sia libero, altrimenti si posa su una pila e cade. Dopo,
    ``y`` non e' piu' libero: e' il fatto che impedisce di impilarci sopra un
    terzo cubo senza accorgersene.
    """
    return Action(
        name="stack",
        args=(x, y),
        pre=frozenset({holding(x), clear(y)}),
        add=frozenset({on(x, y), clear(x), arm_empty()}),
        remove=frozenset({holding(x), clear(y)}),
    )


def unstack(x: str, y: str) -> Action:
    """Toglie ``x`` da sopra ``y``.

    **L'effetto che distingue questa azione da ``pickup``:** ``y`` torna libero.
    E' cio' che permette al pianificatore di scoprire da solo che per prendere
    un cubo sepolto deve prima sgombrarlo — nessuno glielo insegna, e' una
    conseguenza del modello.
    """
    return Action(
        name="unstack",
        args=(x, y),
        pre=frozenset({on(x, y), clear(x), arm_empty()}),
        add=frozenset({holding(x), clear(y)}),
        remove=frozenset({on(x, y), clear(x), arm_empty()}),
    )


def put_in_zone(x: str, zone: str) -> Action:
    """Posa ``x`` dentro una zona.

    Non e' del blocks world classico: serve allo smistamento. Dal punto di vista
    logico e' un ``putdown`` che aggiunge anche l'appartenenza alla zona — sul
    tavolo ci si posa comunque, ma in un punto scelto.
    """
    return Action(
        name="put_in_zone",
        args=(x, zone),
        pre=frozenset({holding(x)}),
        add=frozenset({clear(x), on_table(x), arm_empty(), in_zone(x, zone)}),
        remove=frozenset({holding(x)}),
    )


def all_actions(objects: list[str], zones: list[str]) -> list[Action]:
    """Tutte le azioni possibili su questi oggetti.

    Il pianificatore ha bisogno dell'elenco completo per cercare: prova quelle
    applicabili e scarta le altre. Con pochi oggetti l'elenco e' piccolo — con n
    oggetti le azioni sono nell'ordine di n**2, che per 5 cubi fa una manciata.

    Le combinazioni impossibili (impilare un cubo su se' stesso) sono escluse
    qui invece che nelle precondizioni: e' piu' economico non generarle.
    """
    actions: list[Action] = []

    for x in objects:
        actions.append(pickup(x))
        actions.append(putdown(x))

        for y in objects:
            if x == y:
                continue
            actions.append(stack(x, y))
            actions.append(unstack(x, y))

        for z in zones:
            actions.append(put_in_zone(x, z))

    return actions
