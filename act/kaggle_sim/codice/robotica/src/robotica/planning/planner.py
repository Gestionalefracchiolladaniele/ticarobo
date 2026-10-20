"""Dallo stato all'obiettivo: trovare la sequenza di azioni.

Il pianificatore fa una cosa sola: data la situazione attuale e cio' che si
vuole, restituisce i passi. E' **deterministico** — stesso stato e stesso
obiettivo danno sempre lo stesso piano, il che rende un fallimento
riproducibile e quindi diagnosticabile.

Attenzione a cosa e' deterministico: il *piano*, non l'*esecuzione*. La stessa
sequenza eseguita due volte puo' finire diversamente, perche' la presa puo'
cedere. E' una distinzione che vale la pena tenere a mente: se il piano e'
sbagliato il problema e' nel modello, se il piano e' giusto e l'esecuzione
fallisce il problema e' fisico. Senza questa separazione i due casi sono
indistinguibili.

**La ricerca e' in ampiezza**, quindi il primo piano trovato e' il piu' corto.
Con cinque cubi lo spazio degli stati e' piccolo e la ricerca finisce in
millisecondi; un algoritmo euristico servirebbe su problemi molto piu' grandi e
si capirebbe molto meno. Se un domani servisse, il modello e' gia' nella forma
che un motore PDDL si aspetta.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from robotica.planning.actions import Action, all_actions
from robotica.planning.state import Fact, State, describe

# Un piano piu' lungo di cosi', con pochi cubi, significa quasi sempre un
# modello sbagliato e non un problema difficile. Meglio fermarsi e dirlo.
MAX_DEPTH = 20

# Quanti stati visitare prima di arrendersi. Protegge dai casi in cui
# l'obiettivo e' irraggiungibile: senza, la ricerca esplorerebbe tutto lo
# spazio prima di accorgersene.
MAX_VISITED = 200_000


@dataclass
class Plan:
    """Il risultato di una pianificazione: i passi, o il motivo del fallimento.

    Si valuta come vero/falso (``if plan:``), come i risultati di ``safety/``.
    La convenzione e' la stessa di tutto il progetto: un esito che si puo'
    interrogare, con dentro il motivo.
    """

    actions: list[Action]
    reason: str = ""
    visited: int = 0

    def __bool__(self) -> bool:
        return self.reason == ""

    def __len__(self) -> int:
        return len(self.actions)

    def explain(self) -> str:
        """Il piano in forma leggibile, da mostrare prima di eseguire.

        Un piano deterministico serve a poco se non lo si puo' guardare: e' lo
        stesso principio di ``safety/``, che rifiuta *spiegando*. Qui si approva
        *vedendo*.
        """
        if not self:
            return f"NESSUN PIANO: {self.reason}"
        if not self.actions:
            return "Niente da fare: l'obiettivo e' gia' soddisfatto."

        righe = [f"Piano in {len(self.actions)} passi:"]
        righe += [f"  {i}. {a}" for i, a in enumerate(self.actions, 1)]
        return "\n".join(righe)


def satisfies(state: State, goal: frozenset[Fact]) -> bool:
    """Se lo stato soddisfa l'obiettivo.

    L'obiettivo e' **parziale**: chiede che certi fatti siano veri, non dice
    nulla sugli altri. ``on(blue, red)`` e' soddisfatto a prescindere da dove
    sia il terzo cubo. E' cio' che permette di esprimere "voglio questo" senza
    dover descrivere il mondo intero.
    """
    return goal <= state


def plan(
    state: State,
    goal: frozenset[Fact],
    objects: list[str],
    zones: list[str] | None = None,
    max_depth: int = MAX_DEPTH,
) -> Plan:
    """Cerca la sequenza piu' corta dallo stato all'obiettivo.

    Ricerca in ampiezza con memoria degli stati gia' visti. La memoria non e'
    un'ottimizzazione: senza, il pianificatore prenderebbe un cubo, lo
    poserebbe, lo riprenderebbe, all'infinito — sono tutti stati raggiungibili
    e tutti gia' visti.
    """
    if satisfies(state, goal):
        return Plan(actions=[])

    actions = all_actions(objects, zones or [])
    if not actions:
        return Plan(actions=[], reason="nessuna azione disponibile: manca l'elenco degli oggetti")

    # Ogni voce: (stato, passi fatti per arrivarci).
    frontier: deque[tuple[State, list[Action]]] = deque([(state, [])])
    seen: set[State] = {state}

    while frontier:
        current, path = frontier.popleft()

        if len(path) >= max_depth:
            continue

        for action in actions:
            if not action.applicable(current):
                continue

            nxt = action.apply(current)
            if nxt in seen:
                continue

            steps = [*path, action]
            if satisfies(nxt, goal):
                return Plan(actions=steps, visited=len(seen))

            seen.add(nxt)
            frontier.append((nxt, steps))

            if len(seen) > MAX_VISITED:
                return Plan(
                    actions=[],
                    reason=(
                        f"ricerca interrotta dopo {len(seen)} stati: "
                        "il limite e' stato raggiunto prima di esplorare tutte le "
                        "possibilita'.\n"
                        "NON significa che l'obiettivo sia irraggiungibile: "
                        "significa che questa ricerca si e' arresa.\n"
                        f"L'obiettivo riguarda {len({a for f in goal for a in f.args})} "
                        "fra oggetti e zone; la ricerca in ampiezza cresce come una "
                        "potenza e oltre i cinque oggetti non ce la fa.\n"
                        "Se i sotto-obiettivi sono indipendenti, vanno decomposti "
                        "(vedi planning/decompose.py e D14)."
                    ),
                    visited=len(seen),
                )

    return Plan(
        actions=[],
        reason=(
            f"nessuna sequenza porta all'obiettivo entro {max_depth} passi.\n"
            f"Obiettivo richiesto:\n{describe(goal)}\n"
            f"Stato di partenza:\n{describe(state)}"
        ),
        visited=len(seen),
    )
