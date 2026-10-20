"""Eseguire un piano, e riprendersi quando un passo fallisce.

Il pianificatore dice **quali** passi; qui si decide **come** eseguirli e cosa
fare quando uno non riesce. Sono due problemi diversi e conviene tenerli
separati: un piano sbagliato e' un errore di modello, riproducibile e
correggibile una volta per tutte; un passo che fallisce con un piano giusto e'
fisica, e va gestito mentre accade.

**Il recupero ha livelli.** Il benchmark LIBERO-RECOVER (2026) li ordina cosi':

1. *ritentare* l'azione identica
2. *adattare* l'azione (piu' in basso, piu' piano)
3. *rimettere a posto* l'oggetto, se e' finito altrove
4. *sistemare* la scena

Qui si copre il livello 1 con ``py_trees``, e i livelli 2-3 **senza codice
apposta**: dopo i ritentativi si rilegge lo stato vero e si ripianifica. Se il
cubo e' caduto, il nuovo piano lo raccoglie — perche' il pianificatore vede lo
stato cambiato, non perche' qualcuno abbia previsto quel caso.

Ritentare all'infinito e' l'errore da non fare: un robot che riprova
cinquecento volte la stessa presa fallita non e' tenace, e' rotto. Da qui il
limite, e la ripianificazione come seconda linea.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from robotica.planning.actions import Action
from robotica.planning.planner import Plan, plan, satisfies
from robotica.planning.state import Fact, State, describe

# Quante volte ripetere lo stesso passo prima di rileggere lo stato.
# Tre e' un compromesso: abbastanza per assorbire un fallimento casuale (la
# pinza scivola), non tanto da insistere su qualcosa di impossibile.
MAX_RETRY = 3

# Quante volte ripianificare prima di arrendersi. Protegge dal caso in cui
# ogni piano fallisce allo stesso modo: senza, si ripianificherebbe in eterno.
MAX_REPLAN = 3


@dataclass
class Step:
    """Come e' andato un passo, per il resoconto."""

    action: Action
    ok: bool
    attempts: int
    detail: str = ""

    def __str__(self) -> str:
        esito = "ok" if self.ok else "FALLITO"
        tentativi = "" if self.attempts == 1 else f" ({self.attempts} tentativi)"
        coda = f" — {self.detail}" if self.detail else ""
        return f"{self.action}: {esito}{tentativi}{coda}"


@dataclass
class Outcome:
    """L'esito di un'esecuzione completa.

    Si valuta come vero/falso, come ``Plan`` e come i risultati di ``safety/``.
    """

    success: bool
    steps: list[Step] = field(default_factory=list)
    plans: int = 1
    reason: str = ""
    parts: int = 1
    """In quanti sotto-obiettivi era stato spezzato l'obiettivo.

    Serve a non confondere due cose molto diverse che producono entrambe piu'
    di un piano: **ripianificare** (qualcosa e' andato storto) e **decomporre**
    (l'obiettivo era grande e lo si e' diviso apposta). Dire "ripianificato 5
    volte" dopo uno smistamento perfetto e' un resoconto falso, e un resoconto
    falso vale meno di nessun resoconto.
    """

    def __bool__(self) -> bool:
        return self.success

    def explain(self) -> str:
        righe = []
        if self.parts > 1:
            extra = self.plans - self.parts
            nota = f", di cui {extra} ripianificati" if extra > 0 else ""
            righe.append(f"Obiettivo diviso in {self.parts} sotto-obiettivi{nota}.")
        elif self.plans > 1:
            righe.append(f"Ripianificato {self.plans - 1} volta/e.")
        righe += [f"  {s}" for s in self.steps]
        if self.reason:
            righe.append(f"\n{self.reason}")
        if self.success:
            righe.append("\nObiettivo raggiunto.")
        return "\n".join(righe)


Executor = Callable[[Action], tuple[bool, str]]
"""Chi esegue davvero un'azione sul robot.

Restituisce (riuscito, dettaglio). Tenerlo come parametro invece che cablarlo
permette di provare la logica di recupero senza fisica: un esecutore finto che
fallisce a comando prova in un secondo cio' che in simulazione richiederebbe
minuti.
"""

StateReader = Callable[[], State]
"""Come si rilegge lo stato vero dalla scena. Vedi ``planning/state.py``."""


def run_plan(
    the_plan: Plan,
    execute: Executor,
    max_retry: int = MAX_RETRY,
) -> tuple[bool, list[Step], int]:
    """Esegue i passi in ordine, ritentando quello che fallisce.

    Restituisce (tutto riuscito, resoconto, indice del passo fallito).
    L'indice serve a chi chiama per sapere **da dove** riprendere: il piano non
    va rifatto da capo se i primi passi sono andati a buon fine.
    """
    steps: list[Step] = []

    for i, action in enumerate(the_plan.actions):
        ok = False
        detail = ""
        attempts = 0

        while attempts < max_retry:
            attempts += 1
            ok, detail = execute(action)
            if ok:
                break

        steps.append(Step(action=action, ok=ok, attempts=attempts, detail=detail))
        if not ok:
            return False, steps, i

    return True, steps, -1


def achieve(
    goal: frozenset[Fact],
    read_state: StateReader,
    execute: Executor,
    objects: list[str],
    zones: list[str] | None = None,
    max_retry: int = MAX_RETRY,
    max_replan: int = MAX_REPLAN,
    on_plan: Callable[[Plan], None] | None = None,
    split: bool = True,
    on_note: Callable[[str], None] | None = None,
) -> Outcome:
    """Raggiunge un obiettivo: pianifica, esegue, e se serve ripianifica.

    E' la pipeline completa, meno la traduzione dell'intento:

        stato -> piano -> esecuzione -> (fallimento) -> stato -> piano -> ...

    ``on_plan`` viene chiamato con ogni piano **prima** di eseguirlo. Serve a
    mostrarlo, o a farlo approvare: un piano deterministico vale poco se non lo
    si puo' guardare prima che il braccio si muova.

    ``split`` permette di spezzare un obiettivo grande in sotto-obiettivi
    indipendenti (vedi ``decompose.py``). Serve con molti oggetti, dove cercare
    un piano unico costerebbe troppo. Si puo' disattivare per confrontare i due
    modi — che e' esattamente l'esperimento che rende visibile il guadagno.
    """
    if split:
        from robotica.planning.decompose import decompose

        parti = decompose(goal, read_state())
        if parti:
            if on_note is not None:
                on_note(parti.explain())
            return _achieve_in_parts(
                parti.parts, read_state, execute, objects, zones,
                max_retry, max_replan, on_plan, on_note,
            )
        if on_note is not None:
            on_note(parti.explain())

    all_steps: list[Step] = []

    for attempt in range(1, max_replan + 1):
        state = read_state()

        if satisfies(state, goal):
            return Outcome(success=True, steps=all_steps, plans=attempt)

        current = plan(state, goal, objects, zones)
        if not current:
            return Outcome(
                success=False,
                steps=all_steps,
                plans=attempt,
                reason=f"NESSUN PIANO: {current.reason}",
            )

        if on_plan is not None:
            on_plan(current)

        ok, steps, failed_at = run_plan(current, execute, max_retry)
        all_steps.extend(steps)

        if ok:
            # Si verifica sul mondo, non sul modello: il piano poteva essere
            # giusto e l'esecuzione imprecisa. E' la stessa distinzione fra
            # "eseguito" e "riuscito" che i tool MCP fanno gia'.
            if satisfies(read_state(), goal):
                return Outcome(success=True, steps=all_steps, plans=attempt)

            # Tutti i passi riusciti ma l'obiettivo no: il modello non
            # corrisponde al mondo. Ripianificare dallo stato vero e' l'unica
            # mossa sensata.
            continue

        failed = current.actions[failed_at]
        if attempt == max_replan:
            return Outcome(
                success=False,
                steps=all_steps,
                plans=attempt,
                reason=(
                    f"Arreso dopo {attempt} piani: '{failed}' continua a fallire.\n"
                    f"Stato attuale:\n{describe(read_state())}"
                ),
            )

    return Outcome(
        success=False,
        steps=all_steps,
        plans=max_replan,
        reason="Esaurite le ripianificazioni senza raggiungere l'obiettivo.",
    )


def _achieve_in_parts(
    parts: list[frozenset[Fact]],
    read_state: StateReader,
    execute: Executor,
    objects: list[str],
    zones: list[str] | None,
    max_retry: int,
    max_replan: int,
    on_plan: Callable[[Plan], None] | None,
    on_note: Callable[[str], None] | None,
) -> Outcome:
    """Raggiunge i sotto-obiettivi uno alla volta.

    Ogni parte parte dallo stato lasciato dalla precedente — non da una
    simulazione di dove *dovrebbe* essere — perche' vale la stessa regola di
    sempre: lo stato si rilegge dalla scena. E' anche il motivo per cui questo
    ciclo e' robusto ai fallimenti: se una parte va storta, la successiva vede
    il mondo com'e' davvero, non come sarebbe dovuto essere.

    **Un fallimento non ferma le altre parti.** Sono indipendenti per
    costruzione, quindi arrendersi su tutto perche' un cubo e' caduto dal tavolo
    significherebbe lasciare gli altri tre dove sono per niente. Si prosegue e
    si riporta com'e' andata: tre su quattro e' un esito, non un errore.
    """
    all_steps: list[Step] = []
    plans = 0
    falliti: list[str] = []

    for i, part in enumerate(parts, 1):
        if on_note is not None:
            from robotica.planning.goals import describe_goal

            testo = describe_goal(part).strip()
            on_note(f"[{i}/{len(parts)}] {testo}")

        out = achieve(
            part,
            read_state,
            execute,
            objects,
            zones,
            max_retry=max_retry,
            max_replan=max_replan,
            on_plan=on_plan,
            split=False,  # gia' decomposto: spezzare ancora sarebbe un ciclo
            on_note=on_note,
        )

        all_steps.extend(out.steps)
        plans += out.plans
        if not out:
            falliti.append(f"sotto-obiettivo {i}: {out.reason}")

    if falliti:
        return Outcome(
            success=False,
            steps=all_steps,
            plans=plans,
            parts=len(parts),
            reason=(
                f"{len(falliti)} sotto-obiettivi su {len(parts)} non raggiunti:\n"
                + "\n".join(f"  {r}" for r in falliti)
            ),
        )

    return Outcome(success=True, steps=all_steps, plans=plans, parts=len(parts))
