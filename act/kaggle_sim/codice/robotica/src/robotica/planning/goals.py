"""Gli obiettivi: cosa si vuole, non come ottenerlo.

E' il punto in cui l'intento in italiano diventa qualcosa che il pianificatore
puo' usare. La traduzione la fa un LLM; qui si definisce **il vocabolario che
puo' usare** e si **verifica** cio' che produce.

La verifica non e' un dettaglio. La ricerca misura la traduzione di obiettivi
chiari al 98-100%, ma degli obiettivi ambigui al 14-19%: *"metti a posto i
cubi"* e' indovinare. Il punto e' che il fallimento sia **visibile prima del
movimento** — si legge l'obiettivo tradotto e si vede subito che ha capito
male, invece di scoprirlo guardando il braccio sbagliare.

E' la stessa idea di D2, spostata a monte: l'agent propone, il sistema
verifica.
"""

from __future__ import annotations

from robotica.planning.state import Fact, in_zone, on, on_table

# I predicati che un obiettivo puo' usare. Non tutti quelli dello stato:
# `holding` e `clear` descrivono il mondo, ma come *obiettivo* non hanno senso —
# nessuno vuole "che la pinza tenga il cubo" come fine in se'.
#
# `color` non c'e' per un motivo diverso: e' un fatto **statico**. Chiedere
# "che il cubo sia rosso" come obiettivo non ha senso, perche' nessuna azione
# puo' renderlo vero. Il colore si usa per *selezionare* gli oggetti, ed e' cio'
# che fa `color_in_zone`.
GOAL_PREDICATES = {
    "on": 2,  # on(x, y): x sopra y
    "on_table": 1,  # on_table(x): x sul tavolo
    "in_zone": 2,  # in_zone(x, zona): x dentro la zona
    "color_in_zone": 2,  # color_in_zone(colore, zona): tutti i cubi di quel colore
}


class GoalError(ValueError):
    """Un obiettivo che non si puo' usare, col motivo scritto per essere letto."""


def parse_goal(
    spec: list[tuple[str, ...]],
    objects: list[str],
    zones: list[str],
    colors: dict[str, str] | None = None,
) -> frozenset[Fact]:
    """Costruisce un obiettivo verificato da una lista di predicati.

    ``spec`` e' quello che l'LLM produce, per esempio::

        [("on", "cube_blue", "cube_red")]
        [("in_zone", "cube_red", "green"), ("in_zone", "cube_blue", "green")]
        [("color_in_zone", "rosso", "buoni")]

    ``colors`` mappa oggetto -> colore e serve solo a ``color_in_zone``, che e'
    **quantificato**: dice "tutti i rossi" e va espanso nei singoli cubi prima
    di arrivare al pianificatore. L'espansione avviene qui, non nel
    pianificatore, per una ragione precisa — cosi' il piano resta leggibile in
    termini di oggetti concreti, e chi lo guarda prima di eseguirlo vede
    *quali* cubi si muoveranno, non una formula.

    Sollevare invece di restituire un errore e' voluto: un obiettivo malformato
    non deve arrivare al pianificatore, che lo troverebbe semplicemente
    irraggiungibile senza saper dire perche'.
    """
    colors = colors or {}
    if not spec:
        raise GoalError("obiettivo vuoto: serve almeno un fatto da raggiungere")

    facts: set[Fact] = set()

    for item in spec:
        if not item:
            raise GoalError("predicato vuoto nell'obiettivo")

        name, *args = item

        if name not in GOAL_PREDICATES:
            raise GoalError(
                f"predicato '{name}' sconosciuto. "
                f"Disponibili: {', '.join(sorted(GOAL_PREDICATES))}"
            )

        atteso = GOAL_PREDICATES[name]
        if len(args) != atteso:
            raise GoalError(
                f"'{name}' vuole {atteso} argomento/i, ne ha ricevuti {len(args)}: {args}"
            )

        if name == "on":
            x, y = args
            _check_object(x, objects)
            _check_object(y, objects)
            if x == y:
                raise GoalError(f"'{x}' non puo' stare sopra se stesso")
            facts.add(on(x, y))

        elif name == "on_table":
            _check_object(args[0], objects)
            facts.add(on_table(args[0]))

        elif name == "in_zone":
            x, z = args
            _check_object(x, objects)
            _check_zone(z, zones)
            facts.add(in_zone(x, z))

        elif name == "color_in_zone":
            c, z = args
            _check_zone(z, zones)

            if not colors:
                raise GoalError(
                    "questa scena non conosce i colori: 'color_in_zone' non e' utilizzabile. "
                    "Usa 'in_zone' elencando gli oggetti."
                )

            scelti = sorted(o for o, oc in colors.items() if oc == c and o in objects)
            if not scelti:
                raise GoalError(
                    f"nessun oggetto di colore '{c}' nella scena. "
                    f"Colori presenti: {', '.join(sorted(set(colors.values())))}"
                )

            facts.update(in_zone(o, z) for o in scelti)

    _check_consistency(facts)
    return frozenset(facts)


def _check_object(name: str, objects: list[str]) -> None:
    if name not in objects:
        raise GoalError(
            f"oggetto '{name}' non presente nella scena. Disponibili: {', '.join(sorted(objects))}"
        )


def _check_zone(name: str, zones: list[str]) -> None:
    if name not in zones:
        raise GoalError(
            f"zona '{name}' inesistente. Disponibili: {', '.join(sorted(zones))}"
        )


def _check_consistency(facts: set[Fact]) -> None:
    """Contraddizioni che si vedono senza pianificare.

    Costa poco e da' un messaggio molto piu' chiaro di "nessun piano trovato":
    un obiettivo impossibile e' un errore di traduzione, non un problema di
    ricerca, e va detto come tale.
    """
    sopra: dict[str, str] = {}  # chi sta sopra -> su cosa
    sotto: dict[str, str] = {}  # chi sta sotto -> chi ci sta sopra

    for f in facts:
        if f.predicate != "on":
            continue
        x, y = f.args

        if x in sopra and sopra[x] != y:
            raise GoalError(f"'{x}' non puo' stare sopra sia '{sopra[x]}' sia '{y}'")
        if y in sotto and sotto[y] != x:
            raise GoalError(f"su '{y}' non possono stare sia '{sotto[y]}' sia '{x}'")

        sopra[x] = y
        sotto[y] = x

    # Cicli: a su b, b su a.
    for start in sopra:
        visto = {start}
        cur = start
        while cur in sopra:
            cur = sopra[cur]
            if cur in visto:
                raise GoalError(f"la pila richiesta si chiude su se' stessa attorno a '{cur}'")
            visto.add(cur)

    # Un oggetto non puo' essere sopra qualcosa e sul tavolo insieme.
    on_table_names = {f.args[0] for f in facts if f.predicate == "on_table"}
    conflitti = on_table_names & set(sopra)
    if conflitti:
        nome = sorted(conflitti)[0]
        raise GoalError(f"'{nome}' non puo' essere sul tavolo e sopra '{sopra[nome]}' insieme")


def describe_goal(goal: frozenset[Fact]) -> str:
    """L'obiettivo in italiano, da mostrare prima di pianificare.

    Serve a far vedere *cosa si e' capito*: e' il punto in cui un errore di
    traduzione si scopre a costo zero.
    """
    righe = []
    for f in sorted(goal, key=str):
        if f.predicate == "on":
            righe.append(f"  {f.args[0]} sopra {f.args[1]}")
        elif f.predicate == "on_table":
            righe.append(f"  {f.args[0]} sul tavolo")
        elif f.predicate == "in_zone":
            righe.append(f"  {f.args[0]} nella zona {f.args[1]}")
        else:
            righe.append(f"  {f}")
    return "\n".join(righe)
