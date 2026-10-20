"""Planning: dire **cosa** si vuole, non **come** ottenerlo.

La differenza col resto del progetto sta tutta qui. Con i tool MCP si dice
*prendi il cubo rosso, posalo in (0.14, -0.22)*: il come. Con il planning si
dice *voglio il blu sopra il rosso*, e i passi li trova il sistema.

La pipeline, in ordine:

1. **obiettivo** (``goals``) — l'intento tradotto in fatti, e verificato prima
   di muovere qualsiasi cosa
2. **stato** (``state``) — la scena letta come fatti, non come coordinate
3. **piano** (``planner``) — la sequenza piu' corta, deterministica e
   ispezionabile prima dell'esecuzione
4. **esecuzione** (``execution``) — i passi eseguiti, ritentando quelli che
   falliscono e ripianificando se ritentare non basta
5. **ponte** (``bridge``) — le azioni logiche tradotte in movimenti, passando
   dalle verifiche di ``safety/``

Con molti oggetti si aggiunge un passo prima del piano: **decomporre**
(``decompose``). Smistare quattro cubi non e' un problema da otto passi, sono
quattro problemi da due — e risolverli separatamente evita che la ricerca
esplori un numero di combinazioni che cresce come una potenza.

Cio' che e' deterministico e' il **piano**, non l'**esecuzione**: la stessa
sequenza puo' riuscire o fallire a seconda di come la presa tiene. La
distinzione e' utile quando qualcosa va storto — un piano sbagliato e' un
errore di modello, riproducibile; un piano giusto che fallisce e' fisica.

Vedi D13 in ``docs/decisioni.md``.
"""

from robotica.planning.actions import Action, all_actions
from robotica.planning.decompose import Decomposition, decompose
from robotica.planning.execution import Outcome, achieve, run_plan
from robotica.planning.goals import GoalError, describe_goal, parse_goal
from robotica.planning.planner import Plan, plan, satisfies
from robotica.planning.state import (
    Fact,
    State,
    WorldModel,
    arm_empty,
    clear,
    color,
    describe,
    holding,
    in_zone,
    on,
    on_table,
)

__all__ = [
    "Action",
    "Decomposition",
    "Fact",
    "GoalError",
    "Outcome",
    "Plan",
    "State",
    "WorldModel",
    "achieve",
    "all_actions",
    "arm_empty",
    "clear",
    "color",
    "decompose",
    "describe",
    "describe_goal",
    "holding",
    "in_zone",
    "on",
    "on_table",
    "parse_goal",
    "plan",
    "run_plan",
    "satisfies",
]
