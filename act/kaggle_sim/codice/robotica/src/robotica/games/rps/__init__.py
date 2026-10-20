"""Sasso-carta-forbici: vincere **prevedendo**, non barando sulla velocita'.

Il robot di Tokyo (Ishikawa Watanabe) vince il 100% delle volte leggendo la mano
mentre si forma e rispondendo in 1 ms. E' un trionfo di hardware, ed e' un
imbroglio sulla velocita'.

Qui si fa la cosa opposta: la previsione e' **pubblica prima** che la persona
tiri, quindi non c'e' niente da leggere di nascosto. Il robot vince perche' ha
capito come pensi, e quando sbaglia hai battuto una macchina.

I pezzi, dal piu' indipendente al piu' legato al robot:

- ``gioco`` — le regole. Non sa cosa sia un robot
- ``predittore`` — cinque modelli in gara a punteggio decaduto: gioca **sempre**
  la mossa col miglior valore atteso, e dichiara la previsione solo quando e'
  onesto farlo. Le due decisioni sono separate
- ``registro`` — i numeri, e **se sono una prova o fortuna**
- ``avversari`` — comportamenti noti, per dire se il codice e' giusto
- ``sessione`` — il ciclo di una mano, uno solo per gioco vero e banco

Nessuno di questi importa MuJoCo. **Il predittore trasferisce al 100%**: quando
il braccio arrivera', questo codice non cambia di una riga — cambia solo chi fa
il gesto.

Si prova da terminale::

    uv run python -m robotica.games.rps.cli gioca
    uv run python -m robotica.games.rps.cli banco

Il metro e i numeri: ``esperimenti/so101/sasso-carta-forbici.md``.
"""

from robotica.games.rps.gioco import Esito, Mossa, batte, esito, leggi_mossa, vince_contro
from robotica.games.rps.predittore import Predittore, Previsione, Storia, valore_atteso
from robotica.games.rps.registro import CASO, Partita, Registro
from robotica.games.rps.sessione import gioca_contro, riassunto_banco

__all__ = [
    "CASO",
    "Esito",
    "Mossa",
    "Partita",
    "Predittore",
    "Previsione",
    "Registro",
    "Storia",
    "batte",
    "esito",
    "gioca_contro",
    "leggi_mossa",
    "riassunto_banco",
    "valore_atteso",
    "vince_contro",
]
