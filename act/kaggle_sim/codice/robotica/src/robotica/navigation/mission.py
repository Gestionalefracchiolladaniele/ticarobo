"""La missione: ordini, batteria, e la decisione di rinunciare.

**Cosa rende un AMR diverso da un nastro trasportatore.** Un nastro porta le
cose da A a B: e' un movimento. Un AMR riceve **richieste che arrivano nel
tempo** e deve decidere in che ordine servirle, mentre l'ambiente cambia sotto.
Il valore sta nella decisione, non nella traiettoria.

**Le tre categorie, che sono quelle del nastro (D17) in un'altra forma.** Li'
c'erano due zone piu' un colore che si lascia passare, e *non agire non era
fallire*. Qui:

``servito``
    l'ordine e' stato portato a termine
``rinunciato``
    il robot ha **scelto** di non servirlo -- batteria insufficiente. Non e'
    un fallimento: e' la decisione giusta, e un robot che accetta tutto si
    ferma in mezzo al corridoio
``fallito``
    il robot ci ha provato e non ce l'ha fatta: percorso inesistente,
    ostacolo insuperabile, blocco

Contarli insieme renderebbe un esito non attribuibile. Un sistema che
rinuncia venti volte funziona benissimo; uno che fallisce venti volte e' rotto.

**La batteria non e' scenografia.** E' cio' che trasforma "vai la'" in una
decisione: ogni ordine ha un costo in energia, e accettarne uno che non si
riesce a concludere significa restare a secco lontano dalla base. Il
fallimento da congelare e' proprio quello -- *un robot che decide guardando
solo la distanza dell'ordine si scarica in fondo al corridoio*.

**Perche' non ``planning/``.** La scelta e' deliberata e vale spiegarla: il
pianificatore di ``planning/`` cerca **sequenze di azioni** per raggiungere uno
stato dichiarato, ed e' bravissimo a quello (D13). Qui il problema e' diverso:
non c'e' una sequenza da scoprire -- si va, si torna -- ma una **scelta
ripetuta sotto un vincolo di risorsa**. Sono due problemi distinti, e infilare
il secondo nel primo vorrebbe dire modellare la batteria come un predicato
simbolico, cioe' discretizzare una grandezza continua per far finta che sia
logica.

Il collegamento con ``planning/`` resta possibile e sta altrove: un obiettivo
``at(robot, zona)`` si dichiara e si risolve con lo stesso pianificatore. Ed
e' **il test vero della tesi del transfer**, perche' quel codice non e' stato
scritto per un robot mobile.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

# Quanta energia costa un metro percorso, in frazione di carica.
# Scelto perche' il magazzino sia percorribile alcune volte con una carica:
# 10 x 8 m, i percorsi tipici sono 6-10 m, e 0.012/m significa che con la
# carica piena si fanno ~80 m. E' un numero **dichiarato**, non misurato da
# un datasheet: nessuna simulazione conosce il consumo di un robot vero.
ENERGY_PER_METER = 0.012

# Quanta energia se ne va stando fermi (elettronica, sensori), al secondo.
IDLE_DRAIN_PER_SECOND = 0.0005

# Quanta carica si recupera al secondo in stazione.
CHARGE_PER_SECOND = 0.08

# Sotto questa carica il robot non accetta piu' ordini nuovi e torna a
# caricarsi. E' la soglia di **prudenza**, ed e' separata dal calcolo
# dell'energia necessaria proprio perche' un robot che parte con l'esatto
# necessario arriva con zero -- e zero non e' un margine.
RESERVE_FRACTION = 0.25


class OrderState(Enum):
    ATTESA = "in attesa"
    IN_CORSO = "in corso"
    SERVITO = "servito"
    RINUNCIATO = "rinunciato"
    FALLITO = "fallito"


@dataclass
class Order:
    """Un ordine di trasporto: prendi a ``pickup``, porta a ``dropoff``."""

    order_id: int
    pickup: str
    dropoff: str
    state: OrderState = OrderState.ATTESA
    reason: str = ""

    def __str__(self) -> str:
        return f"#{self.order_id} {self.pickup} -> {self.dropoff}"


@dataclass
class Battery:
    """La carica, e cosa si puo' permettere.

    ``level`` e' 0..1. Che sia una frazione e non ampere-ora e' voluto: il
    numero vero dipende dal robot, e quello che conta qui e' il **rapporto**
    fra quanto si ha e quanto serve.
    """

    level: float = 1.0
    reserve: float = RESERVE_FRACTION

    def consume_distance(self, meters: float) -> None:
        self.level = max(0.0, self.level - meters * ENERGY_PER_METER)

    def consume_idle(self, seconds: float) -> None:
        self.level = max(0.0, self.level - seconds * IDLE_DRAIN_PER_SECOND)

    def charge(self, seconds: float) -> None:
        self.level = min(1.0, self.level + seconds * CHARGE_PER_SECOND)

    def energy_for(self, meters: float) -> float:
        return meters * ENERGY_PER_METER

    def can_afford(self, meters: float) -> bool:
        """C'e' energia per ``meters``, **lasciando la riserva**?

        La riserva e' il punto. Senza, il robot accetta un ordine che consuma
        esattamente cio' che ha e arriva a destinazione con zero -- lontano
        dalla stazione di ricarica, cioe' fermo per sempre. La riserva e' la
        differenza fra "ce la faccio" e "ce la faccio e poi torno".
        """
        return self.level - self.energy_for(meters) >= self.reserve

    @property
    def low(self) -> bool:
        return self.level <= self.reserve

    def __str__(self) -> str:
        return f"{self.level * 100:.0f}%"


@dataclass
class MissionStats:
    """I contatori. **Separati per categoria**, come sul nastro (D17)."""

    served: int = 0
    declined: int = 0
    failed: int = 0
    recharges: int = 0
    distance_m: float = 0.0
    replans: int = 0
    obstacles_avoided: int = 0
    collisions: int = 0
    # il caso peggiore: il robot si e' scaricato lontano dalla base
    stranded: bool = False

    def summary(self) -> str:
        return (
            f"serviti {self.served}, rinunciati {self.declined}, falliti {self.failed} | "
            f"{self.distance_m:.1f} m, {self.recharges} ricariche, "
            f"{self.replans} ripianificazioni, {self.collisions} urti"
            + (" | SCARICO LONTANO DALLA BASE" if self.stranded else "")
        )


class Dispatcher:
    """Decide quale ordine servire, e se rinunciare.

    **La politica e' esplicita e sostituibile**, perche' e' la cosa che
    l'esperimento mette alla prova. Due politiche, e la differenza si misura:

    ``"vicino"``
        serve l'ordine il cui punto di prelievo e' piu' vicino. E' la scelta
        ovvia, ed e' quella che **si scarica in fondo al corridoio**: guarda
        la distanza e ignora la carica.
    ``"prudente"``
        come sopra, ma **rifiuta** gli ordini che non puo' concludere
        lasciando la riserva, e torna a caricarsi quando serve.

    Che si possano confrontare e' il punto: la seconda vale qualcosa solo se
    la prima fallisce davvero, e un fallimento congelato e' cio' che lo
    dimostra (D16, D19).
    """

    def __init__(self, policy: str = "prudente") -> None:
        if policy not in ("vicino", "prudente"):
            raise ValueError(f"politica '{policy}' sconosciuta: 'vicino' o 'prudente'")
        self.policy = policy
        self.queue: deque[Order] = deque()
        self._next_id = 1

    def submit(self, pickup: str, dropoff: str) -> Order:
        """Accoda un ordine. Ritorna l'oggetto, cosi' se ne segue lo stato."""
        o = Order(self._next_id, pickup, dropoff)
        self._next_id += 1
        self.queue.append(o)
        return o

    @property
    def pending(self) -> list[Order]:
        return [o for o in self.queue if o.state is OrderState.ATTESA]

    def choose(
        self,
        robot_xy: np.ndarray,
        battery: Battery,
        distance_fn,
        charger_xy: np.ndarray,
    ) -> tuple[Order | None, str]:
        """Quale ordine servire adesso, e perche'.

        ``distance_fn(a, b)`` da' la lunghezza **del percorso**, non la
        distanza in linea d'aria: passa dal planner, quindi tiene conto degli
        scaffali. Usare la linea d'aria sottostimerebbe di molto in un
        magazzino, e il robot accetterebbe ordini che non puo' concludere --
        un altro numero plausibile al posto di una misura.

        Ritorna ``(ordine, motivo)``. Con ``None`` il motivo dice **perche'**:
        coda vuota, oppure nessun ordine sostenibile. Sono cose diverse, e un
        sistema che si arrende deve dire di essersi arreso (D14).
        """
        pending = self.pending
        if not pending:
            return None, "nessun ordine in attesa"

        scored: list[tuple[float, float, Order]] = []
        for o in pending:
            try:
                d_to_pickup = distance_fn(robot_xy, o.pickup)
                d_job = distance_fn(o.pickup, o.dropoff)
                d_home = distance_fn(o.dropoff, charger_xy)
            except Exception:
                # un ordine irraggiungibile non blocca gli altri: si segna e
                # si va avanti. Che sia FALLITO e non RINUNCIATO e' la
                # distinzione che conta -- qui il robot non ha scelto.
                o.state = OrderState.FALLITO
                o.reason = "nessun percorso verso questo ordine"
                continue
            scored.append((d_to_pickup, d_to_pickup + d_job + d_home, o))

        if not scored:
            return None, "nessun ordine raggiungibile"

        scored.sort(key=lambda t: t[0])

        if self.policy == "vicino":
            # la politica ingenua: il piu' vicino, punto. Non guarda la
            # carica -- ed e' esattamente il fallimento da congelare.
            return scored[0][2], "il piu' vicino"

        for _d_pick, d_total, o in scored:
            if battery.can_afford(d_total):
                return o, f"sostenibile: {d_total:.1f} m, carica {battery}"

        # nessuno sostenibile: **non e' un fallimento**, e' una decisione.
        # Il messaggio nomina il motivo perche' chi guarda non deve dedurlo.
        best = min(scored, key=lambda t: t[1])
        return None, (
            f"carica {battery} insufficiente: il piu' economico dei {len(scored)} "
            f"ordini costa {best[1]:.1f} m "
            f"({battery.energy_for(best[1]) * 100:.0f}% + {self_reserve(battery)}% di riserva)"
        )

    def decline_all(self, reason: str) -> int:
        """Rinuncia a tutti gli ordini in attesa. Ritorna quanti."""
        n = 0
        for o in self.pending:
            o.state = OrderState.RINUNCIATO
            o.reason = reason
            n += 1
        return n


def self_reserve(b: Battery) -> int:
    return int(b.reserve * 100)
