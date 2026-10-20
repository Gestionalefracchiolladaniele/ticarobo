"""Registro delle chiamate e autorizzazione al movimento.

Due presidi che stanno accanto a ``safety/`` ma rispondono a domande diverse.
``safety/`` chiede *"questo comando e' geometricamente sicuro?"*. Qui si chiede
*"chi ha autorizzato il movimento, e cosa e' stato eseguito davvero?"*.

**Registro.** Per un progetto il cui valore e' la verifica (D2), non sapere
cosa l'agent ha chiamato e' una lacuna: senza registro, un fallimento si puo'
solo raccontare, non ricostruire. Ogni chiamata finisce qui, con esito e
motivo del rifiuto.

**Autorizzazione.** Il movimento va abilitato prima di essere eseguito, e
l'abilitazione **scade**. Un agent che legge un comando in una pagina web e
decide di muovere il braccio trova il sistema disarmato. La scadenza serve
perche' un'autorizzazione data un'ora fa per un compito finito non copre cio'
che l'agent decide adesso.

L'idea viene da ``robot-runtime`` (MIT), che la formula bene: un controllo di
sicurezza raggiungibile solo passando dall'agent non e' un controllo di
sicurezza. Per questo la revoca non richiede nulla e vale subito.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Durata dell'autorizzazione. Abbastanza per una sequenza di manipolazione,
# poco perche' un'autorizzazione dimenticata resti attiva a lungo.
ARM_DURATION = 120.0

# Quante chiamate si tengono. Il registro sta in memoria: serve a ricostruire
# la sessione corrente, non a fare da archivio storico.
MAX_ENTRIES = 200


@dataclass
class Entry:
    """Una chiamata a un tool, con il suo esito."""

    tool: str
    args: str
    outcome: str  # "ok" | "rifiutato" | "fallito"
    detail: str
    at: float = field(default_factory=time.time)

    def __str__(self) -> str:
        when = time.strftime("%H:%M:%S", time.localtime(self.at))
        line = f"{when}  {self.outcome.upper():10} {self.tool}({self.args})"
        return f"{line}\n{' ' * 12}{self.detail}" if self.detail else line


class AuditLog:
    """Le chiamate della sessione, in ordine."""

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        self._entries: list[Entry] = []
        self._max = max_entries

    def record(self, tool: str, args: str, outcome: str, detail: str = "") -> None:
        self._entries.append(Entry(tool, args, outcome, detail))
        if len(self._entries) > self._max:
            del self._entries[: -self._max]

    def recent(self, limit: int = 20) -> list[Entry]:
        return self._entries[-limit:]

    def counts(self) -> dict[str, int]:
        """Quante chiamate per esito. Dice a colpo d'occhio come sta andando."""
        out: dict[str, int] = {}
        for e in self._entries:
            out[e.outcome] = out.get(e.outcome, 0) + 1
        return out

    def __len__(self) -> int:
        return len(self._entries)


class MotionGate:
    """Autorizzazione al movimento, con scadenza.

    Le letture (``get_scene``) restano sempre libere: osservare non muove
    nulla, e un agent che non puo' guardare non puo' nemmeno capire perche'
    e' stato fermato.
    """

    def __init__(self, duration: float = ARM_DURATION) -> None:
        self.duration = duration
        self._until = 0.0
        self._reason = ""

    @property
    def armed(self) -> bool:
        return time.time() < self._until

    @property
    def remaining(self) -> float:
        return max(0.0, self._until - time.time())

    def arm(self, reason: str) -> float:
        """Abilita il movimento. Il motivo finisce nel registro."""
        self._reason = reason
        self._until = time.time() + self.duration
        return self.duration

    def disarm(self) -> None:
        """Revoca subito. Non richiede nulla, di proposito."""
        self._until = 0.0
        self._reason = ""

    def refuse_reason(self) -> str:
        """Perche' il movimento e' bloccato, e cosa fare."""
        return (
            "RIFIUTATO: il movimento non e' autorizzato.\n"
            "Chiama arm_motion(reason='...') per abilitarlo, dichiarando cosa "
            f"stai per fare. L'autorizzazione dura {self.duration:.0f} secondi.\n"
            "Le letture come get_scene restano sempre disponibili."
        )

    def status(self) -> str:
        if not self.armed:
            return "movimento NON autorizzato"
        return f"movimento autorizzato ancora per {self.remaining:.0f} s ({self._reason})"
