"""Le regole di sasso-carta-forbici, in un posto solo.

Sembra troppo poco per meritare un modulo. Lo merita per un motivo: la tabella
"chi batte chi" e' l'unico punto del progetto dove un errore di segno passa
inosservato. Un predittore che sceglie la mossa *perdente* invece di quella
vincente funziona benissimo — conta, prevede, spiega — e perde il 66% delle
partite. Il sintomo non parla di segni: parla di un predittore scarso.

Per questo l'esito non e' scritto a mano ma **calcolato**: le tre mosse stanno
in cerchio, ognuna batte la precedente. In aritmetica modulo 3::

    (a - b) % 3 == 1   ->   a batte b

Non c'e' nessuna casella da sbagliare, perche' non ci sono caselle.
"""

from __future__ import annotations

from enum import IntEnum


class Mossa(IntEnum):
    """Le tre mosse, in ordine di cerchio: ognuna batte la precedente.

    L'ordine **non e' arbitrario**: carta batte sasso, forbici batte carta,
    sasso batte forbici. Cambiare l'ordine dei valori rompe ``batte`` senza
    rompere nient'altro, quindi non si cambia.
    """

    SASSO = 0
    CARTA = 1
    FORBICI = 2

    def __str__(self) -> str:
        return self.name.lower()

    @property
    def lettera(self) -> str:
        """L'iniziale, per digitarla da terminale."""
        return self.name[0].lower()


#: Le tre mosse in ordine, per iterare senza dipendere da ``IntEnum.__iter__``.
MOSSE: tuple[Mossa, ...] = (Mossa.SASSO, Mossa.CARTA, Mossa.FORBICI)


def batte(a: Mossa, b: Mossa) -> bool:
    """``a`` vince contro ``b``?"""
    return (a - b) % 3 == 1


def vince_contro(m: Mossa) -> Mossa:
    """La mossa che batte ``m``.

    E' la funzione che usa il predittore: prevede la mossa dell'umano, poi gioca
    questa. Un solo punto in cui puo' esserci il segno sbagliato, e c'e' un test
    che lo verifica su tutte e tre le mosse.
    """
    return Mossa((m + 1) % 3)


def perde_contro(m: Mossa) -> Mossa:
    """La mossa che perde contro ``m``. Serve agli avversari finti del banco."""
    return Mossa((m - 1) % 3)


class Esito(IntEnum):
    """L'esito **dal punto di vista del robot**.

    I valori sono scelti perche' sommarli dia il punteggio netto: +1 vittoria,
    -1 sconfitta, 0 patta. Il netto ha baseline **zero**, il che rende immediato
    dire se si sta facendo meglio del caso; il tasso di vittoria no, perche' le
    patte lo comprimono.
    """

    SCONFITTA = -1
    PATTA = 0
    VITTORIA = 1

    def __str__(self) -> str:
        return self.name.lower()


def esito(robot: Mossa, umano: Mossa) -> Esito:
    """Chi ha vinto, dal punto di vista del robot."""
    if robot == umano:
        return Esito.PATTA
    return Esito.VITTORIA if batte(robot, umano) else Esito.SCONFITTA


def leggi_mossa(testo: str) -> Mossa | None:
    """Interpreta cio' che l'utente ha digitato: ``s``, ``carta``, ``F``...

    Restituisce ``None`` se non si capisce, invece di sollevare: al terminale
    una battuta a vuoto non e' un errore del programma, e in piazza "non ho
    capito il gesto" e' un esito che si **conta**, non si evita.
    """
    t = testo.strip().lower()
    if not t:
        return None
    for m in MOSSE:
        if t == m.lettera or t == str(m):
            return m
    return None
