"""Far girare le partite: **un solo motore** per il gioco vero e per il banco.

Il ciclo di una mano e' di quattro passi e l'ordine non e' negoziabile:

1. il predittore **dichiara** la previsione (senza vedere la mossa)
2. l'umano tira
3. si calcola l'esito
4. il predittore **osserva** la mossa vera

Se questo ciclo esistesse in due copie — una per il terminale, una per il banco
— prima o poi divergerebbero, e i numeri del banco descriverebbero un programma
diverso da quello che gioca in piazza. Con una copia sola, misurare il banco
significa misurare il gioco.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from robotica.games.rps.avversari import Avversario, esito_umano, legge_l_annuncio
from robotica.games.rps.gioco import Esito, Mossa, esito
from robotica.games.rps.memoria import Piazza
from robotica.games.rps.predittore import Predittore, Previsione
from robotica.games.rps.registro import Registro


def gioca_contro(
    avversario: Avversario,
    partite: int,
    predittore: Predittore | None = None,
) -> Registro:
    """Fa giocare il predittore contro un avversario finto, e registra tutto.

    L'avversario riceve la storia di **entrambi**: alcune strategie reagiscono
    a cio' che il robot ha fatto (``anti_ultimo``, ``win_stay``), ed e' giusto
    che possano. Il predittore invece riceve solo le mosse dell'umano, e solo
    dopo che sono state giocate.
    """
    pred = predittore if predittore is not None else Predittore(seme=0)
    reg = Registro()
    mie: list[Mossa] = []  # le mosse dell'avversario
    sue: list[Mossa] = []  # le mosse del robot

    for _ in range(partite):
        previsione = pred.prevedi()
        # Il lettore vede lo schermo, cioe' solo l'annuncio: mai la mossa.
        if legge_l_annuncio(avversario):
            mossa_umana = avversario(mie, sue, previsione.attesa)
        else:
            mossa_umana = avversario(mie, sue)

        reg.aggiungi(previsione, mossa_umana, esito(previsione.mossa, mossa_umana))

        pred.osserva(mossa_umana, previsione)
        mie.append(mossa_umana)
        sue.append(previsione.mossa)

    return reg


@dataclass
class Tavolo:
    """Lo stesso ciclo, spezzato in due tempi per chi gioca **a eventi**.

    Una finestra non aspetta come ``input()``: ridisegna di continuo e il tasto
    arriva quando arriva. Il ciclo resta quello sopra, nello stesso ordine:

    - ``previsione()`` e' il passo 1, e si calcola **una volta per mano**. Non
      e' una lettura: ``prevedi()`` memorizza cosa ha detto ogni modello (serve
      al credito) e consuma il generatore casuale. Chiamarlo a ogni fotogramma
      cambierebbe la previsione sotto gli occhi del giocatore.
    - ``tira()`` fa i passi 2-4: esito, registro, e solo alla fine ``osserva``.
    """

    predittore: Predittore
    mani: int = 10
    registro: Registro = field(default_factory=Registro)
    #: La memoria di piazza, se c'e': il fondo da cui parte ogni sfidante, e
    #: dove finisce cio' che ha insegnato quando se ne va.
    piazza: Piazza | None = None
    _corrente: Previsione | None = None

    def __post_init__(self) -> None:
        if self.piazza is not None:
            self.predittore.innesta(self.piazza.fondo())

    def previsione(self) -> Previsione:
        if self._corrente is None:
            self._corrente = self.predittore.prevedi()
        return self._corrente

    def tira(self, umana: Mossa) -> tuple[Previsione, Esito]:
        """La mossa dell'umano: chiude la mano e restituisce cosa e' successo."""
        prev = self.previsione()
        e = esito(prev.mossa, umana)
        self.registro.aggiungi(prev, umana, e)
        self.predittore.osserva(umana, prev)
        self._corrente = None
        return prev, e

    @property
    def finita(self) -> bool:
        return len(self.registro) >= self.mani

    def rivincita(self) -> None:
        """Partita nuova, **stessa persona**: il predittore ricorda."""
        self.registro = Registro()

    def congeda(self) -> None:
        """Lo sfidante se ne va: cio' che ha insegnato entra nella piazza.

        Si chiama una volta per sfidante (``nuovo_giocatore`` lo fa, e chi
        chiude il programma lo chiama per l'ultimo). Chiamarla due volte
        verserebbe due volte le stesse mani.
        """
        if self.piazza is not None and len(self.predittore.storia):
            self.piazza.versa(self.predittore.raccolto())

    def nuovo_giocatore(self) -> None:
        """Persona nuova: si dimentica **la persona**, non la piazza."""
        self.congeda()
        self.predittore.azzera()
        if self.piazza is not None:
            self.predittore.innesta(self.piazza.fondo())
        self.registro = Registro()
        self._corrente = None


def gioca_piazza(
    fabbrica: Callable[[int], Avversario],
    persone: int,
    mani: int,
    seme: int = 7,
    piazza: Piazza | None = None,
    impara: bool = True,
    pubblica: bool = True,
) -> Registro:
    """Tante persone, poche mani ciascuna, predittore **nuovo** per ognuna.

    E' la misura che conta per la piazza: dieci mani da sole sono quasi tutto
    rumore, il netto sommato su trecento persone no. ``fabbrica(seme)``
    costruisce la persona ``i``-esima col seme ``seme + i``, cosi' la misura
    gira uguale ogni volta.

    Con ``piazza`` ogni persona parte dal fondo della popolazione e, se
    ``impara``, ci entra quando se ne va: e' la memoria di piazza.
    """
    tot = Registro()
    for i in range(persone):
        pred = Predittore(seme=seme + i, pubblica=pubblica)
        if piazza is not None:
            pred.innesta(piazza.fondo())
        reg = gioca_contro(fabbrica(seme + i), mani, pred)
        if piazza is not None and impara:
            piazza.versa(pred.raccolto())
        tot.partite.extend(reg.partite)
    return tot


def riassunto_banco(partite: int = 1000, seme: int = 1234) -> list[tuple[str, Registro]]:
    """Il predittore contro tutti gli avversari del banco.

    Ognuno con un predittore **nuovo**: le abitudini di un avversario non devono
    contaminare la misura sul successivo, esattamente come in piazza si azzera a
    ogni persona.
    """
    from robotica.games.rps.avversari import banco

    return [
        (nome, gioca_contro(avv, partite, Predittore(seme=seme)))
        for nome, avv in banco(seme).items()
    ]


__all__ = ["Tavolo", "gioca_contro", "gioca_piazza", "riassunto_banco", "esito_umano"]
