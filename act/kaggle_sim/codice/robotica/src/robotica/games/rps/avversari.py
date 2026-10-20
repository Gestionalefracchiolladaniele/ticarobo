"""Avversari finti dal comportamento **noto**: il banco di prova del predittore.

Un predittore provato solo contro una persona vera non si puo' debuggare: se fa
il 45% non sai se il modello e' mediocre o se hai sbagliato un indice, perche'
non sai cosa *avrebbe dovuto* fare.

Qui ogni avversario ha una regola dichiarata, e da quella regola discende cosa
il predittore deve ottenere. Diventa un test, non un'impressione.

===================  =========================================  ==============
avversario           regola                                     atteso
===================  =========================================  ==============
``casuale``          uniforme 1/3                               **~33%**
``costante``         sempre la stessa                           ~100%
``ciclico``          sasso, carta, forbici, sasso...            ~100%
``ripetitore``       ripete la propria ultima mossa             ~100%
``alternante``       cambia sempre, con un giro fisso           ~100%
``win_stay``         tiene se vince, cambia se perde            **>50%**
``anti_ultimo``      gioca cio' che batte la tua ultima         ~99% (era 53%)
``sbilanciato``      sasso il 60% delle volte                   **>40%**
===================  =========================================  ==============

**Il piu' importante e' ``casuale``, ed e' quello che falsifica.** Contro una
sequenza davvero uniforme nessun predittore puo' superare 1/3: non c'e' niente
da prevedere. Se il numero esce alto, il predittore sta guardando la mossa
corrente — il baco piu' facile da fare qui e il piu' difficile da vedere, perche'
il sintomo ha l'aspetto di un successo.

``win_stay`` merita una riga in piu': e' il bias umano meglio documentato. Le
persone tengono la mossa che ha appena vinto e cambiano quando perdono. Non e'
un avversario di comodo, e' il motivo per cui la tesi del progetto sta in piedi.
"""

from __future__ import annotations

import random
from collections.abc import Callable

from robotica.games.rps.gioco import MOSSE, Esito, Mossa, perde_contro, vince_contro

#: Un avversario: date le proprie mosse passate e quelle del robot, la prossima.
#: Le due liste hanno la stessa lunghezza e sono in ordine cronologico.
Avversario = Callable[[list[Mossa], list[Mossa]], Mossa]


def casuale(seme: int | None = None) -> Avversario:
    """Uniforme sulle tre mosse. **Imbattibile**: il predittore deve fare ~33%."""
    rng = random.Random(seme)

    def gioca(mie: list[Mossa], sue: list[Mossa]) -> Mossa:
        return rng.choice(MOSSE)

    return gioca


def costante(mossa: Mossa = Mossa.SASSO) -> Avversario:
    """Sempre la stessa mossa. Il caso piu' facile: se qui non va, e' rotto."""

    def gioca(mie: list[Mossa], sue: list[Mossa]) -> Mossa:
        return mossa

    return gioca


def ciclico(passo: int = 1) -> Avversario:
    """Gira fra le tre mosse con passo fisso. Lo prende l'ordine 1."""

    def gioca(mie: list[Mossa], sue: list[Mossa]) -> Mossa:
        if not mie:
            return Mossa.SASSO
        return Mossa((mie[-1] + passo) % 3)

    return gioca


def ripetitore(prima: Mossa = Mossa.SASSO) -> Avversario:
    """Ripete la propria ultima mossa. Degenera in ``costante`` ma per un'altra
    via: qui e' l'ordine 1 a doverlo cogliere, non l'ordine 0."""

    def gioca(mie: list[Mossa], sue: list[Mossa]) -> Mossa:
        return mie[-1] if mie else prima

    return gioca


def win_stay_lose_shift(seme: int | None = None) -> Avversario:
    """Tiene se ha vinto, cambia se ha perso. **Il bias umano vero.**

    Su patta cambia a caso: le persone in patta non hanno una regola stabile,
    ed e' la parte che rende questo avversario non del tutto prevedibile.
    """
    rng = random.Random(seme)

    def gioca(mie: list[Mossa], sue: list[Mossa]) -> Mossa:
        if not mie:
            return rng.choice(MOSSE)
        ultima, contro = mie[-1], sue[-1]
        if ultima == contro:
            return rng.choice(MOSSE)
        # L'esito e' dal punto di vista dell'umano: ha vinto se la sua batte.
        ho_vinto = (ultima - contro) % 3 == 1
        return ultima if ho_vinto else vince_contro(contro)

    return gioca


def anti_ultimo(prima: Mossa = Mossa.SASSO) -> Avversario:
    """Gioca cio' che batte l'ultima mossa del robot.

    **Il limite del predittore, misurato invece che immaginato.** Scrivendo il
    file mi aspettavo ~80%: la regola e' deterministica e dipende solo da cio'
    che il robot ha appena fatto, che il robot conosce. Il banco ha detto
    **53%**, e l'attesa era sbagliata, non il codice.

    Il motivo e' un anello. Questo modello guarda **la storia dell'umano**, ma
    le mosse di ``anti_ultimo`` non dipendono dalla storia dell'umano: dipendono
    dal robot. E quando il predittore si arrende tira **a caso**, quindi la
    mossa successiva dell'avversario discende da un dado che nessun Markov
    sull'umano puo' vedere. Robot imprevedibile -> avversario imprevedibile ->
    robot che si arrende -> da capo.

    **Poi e' stato risolto, e non da una modifica pensata per questo caso.**
    Dopo la penalita' sugli errori (``-1``), MIN_OSSERVAZIONI=1 e il modello
    reattivo, il banco misura **98,8%**. Il reattivo condiziona su
    ``(mossa del robot, esito)``, che e' esattamente cio' su cui ``anti_ultimo``
    decide: bastava guardare la variabile giusta.

    Resta qui come **promemoria di metodo**: l'attesa iniziale era 80% per
    intuizione, la misura disse 53%, e la spiegazione che scrissi allora
    (*"limite strutturale"*) era sbagliata. Era un difetto risolvibile che
    avevo dichiarato insuperabile. L'attesa resta bassa di proposito: e' il
    caso in cui mi sono sbagliato due volte.
    """

    def gioca(mie: list[Mossa], sue: list[Mossa]) -> Mossa:
        return vince_contro(sue[-1]) if sue else prima

    return gioca


def sbilanciato(peso: float = 0.6, mossa: Mossa = Mossa.SASSO, seme: int | None = None):
    """Una mossa piu' frequente delle altre, senza nessuna struttura temporale.

    Solo l'ordine 0 puo' coglierlo: non c'e' niente da vedere nelle transizioni.
    Serve a verificare che la gara scelga il modello **giusto**, non il piu'
    sofisticato.
    """
    rng = random.Random(seme)
    altre = [m for m in MOSSE if m != mossa]

    def gioca(mie: list[Mossa], sue: list[Mossa]) -> Mossa:
        return mossa if rng.random() < peso else rng.choice(altre)

    return gioca


def perdente(seme: int | None = None) -> Avversario:
    """Gioca sempre cio' che **perde** contro l'ultima mossa del robot.

    Non e' un avversario realistico: e' un rivelatore. Se il predittore ha il
    segno invertito in ``vince_contro``, contro questo perde in modo vistoso
    invece di sembrare solo mediocre.
    """
    rng = random.Random(seme)

    def gioca(mie: list[Mossa], sue: list[Mossa]) -> Mossa:
        return perde_contro(sue[-1]) if sue else rng.choice(MOSSE)

    return gioca


def lettore(
    seme: int | None = None, quanto: float = 1.0, altrimenti: Avversario | None = None
) -> Avversario:
    """**Legge la previsione sullo schermo** e gioca per batterla.

    E' il difetto aperto del documento, trovato da una persona vera: se il
    robot annuncia "prevedo A", giochera' cio' che batte A; il lettore gioca
    cio' che batte *quello*, cioe' ``A + 2`` in aritmetica modulo 3.

    Riceve **solo l'annuncio** (``attesa``), cioe' cio' che e' sullo schermo:
    la mossa del robot non la vede, altrimenti barerebbe lui.

    ``quanto``: la frazione di mani in cui legge. ``altrimenti``: cosa fa quando
    non legge o non c'e' niente da leggere — a caso se ``None``, altrimenti le
    sue abitudini (per esempio ``win_stay_lose_shift``). La differenza conta:
    una persona vera davanti a uno schermo muto non tira a caso.
    """
    rng = random.Random(seme)

    def gioca(mie: list[Mossa], sue: list[Mossa], annuncio: Mossa | None = None) -> Mossa:
        if annuncio is not None and rng.random() < quanto:
            return vince_contro(vince_contro(annuncio))
        return altrimenti(mie, sue) if altrimenti is not None else rng.choice(MOSSE)

    gioca.legge = True  # type: ignore[attr-defined]
    return gioca


def legge_l_annuncio(avversario: Avversario) -> bool:
    """L'avversario vuole vedere l'annuncio? Solo i lettori lo ricevono."""
    return getattr(avversario, "legge", False)


#: Tutti gli avversari, col seme fissato: il banco gira **uguale ogni volta**.
#: Un banco che cambia risultato a ogni esecuzione non permette di dire se una
#: modifica ha migliorato o peggiorato le cose.
def banco(seme: int = 1234) -> dict[str, Avversario]:
    return {
        "casuale": casuale(seme),
        "costante": costante(),
        "ciclico": ciclico(),
        "ripetitore": ripetitore(),
        "win_stay": win_stay_lose_shift(seme + 1),
        "anti_ultimo": anti_ultimo(),
        "sbilanciato": sbilanciato(seme=seme + 2),
        "perdente": perdente(seme + 3),
    }


#: Cosa il predittore deve ottenere contro ciascuno, come accuratezza.
#: ``(minimo, massimo)``; ``None`` significa "non vincolato da questo lato".
#:
#: Il tetto su ``casuale`` e' la riga piu' importante del file: e' un limite
#: **superiore**, l'unico caso in cui fare troppo bene e' la prova di un baco.
ATTESI: dict[str, tuple[float | None, float | None]] = {
    "casuale": (None, 0.40),
    "costante": (0.90, None),
    "ciclico": (0.90, None),
    "ripetitore": (0.90, None),
    "win_stay": (0.45, None),
    # 53% misurato, non 80% come credevo: vedi il docstring di `anti_ultimo`.
    # La soglia sta poco sopra il caso perche' il limite e' reale e dichiarato.
    "anti_ultimo": (0.40, None),
    "sbilanciato": (0.45, None),
    "perdente": (0.80, None),
}


def esito_umano(umano: Mossa, robot: Mossa) -> Esito:
    """L'esito dal punto di vista dell'**umano**. Serve agli avversari che
    reagiscono al proprio risultato."""
    if umano == robot:
        return Esito.PATTA
    return Esito.VITTORIA if (umano - robot) % 3 == 1 else Esito.SCONFITTA
