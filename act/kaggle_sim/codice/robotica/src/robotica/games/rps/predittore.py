"""Il predittore: **vincere il piu' possibile**, e dire onestamente quanto si sa.

La tesi, in una riga: gli umani non sanno essere casuali. Chiedi a qualcuno una
sequenza "a caso" e produrra' troppi cambi, troppe poche ripetizioni, e reagira'
a cosa e' appena successo. Un modello che conta le transizioni lo sfrutta.


Due decisioni separate, che prima erano fuse
--------------------------------------------

**Cosa gioco** e **cosa dichiaro** sono indipendenti, e tenerle legate e' stato
il difetto peggiore della prima versione.

Prima il predittore, sotto una certa confidenza, si arrendeva e tirava **a
caso**. Sembra prudente ed e' uno spreco: se prevedo sasso al 37%, giocare
carta ha valore atteso positivo, mentre tirare a caso ha valore atteso **zero**.
Si buttava via un vantaggio reale per non dover dichiarare una confidenza bassa.

Adesso:

- si gioca **sempre** la mossa col miglior valore atteso
- si **dichiara** la previsione solo quando e' onesto farlo

Il robot puo' benissimo dire *"non sono sicuro"* e giocare lo stesso. Le due
cose non si condizionano.

Il valore atteso di giocare ``m`` contro una distribuzione ``p`` e'::

    E[m] = P(m batte l'umano) - P(l'umano batte m)

Con tre mosse e' il conto di due termini, e massimizzarlo e' cio' che il tuo
obiettivo chiede: **vincere piu' partite**, non fare la previsione piu' bella.


I modelli in gara
-----------------

Un Markov di ordine 2 ha **9 contesti**. Per dire qualcosa su un contesto
servono alcune osservazioni di *quel* contesto, quindi nelle prime mani la
maggior parte e' vuota. In piazza la partita e' **dieci mani**: un predittore
che si accende alla mano venti, li' non si accende mai.

Allora se ne tengono quattro insieme, e ogni mano decide quello che finora ha
indovinato di piu':

===========  ==========================================  ========================
modello      cosa guarda                                 quando e' utile
===========  ==========================================  ========================
ordine 0     quante volte ha giocato ogni mossa          **subito**, dalla mano 3
ordine 1     cosa fa dopo *una* sua data mossa           da ~8 mani
ordine 2     cosa fa dopo *due* sue date mosse           da ~20 mani
**reattivo** cosa fa dopo una **mia** mossa e un esito   da ~8 mani
===========  ==========================================  ========================

**Il reattivo e' la correzione che conta**, e nasce da una misura. I primi tre
condizionano sulla storia *dell'umano*. Ma il bias umano meglio documentato —
*win-stay, lose-shift*: tieni la mossa che ha vinto, cambia quando perdi —
risponde a cosa ha fatto **il robot** e a com'e' finita. Nessuno dei tre poteva
vederlo, e infatti il banco misurava netto **-0,283** contro quel bias: il
predittore non era debole, guardava la variabile sbagliata.

E' l'errore di frame del progetto, nella sua versione temporale: il nome della
variabile deve dire *rispetto a cosa* e' misurata.


Le meta-strategie, e perche' ne bastano due
-------------------------------------------

Iocaine Powder, che vinse il primo torneo internazionale, tiene **sei** livelli
di second-guessing. Qui ne servono due, e il motivo e' specifico di questo
progetto: **la previsione e' sullo schermo**. La gente non gioca contro il
robot, gioca contro la previsione. Se annuncio "prevedo sasso", chi capisce che
allora giochero' carta tira forbici e mi batte — e la trasparenza, che e' il
valore del progetto, diventa la sua vulnerabilita'.

Quindi:

- **livello 0** — prevedo la tua mossa
- **livello 1** — prevedo che tu stia rispondendo alla mia previsione

Sei livelli sarebbero sovradimensionati: con dieci mani a persona i livelli alti
non avrebbero mai abbastanza dati, e ogni modello in piu' diluisce la gara.


La regola che rende la misura onesta
------------------------------------

**Il predittore non vede mai la mossa che sta prevedendo.** Non per disciplina
di chi scrive: ``prevedi()`` non prende la mossa, e la mossa entra solo da
``osserva()``, che si chiama dopo.

E' l'errore piu' facile da fare qui, ed e' invisibile, perche' produce un
predittore bravissimo che in piazza fa 33%. Il banco lo prende comunque: contro
un avversario davvero casuale nessun predittore puo' superare 1/3, quindi un
numero piu' alto **e' la firma di un baco**, non un successo.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from robotica.games.rps.gioco import MOSSE, Esito, Mossa, batte, esito

# Quanto pesano i successi vecchi rispetto a quelli nuovi, per mano passata.
# 0.9 vuol dire che un successo di 10 mani fa vale circa un terzo di uno di
# adesso: abbastanza reattivo da seguire un cambio di strategia, abbastanza
# lento da non inseguire il rumore di una mano sola.
DECADIMENTO = 0.9

# Quante osservazioni servono in un contesto prima di fidarsi di quel contesto.
#
# **Misurato, non scelto**, e il valore misurato ha smentito quello scelto.
#
# Era 3, col ragionamento "con 2 osservazioni '2 su 2' e' un evento che il caso
# produce una volta su nove, quindi non significa niente". Corretto in astratto,
# **sbagliato per dieci mani a persona**: un modello con 9 contesti ne vede
# ciascuno una volta sola, non raggiunge mai la soglia, e resta muto per sempre.
#
# Il banco, a 10 mani per persona e 300 persone (netto):
#
#     MIN_OSS   win_stay   sbilanciato   ciclico   casuale   cieco
#           1     -0.028        +0.246    +0.405    +0.006     10%
#           2     -0.177        +0.233    -0.299    +0.009     20%
#           3     -0.225        +0.222    -0.199    +0.010     30%
#           4     -0.205        +0.202    -0.302    +0.003     40%
#
# La riga che decide e' ``ciclico``: un avversario **deterministico** (sasso,
# carta, forbici in ciclo) contro cui il predittore *perdeva* -0.199, perche'
# non arrivava mai a tre osservazioni per contesto. Con 1 diventa +0.405.
#
# ``casuale`` resta a zero a ogni soglia: abbassarla non fa inventare pattern
# dove non ce ne sono, che era il rischio da escludere.
#
# La lezione, e vale oltre questo file: **con sessioni corte, aspettare la
# certezza statistica significa non parlare mai.** Meglio un'informazione debole
# subito che una forte mai — a patto di dichiarare che e' debole, e la
# confidenza lo fa.
MIN_OSSERVAZIONI = 1

# Quanto si "sconta" la certezza di un contesto visto poche volte.
#
# Abbassare MIN_OSSERVAZIONI a 1 ha creato un difetto che le simulazioni non
# avevano visto e la prima partita vera ha trovato subito: **il predittore
# annunciava "confidenza 100%" dopo aver visto un contesto due volte**, e poi
# azzeccava il 57%. In piazza la confidenza e' pubblica, quindi un 100% che
# vale 57% non e' ottimismo, e' affermare senza sapere.
#
# Lo smoothing di Laplace conta ogni mossa come se fosse gia' stata vista ALPHA
# volte. Con 2 osservazioni concordi: (2 + 0.5) / (2 + 1.5) = **71%** invece di
# 100%. Con 20 osservazioni l'effetto quasi sparisce, com'e' giusto.
#
# **Non cambia cosa si gioca**, perche' non cambia l'ordine fra le tre mosse:
# cambia solo cio' che il robot dichiara di sapere. E' una correzione di
# onesta', non di strategia — e il valore atteso, che dipende dalle
# probabilita', diventa piu' realistico di conseguenza.
ALPHA = 0.5

# Sotto questa confidenza il predittore **dichiara** di non sapere.
#
# Non e' piu' una soglia di gioco — la mossa si sceglie sempre per valore
# atteso, anche quando la confidenza e' bassa. Questa soglia decide soltanto
# cosa compare sullo schermo, ed e' una questione di onesta', non di strategia.
#
# 0.40 e' poco sopra il caso (0.333): sotto, annunciare una previsione sarebbe
# affermare senza sapere.
SOGLIA_DICHIARAZIONE = 0.40

#: Il nome dei modelli che partono dalla memoria di piazza (vedi ``memoria.py``).
PREFISSO_PIAZZA = "piazza: "


# Chi legge lo schermo — vedi ``Predittore.stima_lettura``.
#
# Quanto spesso il predittore indovina chi **non** legge. Misurato contro
# win_stay a 10 mani: 52,6%. Si usa 0,5.
PRECISIONE_SENZA_LETTURA = 0.5

# L'a priori: prima di vedere qualcosa, quanto e' credibile ogni frazione di
# lettura. **La maggior parte della gente non legge**: il 70% della massa su
# q = 0, il resto diviso fra 0,1 e 1. E' cio' che impedisce a uno scarto 2
# isolato di bastare.
_IPOTESI_LETTURA: tuple[tuple[float, float], ...] = ((0.0, 0.7),) + tuple(
    (q / 10, 0.3 / 10) for q in range(1, 11)
)


def _argmax(valori: list[float]) -> int:
    """L'indice del massimo. Esplicito perche' ricorre e va letto uguale ovunque."""
    return max(range(len(valori)), key=lambda i: valori[i])


def valore_atteso(mossa: Mossa, dist: list[float]) -> float:
    """Quanto rende giocare ``mossa`` contro la distribuzione ``dist``.

    ``P(vinco) - P(perdo)``, fra -1 e +1. Le patte valgono zero, ed e' giusto:
    una patta non fa punti a nessuno.
    """
    vinco = sum(dist[m] for m in MOSSE if batte(mossa, m))
    perdo = sum(dist[m] for m in MOSSE if batte(m, mossa))
    return vinco - perdo


def mossa_migliore(dist: list[float]) -> tuple[Mossa, float]:
    """La mossa col valore atteso piu' alto, e quel valore.

    Contro una distribuzione piatta tutte e tre valgono zero e la scelta e'
    arbitraria: si prende la prima, e chi chiama aggiunge il caso se serve.
    """
    valori = [valore_atteso(m, dist) for m in MOSSE]
    i = _argmax(valori)
    return MOSSE[i], valori[i]


@dataclass
class _Modello:
    """Base comune: conta cosa segue ogni contesto, e tiene il proprio credito.

    Le sottoclassi cambiano **solo** come si forma il contesto. Il resto —
    conteggi, soglia di competenza, credito scontato — e' identico, e tenerlo
    in un posto solo evita che due modelli divergano in silenzio su cosa
    significhi "non lo so".
    """

    nome: str
    conteggi: dict[tuple, list[float]] = field(default_factory=dict)
    #: Quanto ha indovinato finora, scontato nel tempo. E' il punteggio in gara.
    credito: float = 0.0

    def contesto(self, st: Storia) -> tuple | None:
        """La chiave sotto cui cercare, o ``None`` se la storia non basta."""
        raise NotImplementedError

    def distribuzione(self, st: Storia) -> list[float] | None:
        """Cosa ci si aspetta adesso, come probabilita'.

        ``None`` significa **non lo so**: contesto mai visto, o visto troppo
        poco. Dirlo e' il punto; riempire con una distribuzione piatta
        nasconderebbe l'ignoranza dentro un numero dall'aria innocua.
        """
        ctx = self.contesto(st)
        if ctx is None:
            return None
        conta = self.conteggi.get(ctx)
        if conta is None:
            return None
        tot = sum(conta)
        if tot < MIN_OSSERVAZIONI:
            return None
        # Laplace: nessun contesto vale il 100% solo perche' e' stato visto due
        # volte. Vedi ALPHA — corregge la confidenza dichiarata, non la scelta.
        den = tot + 3 * ALPHA
        return [(c + ALPHA) / den for c in conta]

    def osserva(self, st: Storia, mossa: Mossa) -> None:
        """Registra che in questo contesto e' arrivata ``mossa``.

        ``st`` e' lo stato **prima** della mossa: il contesto si forma li'.
        """
        ctx = self.contesto(st)
        if ctx is not None:
            self.conteggi.setdefault(ctx, [0.0, 0.0, 0.0])[mossa] += 1.0

    def aggiorna_credito(self, previsto: Mossa | None, reale: Mossa) -> None:
        """Sconta il credito passato, **premia se indovina e punisce se sbaglia**.

        La penalita' non c'era nella prima versione, ed era il difetto che
        rendeva inutile tutto il resto. Misurato: il modello ``ordine 0``
        azzeccava il **21,4%** — sotto il caso — e veniva scelto **1312 volte su
        3000**, mentre il ``reattivo``, che azzeccava l'**84,1%**, veniva scelto
        **52 volte**. La gara premiava sistematicamente il modello peggiore.

        Il motivo: senza penalita' il credito misura *quanto spesso un modello
        parla*, non *quanto ha ragione*. Un modello sempre disponibile e quasi
        sempre sbagliato accumula comunque, e uno che parla di rado ma bene non
        recupera mai il distacco.

        Con ``-1`` sugli errori, un modello che sbaglia quattro volte su cinque
        finisce **sotto zero** e lascia il posto. E' la "decayed scoring" della
        letteratura sui bot RPS (Egnor, *Iocaine Powder*, e i vincitori di
        rpscontest): ``+1`` se indovina, ``-1`` se sbaglia, tutto moltiplicato
        per ~0,9 a ogni mano.

        Chi **non aveva una previsione** non guadagna e non perde: subisce solo
        lo sconto. Tacere costa, ma meno che sbagliare — ed e' giusto cosi',
        altrimenti conviene sempre parlare a vanvera.
        """
        self.credito *= DECADIMENTO
        if previsto is None:
            return
        self.credito += 1.0 if previsto == reale else -1.0

    def fonde(self, altro: _Modello, peso: float = 1.0) -> None:
        """Assorbe i conteggi di un altro modello dello stesso tipo.

        Serve alla memoria di piazza: quando uno sfidante se ne va, cio' che ha
        insegnato confluisce nel modello di popolazione. Il credito **non** si
        fonde: e' una misura di "chi sta andando bene adesso", e adesso e' una
        cosa locale alla partita in corso.
        """
        for ctx, conta in altro.conteggi.items():
            mio = self.conteggi.setdefault(ctx, [0.0, 0.0, 0.0])
            for i in range(3):
                mio[i] += conta[i] * peso


@dataclass
class _Markov(_Modello):
    """Ordine fisso sulla storia **dell'umano**: cosa fa dopo le sue mosse."""

    ordine: int = 0

    def contesto(self, st: Storia) -> tuple | None:
        if self.ordine == 0:
            return ()
        if len(st.umano) < self.ordine:
            return None
        return tuple(st.umano[-self.ordine :])


@dataclass
class _Reattivo(_Modello):
    """Cosa fa l'umano dopo una **mia** mossa e un dato esito.

    Il modello del bias *win-stay, lose-shift*. Il contesto e' la coppia
    ``(mia ultima mossa, esito per l'umano)``: 9 combinazioni, come l'ordine 2,
    ma sono le 9 che descrivono una **reazione** invece di un'abitudine.
    """

    def contesto(self, st: Storia) -> tuple | None:
        if not st.robot:
            return None
        # L'esito visto dall'umano: e' lui che decide se tenere o cambiare.
        return (st.robot[-1], esito(st.umano[-1], st.robot[-1]))


@dataclass
class _AntiPrevisione(_Modello):
    """Cosa fa l'umano dopo aver **visto** una certa previsione.

    Il secondo livello. In piazza la previsione e' sullo schermo prima che la
    persona tiri, quindi una parte dei giocatori gioca contro *quella*, non
    contro il robot. Il contesto e' la mossa annunciata, piu' se era dichiarata
    o no: annunciare "prevedo sasso" e annunciare "non so" producono reazioni
    diverse, e confonderle sprecherebbe meta' del segnale.
    """

    def contesto(self, st: Storia) -> tuple | None:
        if not st.annunci:
            return None
        return (st.annunci[-1],)


@dataclass
class Storia:
    """Cio' che e' successo, dal punto di vista di chi deve prevedere.

    Un oggetto solo invece di tre liste sparse, perche' ogni modello ne guarda
    una parte diversa e passarle separate e' come si sbaglia un frame: prima o
    poi qualcuno allinea male gli indici.

    **Le tre liste sono allineate**: alla mano ``i``, il robot ha annunciato
    ``annunci[i]``, ha giocato ``robot[i]``, l'umano ha risposto ``umano[i]``.
    """

    umano: list[Mossa] = field(default_factory=list)
    robot: list[Mossa] = field(default_factory=list)
    #: La mossa **annunciata** come previsione, o ``None`` se il robot ha detto
    #: di non sapere. Non coincide con ``robot``: si annuncia cio' che si
    #: prevede dell'umano, si gioca cio' che conviene.
    annunci: list[Mossa | None] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.umano)

    def azzera(self) -> None:
        self.umano.clear()
        self.robot.clear()
        self.annunci.clear()


@dataclass(frozen=True)
class Previsione:
    """Cosa il robot gioca, e cosa dichiara **prima** che la persona tiri.

    La distinzione fra i due e' il cuore di questa versione. ``mossa`` c'e'
    sempre: il robot gioca comunque la cosa migliore che sa. ``attesa`` e'
    l'annuncio, e puo' mancare quando dichiararlo non sarebbe onesto.
    """

    #: Cosa gioca il robot. **Sempre presente**, sempre la mossa col miglior
    #: valore atteso: non arrendersi e' cio' che fa vincere piu' partite.
    mossa: Mossa
    #: La mossa annunciata come previsione. ``None`` = "non sono sicuro",
    #: detto a voce alta. Non impedisce di giocare.
    attesa: Mossa | None
    #: Quanto e' probabile la mossa piu' probabile dell'umano. A vuoto 1/3.
    confidenza: float
    #: Il valore atteso della mossa giocata, fra -1 e +1. **E' la metrica del
    #: tuo obiettivo**: quanto ci si aspetta di guadagnare da questa mano.
    atteso: float
    #: Il motivo, in italiano, derivato **dagli stessi conteggi** che hanno
    #: deciso. Non una ricostruzione a parte: sarebbe una spiegazione
    #: plausibile e non vera, il difetto contro cui e' costruito il progetto.
    perche: str
    #: Quale modello ha deciso, per il registro.
    modello: str

    @property
    def dichiarata(self) -> bool:
        """Il robot ha annunciato una previsione, o ha detto di non sapere?"""
        return self.attesa is not None

    # Nome storico, tenuto perche' il registro ragiona in termini di
    # "si e' sbilanciato": adesso vuol dire "ha dichiarato", non "ha giocato".
    sicuro = dichiarata


class Predittore:
    """Prevede la prossima mossa di uno sfidante, e sceglie cosa giocare.

    **Tre memorie, tre durate diverse** (vedi ``memoria.py`` per come si
    compongono in piazza):

    - la **partita**: dieci mani, e finisce
    - lo **sfidante**: finche' non dice di essere un altro
    - la **piazza**: per sempre, su tutte le persone

    Questa classe e' il livello *sfidante*. Riceve opzionalmente un fondo di
    partenza dalla piazza, che riempie i conteggi prima della prima mano:
    e' cio' che permette di non essere spenti alla mano 1.
    """

    def __init__(
        self,
        seme: int | None = None,
        fondo: dict[str, dict[tuple, list[float]]] | None = None,
        pubblica: bool = True,
    ) -> None:
        #: La previsione e' sullo schermo dello sfidante? Se no, nessuno puo'
        #: leggerla, e la stima di lettura resta spenta: costerebbe falsi
        #: positivi senza proteggere da niente. Decisione del 2026-09-24: in
        #: piazza la previsione la vede solo il creatore.
        self.pubblica = pubblica
        self.storia = Storia()
        #: Cio' che si impara **da questo sfidante**, e solo da lui.
        self._propri: list[_Modello] = [
            _Markov(nome="ordine 0", ordine=0),
            _Markov(nome="ordine 1", ordine=1),
            _Markov(nome="ordine 2", ordine=2),
            _Reattivo(nome="reattivo"),
            _AntiPrevisione(nome="anti-previsione"),
        ]
        #: **La piazza non si mescola: gareggia.** Stessi modelli, partiti dal
        #: fondo della popolazione. Mescolare il fondo nei modelli propri
        #: aiutava chi somiglia alla gente e affossava chi no (misurato: un
        #: avversario ciclico in una piazza di win_stay da +0,50 a ~0). In gara,
        #: se lo sfidante e' diverso questi sbagliano, prendono -1 e perdono il
        #: posto in poche mani. Mixture of experts: la fiducia la guadagna chi
        #: ha ragione (e' la meta-strategia di Iocaine Powder).
        #:
        #: Stanno **dopo** i propri: a parita' di credito (all'inizio, tutti a
        #: zero) vince chi e' piu' avanti, e alla mano 1 solo loro hanno dati.
        self._piazza: list[_Modello] = [
            _Markov(nome=f"{PREFISSO_PIAZZA}ordine 0", ordine=0),
            _Markov(nome=f"{PREFISSO_PIAZZA}ordine 1", ordine=1),
            _Markov(nome=f"{PREFISSO_PIAZZA}ordine 2", ordine=2),
            _Reattivo(nome=f"{PREFISSO_PIAZZA}reattivo"),
        ]
        self._modelli: list[_Modello] = self._propri + self._piazza
        self._rng = random.Random(seme)
        #: L'ultima previsione di ciascun modello, per il credito.
        self._ultime: dict[str, Mossa | None] = {}
        #: Dopo ogni annuncio A, lo scarto ``(mossa dell'umano - A) % 3``.
        #: 0 = previsione giusta; **2 = ha giocato cio' che batte la mia
        #: risposta**, la firma di chi legge. Per sfidante, non per piazza:
        #: leggere lo schermo e' un comportamento, non un bias della specie.
        self.scarti = [0, 0, 0]
        if fondo:
            self.innesta(fondo)

    # --- previsione -----------------------------------------------------------

    def prevedi(self) -> Previsione:
        """Cosa gioca il robot adesso, e cosa dichiara.

        **Non prende argomenti, ed e' voluto**: se la mossa corrente potesse
        entrare qui, prima o poi entrerebbe.
        """
        migliore = self._scegli_modello()

        # Si registra cosa ciascun modello avrebbe previsto: serve a dare il
        # credito quando la mossa vera arrivera'. Va fatto prima di qualsiasi
        # ritorno, altrimenti la gara non si aggiorna nelle mani in cui il
        # predittore non sa nulla.
        for m in self._modelli:
            d = m.distribuzione(self.storia)
            self._ultime[m.nome] = None if d is None else MOSSE[_argmax(d)]

        if migliore is None:
            return Previsione(
                mossa=self._rng.choice(MOSSE),
                attesa=None,
                confidenza=1 / 3,
                atteso=0.0,
                perche="prima mano: non ho ancora niente su cui contare",
                modello="nessuno",
            )

        modello, dist = migliore
        attesa = MOSSE[_argmax(dist)]

        # **Cosa succede se lo annuncio.** Con probabilita' r lo sfidante legge
        # e gioca attesa + 2; altrimenti fa cio' che dice il modello. La
        # miscela e' la previsione vera *di una mano annunciata*.
        r = self.stima_lettura() if self.pubblica else 0.0
        mix = [(1 - r) * p for p in dist]
        mix[(attesa + 2) % 3] += r
        confidenza = mix[attesa]

        # Si annuncia solo cio' in cui si crede: A dev'essere ancora la mossa
        # piu' probabile **sapendo di annunciarla**, e abbastanza probabile.
        # Contro chi legge sempre non lo e' mai: nessuna previsione pubblica
        # puo' essere vera contro chi la legge (A = A + 2 non ha soluzione).
        dichiara = confidenza >= SOGLIA_DICHIARAZIONE and confidenza >= max(mix)

        # Si gioca contro cio' che succedera' davvero: la miscela se si
        # annuncia, il modello puro se si tace (non c'e' niente da leggere).
        gioco, atteso = mossa_migliore(mix if dichiara else dist)

        # Se la distribuzione e' piatta il valore atteso e' zero su tutte: si
        # tira a caso davvero, perche' fingere una preferenza sarebbe esporre
        # un pattern a chi ci sta studiando, senza guadagnarci niente.
        if atteso <= 0.0:
            gioco = self._rng.choice(MOSSE)

        if dichiara:
            perche = self._spiega(modello, attesa, dist[attesa])
        elif dist[attesa] >= SOGLIA_DICHIARAZIONE:
            # Taceva per la lettura, non per ignoranza: lo si dice.
            lette, annunci = self.scarti[2], sum(self.scarti)
            perche = (
                f"mi stai leggendo: {lette} volte su {annunci} hai giocato "
                "contro il mio annuncio. Stavolta non lo dico, ma gioco"
            )
            confidenza = dist[attesa]
        else:
            confidenza = dist[attesa]
            perche = f"il massimo che vedo e' {confidenza:.0%} su {attesa}: gioco ma non prometto"

        return Previsione(
            mossa=gioco,
            attesa=attesa if dichiara else None,
            confidenza=confidenza,
            atteso=atteso,
            perche=perche,
            modello=modello.nome,
        )

    def stima_lettura(self) -> float:
        """Quanto spesso lo sfidante gioca **contro il mio annuncio**, fra 0 e 1.

        **Una stima bayesiana, perche' i dati sono pochissimi.** La prima
        versione contava l'eccesso di scarti 2 sugli 1, e su dieci mani
        scambiava il rumore per lettura: contro ``win_stay``, che non legge, il
        netto crollava da +0,161 a +0,003. Quattro previsioni sbagliate che
        cadono 3 a 1 sono un lancio di monete, non un indizio.

        Qui si confrontano le ipotesi "legge una frazione q delle mani", per q
        da 0 a 1. Ognuna predice gli scarti:

        - scarto 2 (ha battuto la mia risposta): ``q + (1 - q)(1 - a)/2``
        - scarto 0 (previsione giusta): ``(1 - q) a``
        - scarto 1: ``(1 - q)(1 - a)/2``

        con ``a`` quanto spesso indovino chi non legge. Si pesa ogni ipotesi per
        quanto spiega cio' che si e' visto (la verosimiglianza) e per quanto
        era credibile prima (l'a priori: la maggior parte della gente **non**
        legge). Il risultato e' la media di q a posteriori.
        """
        s0, s1, s2 = self.scarti
        if s0 + s1 + s2 == 0:
            return 0.0
        a = PRECISIONE_SENZA_LETTURA

        # **In logaritmi**: dopo centinaia di mani 0,5**900 non sta in un float
        # e diventa 0, e tutte le ipotesi con lui (misurato: ZeroDivisionError
        # nel banco da 1000). I logaritmi si sommano invece di moltiplicarsi, e
        # si torna indietro sottraendo il massimo (log-sum-exp).
        def log(x: float, n: int) -> float:
            if n == 0:
                return 0.0
            return n * math.log(x) if x > 0 else -math.inf

        logw = []
        for q, prior in _IPOTESI_LETTURA:
            p2 = q + (1 - q) * (1 - a) / 2
            p1 = (1 - q) * (1 - a) / 2
            p0 = (1 - q) * a
            logw.append(math.log(prior) + log(p0, s0) + log(p1, s1) + log(p2, s2))
        top = max(logw)
        w = [math.exp(x - top) for x in logw]
        return sum(q * wi for (q, _), wi in zip(_IPOTESI_LETTURA, w, strict=True)) / sum(w)

    def _scegli_modello(self) -> tuple[_Modello, list[float]] | None:
        """Il modello con piu' credito, fra quelli che hanno qualcosa da dire.

        A parita' di credito vince chi sta **piu' avanti nella lista**, che e'
        ordinata dal piu' semplice al piu' specifico: a parita' di risultati, il
        modello che guarda piu' contesto e' il piu' informativo. All'inizio i
        crediti sono tutti zero e solo l'ordine 0 ha dati, quindi parte quello
        senza bisogno di un caso speciale.
        """
        candidati = [
            (i, m, d)
            for i, m in enumerate(self._modelli)
            if (d := m.distribuzione(self.storia)) is not None
        ]
        if not candidati:
            return None
        i, m, d = max(candidati, key=lambda c: (c[1].credito, c[0]))
        return m, d

    def _spiega(self, modello: _Modello, attesa: Mossa, confidenza: float) -> str:
        """Il motivo in italiano, **dai conteggi che hanno deciso**.

        Ricostruire la frase da altri numeri darebbe una spiegazione che suona
        bene e non corrisponde. Qui contesto e totale vengono dallo stesso
        dizionario che ha prodotto la distribuzione.
        """
        ctx = modello.contesto(self.storia)
        assert ctx is not None  # _scegli_modello ha gia' verificato
        tot = int(sum(modello.conteggi[ctx]))
        q = f"{confidenza:.0%}"

        # Un modello di piazza parla della **gente**, non di te: i suoi conteggi
        # vengono in gran parte da altre persone, e sono osservazioni ridotte
        # (``memoria.FORZA``), quindi niente "N volte".
        if modello.nome.startswith(PREFISSO_PIAZZA):
            if isinstance(modello, _Reattivo):
                mia, es = ctx
                come = {Esito.VITTORIA: "vince", Esito.SCONFITTA: "perde", Esito.PATTA: "pareggia"}
                return (f"chi {come[es]} contro {mia}, di solito poi fa {attesa} "
                        f"(la gente che e' passata di qui: {q})")
            assert isinstance(modello, _Markov)
            if modello.ordine == 0:
                return f"la gente che e' passata di qui gioca {attesa} nel {q} dei casi"
            seq = " e ".join(str(m) for m in ctx)
            return f"dopo {seq}, di solito la gente fa {attesa} ({q})"

        if isinstance(modello, _Reattivo):
            mia, es = ctx
            come = {
                Esito.VITTORIA: "dopo che hai vinto",
                Esito.SCONFITTA: "dopo che hai perso",
                Esito.PATTA: "dopo una patta",
            }[es]
            return f"{come} contro {mia}, fai {attesa} nel {q} dei casi ({tot} volte)"

        if isinstance(modello, _AntiPrevisione):
            annuncio = ctx[0]
            testa = "quando non prometto niente" if annuncio is None else f"quando dico {annuncio}"
            return f"{testa}, rispondi {attesa} nel {q} dei casi ({tot} volte)"

        assert isinstance(modello, _Markov)
        if modello.ordine == 0:
            return f"su {tot} mosse hai fatto {attesa} nel {q} dei casi"
        seq = " e ".join(str(m) for m in ctx)
        return f"dopo {seq} hai fatto {attesa} nel {q} dei casi ({tot} volte finora)"

    # --- apprendimento --------------------------------------------------------

    def osserva(self, mossa: Mossa, previsione: Previsione) -> None:
        """La mossa vera e' arrivata: si assegna il credito e si impara.

        L'ordine conta. Prima il credito, che confronta le previsioni fatte **su
        questa storia** con cio' che e' successo; poi i conteggi, ancora sulla
        storia vecchia, perche' il contesto e' quello **prima** della mossa;
        solo alla fine la storia cresce. Invertire vorrebbe dire contare una
        mossa come contesto di se' stessa.

        Serve la ``previsione`` e non solo la mossa: il modello reattivo ha
        bisogno di cosa ha giocato il robot, e l'anti-previsione di cosa ha
        annunciato.
        """
        for m in self._modelli:
            m.aggiorna_credito(self._ultime.get(m.nome), mossa)
        # Solo i modelli propri imparano dallo sfidante. **Quelli di piazza
        # restano pura popolazione**: se imparassero anche da lui, a piazza
        # vuota sarebbero copie dei propri, vincerebbero la parita' e il robot
        # direbbe "la gente fa carta" parlando di una persona sola (trovato
        # leggendo l'output: una spiegazione plausibile e falsa).
        for m in self._propri:
            m.osserva(self.storia, mossa)
        if previsione.attesa is not None:
            self.scarti[(mossa - previsione.attesa) % 3] += 1
        self.storia.umano.append(mossa)
        self.storia.robot.append(previsione.mossa)
        self.storia.annunci.append(previsione.attesa)
        self._ultime.clear()

    # --- memoria --------------------------------------------------------------

    def raccolto(self) -> dict[str, dict[tuple, list[float]]]:
        """Cio' che **questo** sfidante ha insegnato, da versare nella piazza.

        Solo i modelli propri, che il fondo non l'hanno mai visto: il doppio
        conteggio (la piazza che si rimangia se stessa a ogni persona) e'
        impossibile per costruzione, non evitato con una sottrazione.
        """
        return {m.nome: {k: list(v) for k, v in m.conteggi.items()} for m in self._propri}

    def innesta(self, fondo: dict[str, dict[tuple, list[float]]], peso: float = 1.0) -> None:
        """Riempie i modelli **di piazza** con cio' che la popolazione sa.

        ``fondo`` ha le chiavi dei modelli propri ("ordine 0", "reattivo"...):
        e' il formato di ``raccolto`` e di ``memoria.Piazza``.
        """
        for m in self._piazza:
            nome = m.nome.removeprefix(PREFISSO_PIAZZA)
            for ctx, conta in fondo.get(nome, {}).items():
                mio = m.conteggi.setdefault(ctx, [0.0, 0.0, 0.0])
                for i in range(3):
                    mio[i] += conta[i] * peso

    def azzera(self) -> None:
        """Sfidante nuovo: si ricomincia da zero, **piazza compresa**.

        Chi vuole ripartire dalla piazza la re-innesta (``sessione.Tavolo``
        lo fa): cosi' il fondo e' sempre quello aggiornato, con dentro anche
        lo sfidante appena andato via.
        """
        self.storia.azzera()
        self._ultime.clear()
        self.scarti = [0, 0, 0]
        for m in self._modelli:
            m.conteggi.clear()
            m.credito = 0.0
