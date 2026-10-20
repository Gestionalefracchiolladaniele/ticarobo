"""Il registro: cosa e' successo, e **se quel numero significa qualcosa**.

Una percentuale da sola non e' un risultato. "Ho vinto il 60%" vale tutto o
niente a seconda di quante partite sono: su 12 partite il caso produce 60%
abbastanza spesso da non stupirsi, su 1000 sarebbe straordinario.

Per questo il registro non stampa mai una percentuale nuda. Accanto ci mette
**quante deviazioni standard la separano dal caso**, e a parole se e' prova o
rumore. E' la stessa regola della tabella in CLAUDE.md: un numero che afferma
successo deve portarsi dietro cio' che lo rende verificabile.

La matematica e' una riga. Sotto l'ipotesi "sto tirando a caso", l'accuratezza
osservata su ``n`` partite ha media ``p0 = 1/3`` e deviazione standard::

    sigma = sqrt(p0 * (1 - p0) / n)

Lo scarto in sigma e' ``(osservato - p0) / sigma``. Tre sigma e' la soglia oltre
cui si smette di sospettare la fortuna: sotto l'ipotesi del caso capita circa
una volta su mille.


Le due metriche, e perche' non basta la prima
---------------------------------------------

**Accuratezza**: quante volte il predittore ha indovinato la mossa. Baseline
33,3%. Misura il *modello*.

**Punteggio netto**: ``(vittorie - sconfitte) / partite``. Baseline **zero**.
Misura il *gioco*. Serve perche' a sasso-carta-forbici esistono le patte, e il
tasso di vittoria le comprime: 40% di vittorie puo' voler dire 40-30-30 (bene)
oppure 40-60-0 (malissimo). Il netto distingue i due casi, il tasso no.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from robotica.games.rps.gioco import Esito, Mossa
from robotica.games.rps.predittore import Previsione

#: La baseline: con tre mosse equiprobabili si indovina una volta su tre.
CASO = 1.0 / 3.0

#: Oltre quanti sigma un risultato smette di essere spiegabile con la fortuna.
SIGMA_PROVA = 3.0


@dataclass(frozen=True)
class Partita:
    """Una mano: cosa era stato previsto, cosa e' successo."""

    previsione: Previsione
    umano: Mossa
    esito: Esito

    @property
    def indovinata(self) -> bool:
        """Il predittore aveva **dichiarato** una previsione, ed era giusta.

        Una mano non dichiarata non conta come azzeccata neanche quando la
        mossa giocata vince: il robot non aveva annunciato niente, e il numero
        sul tabellone misura le promesse mantenute, non la fortuna.
        """
        return self.previsione.dichiarata and self.previsione.attesa == self.umano


@dataclass
class Registro:
    """Tutte le partite di una sessione, piu' cio' che se ne deduce."""

    partite: list[Partita] = field(default_factory=list)

    def aggiungi(self, previsione: Previsione, umano: Mossa, esito: Esito) -> None:
        self.partite.append(Partita(previsione, umano, esito))

    # --- conteggi -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.partite)

    @property
    def vittorie(self) -> int:
        return sum(1 for p in self.partite if p.esito is Esito.VITTORIA)

    @property
    def sconfitte(self) -> int:
        return sum(1 for p in self.partite if p.esito is Esito.SCONFITTA)

    @property
    def patte(self) -> int:
        return sum(1 for p in self.partite if p.esito is Esito.PATTA)

    @property
    def rese(self) -> int:
        """Quante volte ha detto di non sapere. Si **conta**, non si nasconde.

        Non e' piu' una resa nel senso di "non gioco": il robot gioca sempre la
        mossa col miglior valore atteso. E' una resa sull'**annuncio** — ha
        giocato senza promettere. Il nome resta per continuita' col tabellone,
        dove "non lo so: 4 volte" e' la riga che dimostra che qualcuno guarda.
        """
        return sum(1 for p in self.partite if not p.previsione.dichiarata)

    @property
    def sbilanciate(self) -> int:
        """Le mani in cui il robot ha **dichiarato** una previsione."""
        return len(self.partite) - self.rese

    @property
    def atteso_medio(self) -> float:
        """Il valore atteso medio delle mosse giocate, fra -1 e +1.

        E' cio' che il predittore **credeva** di guadagnare. Confrontarlo col
        ``netto``, che e' cio' che ha guadagnato davvero, dice se il modello e'
        tarato: se crede +0,4 e ottiene +0,1, sta sopravvalutando quello che sa.
        """
        if not self.partite:
            return 0.0
        return sum(p.previsione.atteso for p in self.partite) / len(self.partite)

    @property
    def indovinate(self) -> int:
        return sum(1 for p in self.partite if p.indovinata)

    # --- metriche -------------------------------------------------------------

    @property
    def accuratezza(self) -> float:
        """Indovinate su **tutte** le partite, rese comprese.

        E' il numero da mostrare in piazza: quante volte, su tutte le mani
        giocate, il robot aveva annunciato la mossa giusta. Una resa non ci
        entra, perche' non annunciava niente.

        **Non si confronta con 1/3.** Con molte rese finisce sotto il caso per
        costruzione, e sembra un disastro quando invece il predittore ha solo
        taciuto. La baseline giusta e' ``caso_atteso``, e il confronto onesto
        e' ``sigma_dal_caso``, che la usa. Misurato: con sessioni da 5 mani si
        arrende nei due terzi dei casi, e l'accuratezza scende al 9% senza che
        il modello sia peggiorato di niente.
        """
        if not self.partite:
            return 0.0
        return self.indovinate / len(self.partite)

    @property
    def accuratezza_sbilanciata(self) -> float:
        """Indovinate sulle sole mani in cui si e' **sbilanciato**.

        E' la grandezza che misura il *modello*, e si confronta con 1/3.
        Da sola sarebbe ingannevole — un predittore che si arrende sempre
        tranne quando e' sicurissimo farebbe il 90% senza saper giocare — per
        questo non sostituisce ``accuratezza``, l'affianca. Le due insieme
        dicono **quanto sa** e **quanto spesso osa**, che sono due difetti
        diversi e vanno visti separati.
        """
        if not self.sbilanciate:
            return 0.0
        return self.indovinate / self.sbilanciate

    @property
    def caso_atteso(self) -> float:
        """Quanto indovinerebbe il caso, **su queste stesse mani**.

        Non 1/3: sulle rese il predittore non afferma niente, quindi quelle
        mani non sono un'occasione mancata di indovinare — non erano una
        previsione. La baseline e' 1/3 sulle sole mani sbilanciate, riportata
        sul totale.

        Confrontare l'accuratezza con 1/3 secco quando ci sono rese e' l'errore
        che ha prodotto "26,8% contro un avversario casuale, 6 sigma sotto il
        caso": un numero che sembrava denunciare un predittore rotto mentre
        denunciava una metrica storta.
        """
        if not self.partite:
            return CASO
        return CASO * self.sbilanciate / len(self.partite)

    @property
    def netto(self) -> float:
        """``(vittorie - sconfitte) / partite``, da -1 a +1. Baseline zero."""
        if not self.partite:
            return 0.0
        return (self.vittorie - self.sconfitte) / len(self.partite)

    @property
    def sigma_dal_caso(self) -> float:
        """Di quante deviazioni standard il predittore supera il caso.

        Il conto si fa **sulle sole mani sbilanciate**: quelle sono le prove
        vere, perche' li' il predittore ha affermato qualcosa e puo' sbagliare.
        Una resa non e' un'estrazione andata male, e' un'estrazione non fatta.

        Includere le rese nel denominatore gonfierebbe il campione con mani in
        cui non c'era niente da indovinare, e farebbe sembrare significativo
        (in negativo) il semplice tacere.

        Negativo significa **peggio del caso**, che contro un avversario che
        reagisce e' possibile davvero e va visto, non nascosto sotto un
        ``abs()``: e' successo, contro ``win_stay`` a sessioni corte.
        """
        n = self.sbilanciate
        if n == 0:
            return 0.0
        sigma = math.sqrt(CASO * (1 - CASO) / n)
        return (self.accuratezza_sbilanciata - CASO) / sigma

    @property
    def e_una_prova(self) -> bool:
        """Il risultato e' distinguibile dalla fortuna?"""
        return self.sigma_dal_caso >= SIGMA_PROVA

    def soglia_per_provare(self) -> float:
        """Quale ``accuratezza_sbilanciata`` servirebbe per essere una prova.

        Utile da mostrare quando la risposta e' no: dice **quanto** manca, e
        quasi sempre la risposta vera e' "non manca accuratezza, mancano
        partite". Su 500 mani sbilanciate la soglia e' 37,7%: in piazza, con
        cento persone da cinque mani, e' un traguardo basso.
        """
        n = self.sbilanciate
        if n == 0:
            return 1.0
        return CASO + SIGMA_PROVA * math.sqrt(CASO * (1 - CASO) / n)

    # --- calibrazione ---------------------------------------------------------

    def calibrazione(self, larghezza: float = 0.1) -> list[tuple[float, float, float, int]]:
        """Quando dice X%, vince davvero X%?

        Raggruppa le previsioni per fascia di confidenza e confronta il
        dichiarato col realizzato. Restituisce, per ogni fascia non vuota:
        ``(inizio, confidenza media dichiarata, accuratezza reale, quante)``.

        E' la metrica piu' in linea con la tesi del progetto. Un predittore
        debole ma calibrato e' onesto; uno forte e spaccone dichiara in piazza
        una sicurezza che non ha, ed e' esattamente il flag che dice "va tutto
        bene" senza esserne la prova.

        Le rese sono escluse: non dichiarano una confidenza su una previsione.
        """
        fasce: dict[int, list[Partita]] = {}
        for p in self.partite:
            if not p.previsione.sicuro:
                continue
            i = int(p.previsione.confidenza / larghezza)
            fasce.setdefault(i, []).append(p)

        out = []
        for i in sorted(fasce):
            gruppo = fasce[i]
            media = sum(p.previsione.confidenza for p in gruppo) / len(gruppo)
            reale = sum(1 for p in gruppo if p.indovinata) / len(gruppo)
            out.append((i * larghezza, media, reale, len(gruppo)))
        return out

    def scarto_calibrazione(self) -> float:
        """Lo scarto medio fra confidenza dichiarata e accuratezza reale.

        Pesato sul numero di previsioni per fascia. La soglia del documento e'
        **meno di 10 punti**, cioe' 0.10 qui.
        """
        fasce = self.calibrazione()
        if not fasce:
            return 0.0
        tot = sum(n for _, _, _, n in fasce)
        return sum(abs(media - reale) * n for _, media, reale, n in fasce) / tot

    # --- il tabellone ---------------------------------------------------------

    def tabellone(self) -> str:
        """Il riassunto da stampare: i numeri **e** se sono una prova."""
        n = len(self.partite)
        if n == 0:
            return "nessuna partita giocata."

        righe = [
            f"partite: {n}     ROBOT {self.vittorie}   TU {self.sconfitte}   patte {self.patte}",
            f"punteggio netto: {self.netto:+.3f}   (il caso da' 0.000)"
            f"   <- quanto ha vinto davvero",
            f"  si aspettava {self.atteso_medio:+.3f}"
            + ("   (si e' sopravvalutato)" if self.atteso_medio > self.netto + 0.15 else ""),
            "",
            f"previsioni azzeccate: {self.indovinate}/{n}  ({self.accuratezza:.1%} di tutte)",
            f"  fra quelle dichiarate: {self.indovinate}/{self.sbilanciate} "
            f"({self.accuratezza_sbilanciata:.1%})   <- questo si confronta col 33.3%",
            f"ha giocato senza promettere: {self.rese} volte su {n}",
        ]

        s = self.sigma_dal_caso
        if self.sbilanciate == 0:
            righe.append("scarto dal caso: non si e' mai sbilanciato, niente da misurare")
        elif self.e_una_prova:
            righe.append(f"scarto dal caso: {s:+.1f} sigma -> e' una prova, non fortuna")
        else:
            soglia = self.soglia_per_provare()
            righe.append(
                f"scarto dal caso: {s:+.1f} sigma -> NON distinguibile dal caso "
                f"(servirebbe {soglia:.1%} su {self.sbilanciate} sbilanciate, o piu' partite)"
            )

        fasce = self.calibrazione()
        if fasce:
            righe.append("")
            righe.append("calibrazione — quando dice X%, vince davvero X%?")
            for inizio, media, reale, quante in fasce:
                segno = "ok" if abs(media - reale) < 0.10 else "SCARTO"
                righe.append(
                    f"  {inizio:.0%}-{inizio + 0.1:.0%}: dichiarato {media:.0%}, "
                    f"reale {reale:.0%}  ({quante} volte)  {segno}"
                )
            righe.append(f"  scarto medio: {self.scarto_calibrazione():.1%} (soglia: 10%)")

        return "\n".join(righe)
