"""Giocare contro il predittore da terminale, e far girare il banco di prova.

Due comandi, due domande diverse::

    uv run python -m robotica.games.rps.cli gioca    # batte TE?
    uv run python -m robotica.games.rps.cli banco    # il codice e' giusto?

``banco`` e' quello che dice se il codice e' giusto, perche' gli avversari
finti hanno comportamento noto. ``gioca`` da' il numero vero ma su poche mani,
e poche mani non distinguono un predittore dalla fortuna: per questo il
tabellone stampa sempre lo scarto dal caso accanto alla percentuale, e dice a
parole quando non e' una prova.


La partita e' di dieci mani, e il motivo e' misurato
----------------------------------------------------

Non e' un numero tondo scelto a caso. Il banco, contro il bias umano
*win-stay, lose-shift*, da' netto **-0,089 a cinque mani** e **+0,163 a dieci**:
fra cinque e dieci c'e' il passaggio da perdere a vincere. A venti sale a
+0,531, ed e' il motivo per cui conviene invitare alla rivincita — due partite
di fila sono venti mani sullo stesso cervello.
"""

from __future__ import annotations

import argparse
import sys

from robotica.games.rps.avversari import ATTESI
from robotica.games.rps.gioco import Esito, esito, leggi_mossa
from robotica.games.rps.predittore import Predittore
from robotica.games.rps.registro import Registro
from robotica.games.rps.sessione import riassunto_banco

MANI_PER_PARTITA = 10

BENVENUTO = """
sasso-carta-forbici — il robot dichiara la previsione PRIMA che tu tiri

  s = sasso    c = carta    f = forbici
  'basta' per smettere

Non puo' barare: annuncia la previsione prima di leggere la tua mossa.
Una partita e' {mani} mani.
"""


def _mostra_previsione(previsione, mano: int, totale: int) -> None:
    print(f"--- mano {mano}/{totale} ---")
    if previsione.dichiarata:
        print(
            f"prevedo: {str(previsione.attesa).upper()}  "
            f"(confidenza {previsione.confidenza:.0%})"
        )
    else:
        print("prevedo: ???  (non sono sicuro, ma gioco lo stesso)")
    print(f"perche': {previsione.perche}")


def _gioca_partita(pred: Predittore, mani: int) -> Registro | None:
    """Una partita. ``None`` se l'utente ha smesso a meta'."""
    reg = Registro()
    mano = 0
    while mano < mani:
        previsione = pred.prevedi()
        _mostra_previsione(previsione, mano + 1, mani)

        try:
            testo = input("tira (s/c/f): ")
        except (EOFError, KeyboardInterrupt):
            print()
            return reg if len(reg) else None
        if testo.strip().lower() in ("basta", "q", "esci"):
            return reg if len(reg) else None

        umana = leggi_mossa(testo)
        if umana is None:
            print("non ho capito. s, c oppure f.\n")
            continue

        e = esito(previsione.mossa, umana)
        parola = {
            Esito.VITTORIA: "vinco io",
            Esito.SCONFITTA: "vinci tu",
            Esito.PATTA: "patta",
        }[e]
        azzeccata = previsione.dichiarata and previsione.attesa == umana
        centro = "  <- te l'avevo detto" if azzeccata else ""
        print(f"tu {umana}, io {previsione.mossa}  ->  {parola}{centro}\n")

        reg.aggiungi(previsione, umana, e)
        pred.osserva(umana, previsione)
        mano += 1
    return reg


def comando_gioca(mani: int, seme: int | None) -> int:
    """Partite da ``mani`` mani, con lo sfidante che puo' rigiocare.

    **La memoria continua finche' e' la stessa persona.** Non c'e' nessun
    riconoscimento: si chiede, e basta. E' lo stesso principio di D19 —
    l'identita' viene dalla continuita', non da un tratto riconosciuto — con in
    piu' che qui non si raccoglie niente su nessuno.

    Se sbaglia (due amici si passano il posto senza dirlo) il danno e' che il
    modello si sporca per qualche mano e poi si riadatta.
    """
    print(BENVENUTO.format(mani=mani))
    pred = Predittore(seme=seme)
    totale = Registro()
    partite_giocate = 0

    while True:
        reg = _gioca_partita(pred, mani)
        if reg is None or not len(reg):
            break
        partite_giocate += 1
        totale.partite.extend(reg.partite)

        print("=" * 64)
        print(f"fine partita {partite_giocate}")
        print(reg.tabellone())
        print("=" * 64)

        try:
            risposta = input(
                "\nancora? [invio] rigiochi tu (mi ricordo di te)  "
                "[n] nuovo giocatore  [basta] smetti: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if risposta in ("basta", "q", "esci"):
            break
        if risposta == "n":
            pred.azzera()
            print("\ngiocatore nuovo: dimentico tutto.\n")
        else:
            print(f"\nbene: so gia' {len(pred.storia)} tue mosse.\n")

    if not len(totale):
        return 0

    print("\n" + "=" * 64)
    print(f"TOTALE — {partite_giocate} partite")
    print(totale.tabellone())
    print("=" * 64)
    return 0


def comando_banco(partite: int, seme: int) -> int:
    """Il predittore contro avversari dal comportamento noto.

    Restituisce 1 se qualcuno esce dalla finestra attesa: cosi' e' usabile
    anche come controllo rapido, non solo come stampa da leggere.
    """
    print(f"\nbanco di prova — {partite} partite per avversario, seme {seme}\n")
    print(f"{'avversario':<14} {'accur.':>8} {'netto':>8} {'sigma':>8} {'nondich':>8}   atteso")
    print("-" * 74)

    problemi = []
    for nome, reg in riassunto_banco(partite, seme):
        basso, alto = ATTESI.get(nome, (None, None))
        acc = reg.accuratezza_sbilanciata
        fuori = (basso is not None and acc < basso) or (alto is not None and acc > alto)
        if fuori:
            problemi.append((nome, acc, basso, alto))

        atteso = (
            f"{basso:.0%}+"
            if alto is None
            else f"<{alto:.0%}"
            if basso is None
            else f"{basso:.0%}-{alto:.0%}"
        )
        print(
            f"{nome:<14} {acc:>7.1%} {reg.netto:>+8.3f} {reg.sigma_dal_caso:>+8.1f} "
            f"{reg.rese:>8}   {atteso}{'   <-- FUORI' if fuori else ''}"
        )

    print()
    if problemi:
        print("FUORI DALLE ATTESE:")
        for nome, acc, basso, alto in problemi:
            if alto is not None and acc > alto:
                print(f"  {nome}: {acc:.1%} sopra il tetto {alto:.0%}.")
                if nome == "casuale":
                    print("    Contro il caso non si puo' prevedere: guarda se il")
                    print("    predittore sta vedendo la mossa corrente.")
            else:
                print(f"  {nome}: {acc:.1%} sotto il minimo {basso:.0%}.")
        return 1

    print("tutti dentro le attese.")
    print("il tetto su 'casuale' e' la riga che conta: la' fare troppo bene e' un baco.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sasso-carta-forbici: il predittore")
    sub = p.add_subparsers(dest="comando", required=True)

    g = sub.add_parser("gioca", help="gioca tu contro il predittore")
    g.add_argument(
        "-n", "--mani", type=int, default=MANI_PER_PARTITA, help="mani per partita (default 10)"
    )
    g.add_argument("--seme", type=int, default=None, help="seme per le mosse casuali")

    b = sub.add_parser("banco", help="il predittore contro avversari noti")
    b.add_argument("-n", "--partite", type=int, default=1000)
    b.add_argument("--seme", type=int, default=1234)

    a = p.parse_args(argv)
    if a.comando == "gioca":
        return comando_gioca(a.mani, a.seme)
    return comando_banco(a.partite, a.seme)


if __name__ == "__main__":
    sys.exit(main())
