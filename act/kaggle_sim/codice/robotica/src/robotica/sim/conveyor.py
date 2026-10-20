"""Misurare un nastro: i contatori, e perche' restano separati.

Il nastro cambia la natura del compito, non la sua difficolta'. Nello
smistamento (D13, D14) i cubi stanno fermi: si legge lo stato, si calcola un
piano, lo si esegue, e il mondo aspetta. Qui **il mondo si muove mentre il piano
viene eseguito**, e un piano calcolato all'inizio scade prima di finire.

Da qui due conseguenze che questo modulo serve a tenere dritte.

## Le due misure non si sommano

| Misura | Domanda | Trasferisce al reale? |
|---|---|---|
| decisione | arriva al punto giusto al momento giusto? | si', 85-98% |
| presa | una volta arrivato, afferra? | no |

Sono capacita' diverse e trasferiscono in modo diverso. Un numero unico —
"12 cubi su 20" — non direbbe **di chi e' la colpa** quando il totale cala: se
il robot decide tardi o se decide bene e la pinza scivola. Il primo problema si
risolve col software, il secondo no, e in simulazione il secondo non e' nemmeno
informativo (la presa simulata non predice quella reale).

## Il terzo esito: non agire non e' fallire

``BELT_RULE`` ha un colore che non va in nessuna zona. Finora ogni cubo che
compariva era un compito, e le risposte erano due: fatto o non fatto. Con una
categoria da ignorare il robot deve poter **decidere di non agire**.

Un cubo giallo che esce dal nastro e' un **successo**; un cubo rosso che fa lo
stesso e' un **errore**. Fisicamente sono lo stesso evento — un cubo che cade
dal bordo — e finirebbero volentieri nello stesso contatore. Se ci finissero, il
resoconto direbbe la stessa cosa di un robot che lavora bene e di uno che non
prende niente.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConveyorStats:
    """I contatori del nastro, tenuti distinti **per costruzione**.

    Non esiste un metodo che li sommi in un unico "successi", ed e' voluto:
    l'unico modo di perdere la distinzione sarebbe sommarli a mano, che e'
    un'operazione che si vede leggendo il codice.
    """

    ingaggiati: int = 0
    """Il robot ha deciso in tempo e si e' mosso verso il cubo.

    E' la misura che **trasferisce al reale**: decidere cosa fare e quando
    dipende dal software, non dall'attrito fra gomma e plastica."""

    afferrati: int = 0
    """Il cubo e' finito davvero in mano.

    E' la misura che **non trasferisce**: la presa simulata dice poco su quella
    reale, e il progetto lo sa da sempre (``transfer-sim-real.md``)."""

    posati: int = 0
    """Il cubo e' arrivato nella zona che gli spettava."""

    persi: int = 0
    """Passato oltre la finestra di presa senza essere ingaggiato.

    **E' un errore**, e va tenuto lontano da ``lasciati``."""

    lasciati: int = 0
    """Lasciato passare di proposito: il colore non va in nessuna zona.

    **E' un successo**, e va tenuto lontano da ``persi``. Stesso evento fisico,
    esito opposto."""

    errori: list[str] = field(default_factory=list)
    """Cosa e' andato storto, in parole. Un contatore che cala senza dire
    perche' costringe a indovinare (D2)."""

    def riga(self) -> str:
        """Una riga leggibile, con i contatori separati."""
        return (
            f"ingaggiati {self.ingaggiati}  afferrati {self.afferrati}  "
            f"posati {self.posati}  persi {self.persi}  lasciati {self.lasciati}"
        )

    def explain(self, attesi_da_smistare: int, attesi_da_lasciare: int) -> str:
        """Il resoconto, che confronta cio' che e' successo con cio' che doveva.

        Si passano gli attesi invece di dedurli: i contatori sanno cosa e'
        accaduto, non cosa avrebbe dovuto accadere, e dedurlo da loro stessi
        renderebbe il resoconto sempre coerente — anche quando e' sbagliato.
        """
        righe = [self.riga(), ""]
        righe.append(
            f"Da smistare: {attesi_da_smistare:>3}   posati correttamente: {self.posati}"
        )
        righe.append(
            f"Da lasciare: {attesi_da_lasciare:>3}   lasciati passare:     {self.lasciati}"
        )

        if self.ingaggiati:
            resa = 100.0 * self.afferrati / self.ingaggiati
            righe.append("")
            righe.append(
                f"Decisione: {self.ingaggiati} su {attesi_da_smistare} ingaggiati in tempo"
            )
            righe.append(
                f"Presa:     {self.afferrati} su {self.ingaggiati} afferrati ({resa:.0f}%)"
            )

        if self.errori:
            righe.append("")
            righe.append("Problemi:")
            righe.extend(f"  - {e}" for e in self.errori[:10])

        return "\n".join(righe)


def sequenza_venti() -> list[str]:
    """Venti cubi, ordinati per mettere alla prova cose diverse.

    Non e' casuale, e nemmeno regolare. Cinque blocchi, ciascuno con una
    domanda a cui un ordine casuale risponderebbe solo per fortuna:

    1. **uno per colore** — ogni categoria funziona da sola?
    2. **due uguali di fila** — il caso peggiore per la presa: il secondo
       arriva mentre il braccio sta ancora posando il primo
    3. **gialli in mezzo ad altri** — il robot li *salta*, o si ferma a
       guardarli? Un giallo isolato in coda proverebbe solo che a un certo
       punto smette
    4. **alternanza fra le due zone** — il braccio cambia lato, che e' il
       movimento piu' lungo e quello dopo cui e' piu' facile perdere il cubo
       successivo
    5. **due gialli di fila, poi due da smistare** — riparte dopo una pausa?

    Un fallimento su una sequenza casuale non si saprebbe ricondurre a quale
    caso; qui si sa leggendo dove si e' rotto.
    """
    return [
        # 1 — uno per colore
        "rosso", "verde", "blu", "giallo",
        # 2 — coppie: il secondo arriva mentre posi il primo
        "rosso", "rosso", "verde", "verde",
        # 3 — gialli fra cubi da prendere
        "giallo", "blu", "giallo", "rosso",
        # 4 — alternanza fra le zone: il braccio cambia lato
        "verde", "blu", "verde", "rosso",
        # 5 — due gialli di fila, poi si riparte
        "giallo", "giallo", "blu", "verde",
    ]
