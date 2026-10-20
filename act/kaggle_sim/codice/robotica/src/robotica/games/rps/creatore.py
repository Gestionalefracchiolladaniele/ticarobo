"""Il diario del creatore: cio' che il giocatore **non** vede.

Lo usano il gioco in 3D (``scripts/rps.py``) e quello col braccio vero
(``scripts/rps_vero.py``). Una copia sola: due diari diversi descriverebbero
due giochi diversi, e le partite non si potrebbero confrontare.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from robotica.games.rps.gioco import Esito, Mossa
from robotica.games.rps.mano import Lettura
from robotica.games.rps.predittore import Previsione
from robotica.games.rps.registro import Registro
from robotica.paths import OUT


class Creatore:
    """La vista di chi ha costruito il robot: **cio' che il giocatore non vede**.

    Nel terminale la previsione compare **prima** del tiro, come la vedrebbe
    il pubblico se fosse pubblica: cosi' si controlla dal vivo che il robot
    decida davvero prima. Nel file, una riga JSON per mano (JSONL): si
    aggiunge senza riscrivere, e se il programma si chiude a meta' le righe
    gia' scritte restano valide.
    """

    def __init__(self, cartella: Path = OUT / "rps") -> None:
        cartella.mkdir(parents=True, exist_ok=True)
        self.file = cartella / f"partite-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
        self.sfidante = 1

    def previsione(self, mano: int, prev: Previsione) -> None:
        detto = f"{str(prev.attesa).upper()} al {prev.confidenza:.0%}" if prev.dichiarata else "???"
        print(f"[creatore] sfidante {self.sfidante}, mano {mano}: prevedo {detto}, "
              f"giochero' {str(prev.mossa).upper()}  ({prev.modello})")
        print(f"           perche': {prev.perche}")

    def mano(
        self, mano: int, prev: Previsione, umana: Mossa, e: Esito, reg: Registro,
        letta: str | None = None,
    ) -> None:
        """``letta``: chi ha letto la mossa ("tastiera", "camera: 12 voti su 14")."""
        colpo = "  <- azzeccata" if prev.dichiarata and prev.attesa == umana else ""
        print(f"           tira {str(umana).upper()} -> {e}{colpo}   "
              f"(azzeccate {reg.indovinate}/{reg.sbilanciate} dichiarate, netto {reg.netto:+.2f})")
        self._scrivi({
            "sfidante": self.sfidante, "mano": mano,
            "attesa": None if prev.attesa is None else str(prev.attesa),
            "confidenza": round(prev.confidenza, 4), "robot": str(prev.mossa),
            "modello": prev.modello, "perche": prev.perche,
            "umano": str(umana), "esito": int(e), "letta": letta,
        })

    def nulla(self, mano: int, prev: Previsione, lettura: Lettura) -> None:
        """Una mano non letta: non conta, ma si scrive. Le mani nulle sono la
        misura della camera: se restano solo sullo schermo, si perdono."""
        self._scrivi({
            "sfidante": self.sfidante, "mano": mano, "robot": str(prev.mossa),
            "nulla": lettura.causa, "motivo": lettura.motivo,
            "voti": {str(m): n for m, n in lettura.voti.items()},
        })

    def _scrivi(self, riga: dict) -> None:
        riga = {"ora": datetime.now().isoformat(timespec="seconds")} | riga
        with self.file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(riga, ensure_ascii=False) + "\n")

    def fine_partita(self, reg: Registro) -> None:
        print(f"\n[creatore] fine partita, sfidante {self.sfidante}\n{reg.tabellone()}\n")

    def nuovo_sfidante(self) -> None:
        self.sfidante += 1
