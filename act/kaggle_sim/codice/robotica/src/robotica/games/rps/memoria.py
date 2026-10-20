"""La memoria di piazza: cio' che la popolazione ha insegnato, su disco.

In piazza le sessioni sono corte — dieci mani — e il predittore riparte da zero
a ogni persona. Ma i bias umani sono in buona parte **della specie**: quasi
tutti aprono con sasso, quasi tutti tengono la mossa che ha vinto. La piazza li
raccoglie da chi e' gia' passato e li passa a chi arriva, cosi' il robot non e'
spento alla mano 1.

Tre memorie, tre durate:

- la **partita**: dieci mani, e finisce (``Registro``)
- lo **sfidante**: finche' non dice di essere un altro (``Predittore``)
- la **piazza**: per sempre, su tutte le persone (questo file)


Due trappole, e come si evitano
-------------------------------

**Il doppio conteggio.** Il fondo innestato finisce dentro i conteggi dello
sfidante. Versare nella piazza *tutti* quei conteggi vorrebbe dire rimangiarsi
la piazza a ogni persona. ``Predittore.raccolto()`` restituisce quindi solo cio'
che **questa persona** ha insegnato: i conteggi meno il fondo.

**La forza dell'a priori.** Dopo trecento persone la piazza ha migliaia di
osservazioni per contesto: innestate cosi', schiaccerebbero le dieci mani della
persona vera, e il robot giocherebbe contro la media invece che contro di lei.
``fondo(forza)`` riduce ogni contesto a **al piu' ``forza`` osservazioni
finte**, con le stesse proporzioni: e' lo stesso principio di Laplace, ma con
un a priori imparato invece che piatto. Il valore di ``FORZA`` e' misurato.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from robotica.games.rps.gioco import Esito, Mossa

Conteggi = dict[str, dict[tuple, list[float]]]

#: Quante osservazioni finte vale al massimo un contesto della piazza.
#:
#: **Misurato** (2026-09-24): 100 persone di riscaldamento, poi 300 misurate a
#: 10 mani, netto. Modelli di piazza **in gara** e **pura popolazione** (non
#: imparano dallo sfidante):
#:
#:     K        win_stay  sbilanciata  mista   casuale  ciclico in piazza ws
#:     0        +0,180    +0,231       +0,132  +0,035   +0,504
#:     4        +0,743    +0,371       +0,351  -0,020   +0,400
#:     16       +0,741    +0,371       +0,347  -0,026   +0,600
#:     64       +0,750    +0,371       +0,329  -0,010   +0,600
#:     infinito +0,717    +0,371       +0,359  -0,008   +0,600
#:
#: Per confronto, **mescolando** il fondo nei modelli propri (la prima
#: versione) il ciclico in piazza ws faceva +0,004: la memoria affossava chi e'
#: diverso dalla media. In gara no. Fra 4 e infinito le differenze sono vicine
#: al rumore; 16 e' buono ovunque. ``casuale`` resta a zero a ogni K: la
#: memoria non inventa pattern.
FORZA = 16.0

_NOMI = {**{m.name: m for m in Mossa}, **{e.name: e for e in Esito}}


def _codifica(ctx: tuple) -> str:
    """Un contesto come testo leggibile: ``["SASSO", "VITTORIA"]``.

    Non numeri: ``Mossa`` ed ``Esito`` sono interi, e ricaricati come interi il
    predittore spiegherebbe "dopo 0 e 2 hai fatto 1". I nomi tornano enum.
    """
    return json.dumps([None if x is None else x.name for x in ctx])


def _decodifica(chiave: str) -> tuple:
    return tuple(None if x is None else _NOMI[x] for x in json.loads(chiave))


@dataclass
class Piazza:
    """I conteggi sommati su tutti gli sfidanti passati."""

    conteggi: Conteggi = field(default_factory=dict)
    sfidanti: int = 0

    def versa(self, raccolto: Conteggi) -> None:
        """Uno sfidante se ne va: cio' che ha insegnato entra nella piazza."""
        if not any(raccolto.values()):
            return
        for nome, contesti in raccolto.items():
            mio = self.conteggi.setdefault(nome, {})
            for ctx, conta in contesti.items():
                acc = mio.setdefault(ctx, [0.0, 0.0, 0.0])
                for i in range(3):
                    acc[i] += conta[i]
        self.sfidanti += 1

    def fondo(self, forza: float = FORZA) -> Conteggi:
        """La piazza come a priori: ogni contesto ridotto a ``forza`` osservazioni."""
        out: Conteggi = {}
        for nome, contesti in self.conteggi.items():
            out[nome] = {}
            for ctx, conta in contesti.items():
                tot = sum(conta)
                if tot <= 0:
                    continue
                k = min(1.0, forza / tot)
                out[nome][ctx] = [c * k for c in conta]
        return out

    # --- disco ----------------------------------------------------------------

    def salva(self, file: Path) -> None:
        """Scrittura **atomica**: prima un file temporaneo, poi lo scambio.

        Se il programma muore a meta' scrittura, resta il file vecchio intero
        invece di uno troncato. In piazza la corrente puo' saltare.
        """
        dati = {
            "sfidanti": self.sfidanti,
            "conteggi": {
                nome: {_codifica(ctx): conta for ctx, conta in contesti.items()}
                for nome, contesti in self.conteggi.items()
            },
        }
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_suffix(file.suffix + ".tmp")
        tmp.write_text(json.dumps(dati, indent=1), encoding="utf-8")
        os.replace(tmp, file)

    @classmethod
    def carica(cls, file: Path) -> Piazza:
        """La piazza salvata, o una vuota se non c'e' ancora."""
        if not file.exists():
            return cls()
        dati = json.loads(file.read_text(encoding="utf-8"))
        conteggi = {
            nome: {_decodifica(k): list(v) for k, v in contesti.items()}
            for nome, contesti in dati["conteggi"].items()
        }
        return cls(conteggi=conteggi, sfidanti=dati["sfidanti"])
