"""Il SO-101 vero visto dal progetto: un gesto entra, ``safety/`` lo verifica, il ferro lo fa.

Dall'altra parte c'e' ``lerobot/esecutore.py``, in un processo e un venv suoi
(numpy 2 la', 1 qui: D22, D23). Gli si parla con righe JSON: un comando, una
risposta.

**La verifica sta qui dentro, non in chi chiama.** Ogni gesto passa da
``safety/gesti.py`` prima di diventare JSON: uno script nuovo non puo'
dimenticarsene, perche' non c'e' un'altra strada verso il ferro. Poi
l'esecutore lo verifica di nuovo sul ferro (range vero, letture fresche,
inseguimento). Due verifiche perche' sanno cose diverse: il modello sa dov'e'
la mano del giocatore, il ferro sa dove arriva davvero il polso.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from robotica.paths import ROOT
from robotica.reale import so101
from robotica.safety.checks import SafetyReport
from robotica.safety.gesti import verifica_gesto

LEROBOT = ROOT.parent / "lerobot"

#: Le regole per una posa (``safety/gesti.py``, ``safety/tavolo.py``).
Verifica = Callable[[mujoco.MjModel, np.ndarray], SafetyReport]
#: Le regole per una sequenza di pose e i tratti fra loro.
VerificaPercorso = Callable[[mujoco.MjModel, list[np.ndarray]], SafetyReport]


class Rifiuto(RuntimeError):
    """Un gesto non e' stato fatto: chi l'ha fermato e perche'."""


@dataclass(frozen=True)
class Esito:
    esito: str  # "fatto" | "rifiutato" | "fermato"
    motivo: str
    errore: dict[str, float]  # gradi LeRobot, a fine movimento
    inizi: tuple[float, ...] = ()  # coreografia: time.time() d'inizio di ogni tappa


class BraccioVero:
    """Avvia l'esecutore e gli passa gesti verificati. Usalo con ``with``: all'uscita
    l'esecutore riporta il braccio da dove e' partito e lascia la coppia."""

    def __init__(
        self, model: mujoco.MjModel, porta: str | None,
        verifica: Verifica = verifica_gesto, percorso: VerificaPercorso | None = None,
    ) -> None:
        """``porta=None``: braccio finto (``esecutore.py --prova``), nessuna porta aperta.

        ``verifica``: le regole per ogni posa. Il gioco usa ``safety/gesti.py`` (il
        default), il tavolo ``safety/tavolo.py``: regole diverse perche' i pericoli
        sono diversi (una persona davanti, o un tavolo e dei bicchieri). Si
        sceglie **quale**, non **se**: senza verifica non si crea un braccio.
        ``percorso``: se c'e', controlla anche i tratti **fra** le pose, dall'ultima
        posa mandata in poi.
        """
        self.model = model
        self._verifica, self._percorso = verifica, percorso
        self._ultima: np.ndarray | None = None  # l'ultima posa mandata: da li' parte il tratto
        self._range_pinza = tuple(model.joint(so101.PINZA).range)
        cmd = [str(LEROBOT / ".venv/bin/python"), str(LEROBOT / "esecutore.py")]
        cmd += ["--prova"] if porta is None else ["--porta", porta]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self._id = 0
        self._in_volo: str | None = None  # il gesto inviato di cui non si e' letta la risposta

    def gesto(self, nome: str, ctrl: np.ndarray, velocita: float | None = None) -> Esito:
        """Fa il gesto e aspetta che finisca, o solleva ``Rifiuto`` dicendo chi l'ha fermato.

        ``velocita`` in gradi/s: l'esecutore ha il suo tetto e lo applica lui.
        """
        self.invia(nome, ctrl, velocita)
        return self.attendi()

    def invia(self, nome: str, ctrl: np.ndarray, velocita: float | None = None) -> None:
        """Verifica e manda il gesto **senza aspettarlo**: il braccio parte, chi chiama
        e' libero (per esempio di guardare la camera mentre il braccio si muove).

        L'esecutore e' un altro processo: il braccio si muove comunque, aspettare
        serve solo a sapere com'e' finita. Lo dice ``attendi()``, da chiamare
        prima del gesto successivo. ``safety/`` resta **prima** dell'invio: un
        gesto rifiutato non parte.
        """
        self._controlla_percorso(nome, [ctrl])
        self._manda(nome, {"pos": self._verificato(nome, ctrl)}, velocita)
        self._ultima = np.asarray(ctrl, dtype=float)

    def invia_tappe(
        self, nome: str, tappe: list[tuple[str, np.ndarray]], velocita: float | None = None
    ) -> None:
        """Come ``invia``, ma una **coreografia**: pose eseguite di fila, senza fermarsi
        fra l'una e l'altra (le battute del conto). **Ogni** posa passa da
        ``safety/``: basta una rifiutata e non parte niente. ``attendi()`` dice
        anche l'ora d'inizio di ogni tappa (``Esito.inizi``)."""
        pose = [self._verificato(f"{nome}/{n}", ctrl) for n, ctrl in tappe]
        self._controlla_percorso(nome, [ctrl for _, ctrl in tappe])
        self._manda(nome, {"tappe": pose}, velocita)
        self._ultima = np.asarray(tappe[-1][1], dtype=float)

    def _verificato(self, nome: str, ctrl: np.ndarray) -> dict[str, float]:
        rep = self._verifica(self.model, ctrl)
        if not rep.ok:
            raise Rifiuto(f"safety rifiuta '{nome}': {rep.explain()}")
        return so101.a_lerobot(ctrl, self._range_pinza)

    def _controlla_percorso(self, nome: str, pose: list[np.ndarray]) -> None:
        """I tratti fra le pose, dall'ultima mandata. Al primo comando non si sa da
        dove parte il braccio (la posa di riposo letta dall'esecutore): si
        controllano solo i tratti fra le pose nuove."""
        if self._percorso is None:
            return
        tratti = ([self._ultima] if self._ultima is not None else []) + list(pose)
        rep = self._percorso(self.model, tratti)
        if not rep.ok:
            raise Rifiuto(f"safety rifiuta il percorso di '{nome}': {rep.explain()}")

    def _manda(self, nome: str, corpo: dict, velocita: float | None) -> None:
        if self._in_volo is not None:
            raise RuntimeError(f"'{self._in_volo}' e' ancora in corso: prima attendi()")
        comando = {"id": self._id, "gesto": nome} | corpo
        if velocita is not None:
            comando["velocita"] = velocita
        self._id += 1
        assert self._proc.stdin is not None and self._proc.stdout is not None
        try:
            self._proc.stdin.write(json.dumps(comando) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError as e:
            raise Rifiuto(f"l'esecutore non c'e' piu' ('{nome}')") from e
        self._in_volo = nome

    def attendi(self) -> Esito:
        """La risposta del gesto inviato: aspetta che il braccio abbia finito."""
        nome, self._in_volo = self._in_volo, None
        if nome is None:
            raise RuntimeError("nessun gesto in corso")
        assert self._proc.stdout is not None
        riga = self._proc.stdout.readline()
        if not riga:
            raise Rifiuto(f"l'esecutore si e' chiuso senza rispondere ('{nome}')")
        r = json.loads(riga)
        esito = Esito(r["esito"], r["motivo"], r["errore"], tuple(r.get("inizi", ())))
        if esito.esito != "fatto":
            raise Rifiuto(f"esecutore: '{nome}' {esito.esito.upper()}: {esito.motivo}")
        return esito

    def chiudi(self) -> int:
        """Fine dei comandi: l'esecutore rientra lento e lascia la coppia."""
        if self._proc.stdin is not None and not self._proc.stdin.closed:
            self._proc.stdin.close()
        return self._proc.wait()

    def __enter__(self) -> BraccioVero:
        return self

    def __exit__(self, *_: object) -> None:
        self.chiudi()
