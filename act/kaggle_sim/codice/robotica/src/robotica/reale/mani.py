"""Le mani dell'agente: pochi comandi sicuri sul SO-101 vero, e gli occhi.

Un agente (un LLM, o una persona) non muove giunti: chiede "le dita qui",
"chiudi", "fammi vedere". Questo modulo traduce ogni richiesta in pose
verificate da ``safety/tavolo.py`` e le manda a ``BraccioVero``. **Lo stato
sta qui** — il braccio acceso, la camera, l'ultima posa — perche' l'esecutore,
quando si chiude, riporta il braccio a riposo: un comando per processo non
potrebbe mai fare "scendi, poi chiudi".

Scelte che proteggono dagli errori dell'agente, non solo dai suoi pericoli:

- ``vai`` passa **sempre in alto**: sale, va sopra il punto, scende. Un punto
  sbagliato di 5 cm porta le dita nel posto sbagliato, non le trascina sul
  tavolo ne' contro un bicchiere.
- Un rifiuto non e' un'eccezione che chiude tutto: e' una **risposta**, col
  motivo, e il braccio resta dov'era. L'agente la legge e si corregge.
- Le coordinate sono quelle del **centro fra le dita** (``centro_dita``), in
  metri nel frame del modello: il punto che conta per prendere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import cv2
import numpy as np

from robotica.reale.braccio import BraccioVero, Rifiuto
from robotica.safety.tavolo import Bicchiere, verifica_percorso, verifica_posa
from robotica.sim.so101_presa import (
    ASSE_BASE,
    TAVOLO,
    VERTICALE,
    centro_dita,
    comando,
    modello,
    pinza_aperta_chiusa,
    pinza_su_caramella,
    postura,
    postura_dita,
)

#: Quota delle dita (centro) per spostarsi da un punto all'altro: sopra caramelle
#: (~1,5 cm). Con la pinza verticale le dita non salgono molto (il polso): a 19 cm
#: dall'asse al massimo 2,5 cm. 3 cm e' la piu' bassa sicura: le dita restano ~2 cm
#: sopra il tavolo, sopra le caramelle (misurato 2026-10-11, out/prove/bordo.py).
QUOTE_DI_TRANSITO = (0.06, 0.05, 0.04, 0.03)

#: Parcheggio (distanza dall'asse, quota della punta, inclinazione): calcolato in
#: ``out/prove/parcheggio.py`` fra le pose che si raggiungono e passano safety.
PARCHEGGIO = (0.12, 0.20, np.radians(30))

VELOCITA = 30.0

#: Di quanto il tavolo vero sta **sotto** quello del modello, visto dal braccio.
#: Con le dita comandate a 2 mm dal tavolo, sul ferro restavano ~2 cm sopra
#: (stima a occhio di lato, 2026-10-12): tre prese chiuse sopra la caramella,
#: che non si e' mai mossa. Gli zeri dei servi sono giusti entro ~8 gradi (D23):
#: a 25 cm di braccio, 5 gradi di spalla sono 2 cm.
#: Prima 1,5 cm (prudente): la presa ancora sopra, e Daniele ha visto mancare
#: un altro centimetro. Quindi 2,5. Vale vicino a dove e' stata misurata
#: (10-13 cm dall'asse): un errore d'angolo non e' uno spostamento costante.
SCARTO_TAVOLO = 0.025


@dataclass
class Mani:
    braccio: BraccioVero
    camera: object | None = None  # reale/camera.CameraRete, o None senza occhi
    cartella: Path = Path("out/agente")
    bicchieri: tuple[Bicchiere, ...] = ()
    tavolo: float = TAVOLO - SCARTO_TAVOLO  # il tavolo vero, nel frame del modello
    _posa: np.ndarray | None = field(default=None, init=False)
    _pinza: str = field(default="chiusa", init=False)

    def __post_init__(self) -> None:
        self.model = self.braccio.model
        self.chiusa, self.aperta = pinza_aperta_chiusa(self.model)
        self.q_caramella = pinza_su_caramella(self.model)
        p = postura(self.model, np.array([ASSE_BASE[0] + PARCHEGGIO[0], 0.0,
                                          TAVOLO + PARCHEGGIO[1]]), PARCHEGGIO[2])  # del modello
        self._park = p

    # --- i tool -------------------------------------------------------------------

    def parcheggia(self) -> dict:
        return self._esegui("parcheggia", [comando(self._park, 0.0, self._q())])

    def vai(self, x: float, y: float, z: float) -> dict:
        """Le dita in (x, y, z), passando in alto. ``z`` sopra il tavolo, in metri."""
        try:
            tappe = self._via_alta(np.array([x, y, self.tavolo + z]))
        except ValueError as e:
            return {"ok": False, "motivo": str(e)}
        return self._esegui(f"vai ({x:.3f}, {y:.3f}, {z:.3f})", tappe)

    def apri(self) -> dict:
        return self._pinza_a("aperta")

    def chiudi(self) -> dict:
        return self._pinza_a("chiusa")

    def stato(self) -> dict:
        if self._posa is None:
            return {"ok": True, "dove": "a riposo (nessun comando ancora)", "pinza": self._pinza}
        d = centro_dita(self.model, np.r_[self._posa[:5], self.q_caramella])
        r = float(np.linalg.norm(d[:2] - ASSE_BASE))
        dita = (d[0], d[1], d[2] - self.tavolo)
        return {"ok": True, "dita": [round(float(v), 4) for v in dita],
                "distanza_asse_cm": round(r * 100, 1), "pinza": self._pinza}

    def foto(self, nome: str = "foto") -> dict:
        if self.camera is None:
            return {"ok": False, "motivo": "nessuna camera collegata"}
        f = self.camera.ultimo()
        if f is None:
            return {"ok": False, "motivo": "nessun fotogramma: il lettore su Windows e' partito?"}
        self.cartella.mkdir(parents=True, exist_ok=True)
        file = self.cartella / f"{nome}.png"
        cv2.imwrite(str(file), f.img)
        return {"ok": True, "file": str(file), "eta_ms": round(f.eta * 1000)}

    # --- dentro ---------------------------------------------------------------------

    def _q(self) -> float:
        return self.aperta if self._pinza == "aperta" else self.chiusa

    def _pinza_a(self, stato: str) -> dict:
        if self._posa is None:
            return {"ok": False, "motivo": "prima un movimento (parcheggia o vai)"}
        vecchia = self._pinza
        self._pinza = stato
        esito = self._esegui(stato, [np.r_[self._posa[:5], self._q()]])
        if not esito["ok"]:
            self._pinza = vecchia
        return esito

    def _dita(self, punto: np.ndarray) -> np.ndarray:
        p = postura_dita(self.model, punto, VERTICALE, self.q_caramella)
        return comando(p, 0.0, self._q())

    def _via_alta(self, punto: np.ndarray) -> list[np.ndarray]:
        """Sopra la posa attuale, sopra il punto, giu'. La prima quota di transito
        che si raggiunge in tutti e due i posti."""
        qui = None
        if self._posa is not None:
            qui = centro_dita(self.model, np.r_[self._posa[:5], self.q_caramella])
        for z in QUOTE_DI_TRANSITO:
            try:
                tappe = []
                if qui is not None and qui[2] - self.tavolo < z - 0.005:
                    tappe.append(self._dita(np.array([qui[0], qui[1], self.tavolo + z])))
                if punto[2] - self.tavolo < z - 0.005:
                    tappe.append(self._dita(np.array([punto[0], punto[1], self.tavolo + z])))
                tappe.append(self._dita(punto))
                return tappe
            except ValueError:
                continue
        raise ValueError(f"nessuna via in alto per ({punto[0]:.3f}, {punto[1]:.3f}): "
                         "il punto e' fuori dalla zona di presa (10-19 cm dall'asse)")

    def _esegui(self, nome: str, tappe: list[np.ndarray]) -> dict:
        rep = verifica_percorso(self.model, ([self._posa] if self._posa is not None else [])
                                + tappe, self.bicchieri, tavolo=self.tavolo)
        if not rep.ok:
            return {"ok": False, "motivo": f"safety: {rep.explain()}"}
        try:
            self.braccio.invia_tappe(nome, [(f"t{i}", t) for i, t in enumerate(tappe)], VELOCITA)
            esito = self.braccio.attendi()
        except Rifiuto as e:
            return {"ok": False, "motivo": str(e)}
        self._posa = tappe[-1]
        return {"ok": True, "errore_servo_gradi": esito.errore, **self.stato()}


def nuove(porta: str | None, camera: object | None = None) -> Mani:
    """Mani con un braccio nuovo (``porta=None``: braccio finto)."""
    model = modello()
    # La stessa quota del tavolo vero anche nella verifica che BraccioVero rifa' su ogni posa.
    tavolo = TAVOLO - SCARTO_TAVOLO
    braccio = BraccioVero(model, porta, partial(verifica_posa, tavolo=tavolo),
                          partial(verifica_percorso, tavolo=tavolo))
    return Mani(braccio, camera, tavolo=tavolo)
