"""Dal pixel al punto del tavolo, con una camera normale (senza profondita').

La camera del kit da' i colori, non le distanze: ``detect.py`` (che segmenta
sulla depth) qui non serve. Ma le caramelle stanno **su un piano**, e fra un
piano visto in foto e il piano vero basta una trasformazione 3x3, l'**omografia**:

    [u, v, 1]  ~  H @ [x, y, 1]      (pixel <- punto del piano, a meno di un fattore)

La si calcola da 4 o piu' coppie (pixel, punto) con ``cv2.findHomography``.
Niente orientamento della camera da stimare, niente scacchiera: meno cose da
misurare, meno modi di sbagliare. Vale **per un piano solo**: un punto piu' in
alto o piu' in basso finisce nel pixel di un altro punto (parallasse). Per
questo si calibra **alla quota a cui si prendera'**.

**Le coppie le fornisce il robot stesso** (calibrazione automatica): mette la
punta della pinza in un punto noto dal modello, e la camera la cerca come
**differenza** fra la foto col braccio li' e quella col braccio parcheggiato.
La punta e' il pixel del braccio **piu' in basso nell'immagine**: e' il punto
piu' vicino al tavolo e, con la camera davanti e in alto, alla camera.

**L'errore si misura lasciando fuori un punto alla volta**: con gli altri si
calcola H, e si guarda dove H manda quello escluso. E' l'errore su un punto
che il calcolo non ha visto — l'unico onesto senza un metro esterno.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

#: Differenza di luminosita' (0-255) oltre la quale un pixel "e' cambiato".
SOGLIA_DIFFERENZA = 30

#: Macchie cambiate piu' piccole di cosi' sono rumore (pixel).
AREA_MINIMA = 150


@dataclass(frozen=True)
class Calibrazione:
    """Pixel <-> tavolo, con l'errore misurato."""

    H: np.ndarray  # 3x3: punto del piano (metri, frame del modello) -> pixel
    quota: float  # z del piano calibrato, metri
    errori_mm: tuple[float, ...]  # lasciandone fuori uno alla volta

    def a_pixel(self, xy: np.ndarray) -> np.ndarray:
        return _applica(self.H, np.atleast_2d(xy))

    def a_tavolo(self, uv: np.ndarray) -> np.ndarray:
        return _applica(np.linalg.inv(self.H), np.atleast_2d(uv))

    @property
    def errore_mm(self) -> float:
        """Il peggiore: e' quello che fa mancare una presa."""
        return max(self.errori_mm)

    def salva(self, file: Path) -> None:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps({"H": self.H.tolist(), "quota": self.quota,
                                    "errori_mm": list(self.errori_mm)}, indent=1))

    @staticmethod
    def carica(file: Path) -> Calibrazione:
        dati = json.loads(file.read_text())
        return Calibrazione(np.array(dati["H"]), dati["quota"], tuple(dati["errori_mm"]))


def _applica(H: np.ndarray, p: np.ndarray) -> np.ndarray:
    o = np.hstack([p, np.ones((len(p), 1))]) @ H.T
    return o[:, :2] / o[:, 2:]


def punta_dalla_differenza(
    riferimento: np.ndarray, ora: np.ndarray, zona: np.ndarray | None = None
) -> tuple[float, float] | None:
    """Il pixel della punta: il punto piu' in basso della macchia cambiata piu' grande.

    ``zona``: maschera booleana dei pixel dove cercare (il tavolo). ``None`` se
    non cambia niente di abbastanza grande: dire "non la vedo" e' una risposta.
    """
    a = cv2.cvtColor(riferimento, cv2.COLOR_BGR2GRAY).astype(np.int16)
    b = cv2.cvtColor(ora, cv2.COLOR_BGR2GRAY).astype(np.int16)
    cambiato = (np.abs(b - a) > SOGLIA_DIFFERENZA).astype(np.uint8)
    if zona is not None:
        cambiato &= zona.astype(np.uint8)
    # L'apertura toglie il rumore ma **rosicchia le parti sottili**, e la punta lo
    # e': misurato, 5 pixel persi (test). Serve solo a scegliere la macchia; la
    # punta si cerca nei pixel originali attorno a lei.
    pulito = cv2.morphologyEx(cambiato, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(pulito)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < AREA_MINIMA:
        return None
    attorno = cv2.dilate((lab == i).astype(np.uint8), np.ones((7, 7), np.uint8))
    ys, xs = np.nonzero(attorno & cambiato)
    basso = ys.max()
    # Al bordo inferiore la punta e' qualche pixel largo: se ne prende il centro.
    return float(xs[ys >= basso - 1].mean()), float(basso)


#: Blu delle caramelle in HSV di OpenCV (tono 0-180): misurato il 2026-10-11,
#: tono 102-103 con saturazione alta (smistamento.md).
BLU_TONO, BLU_SAT, BLU_VAL = (95, 130), 90, 60


def maschera_blu(img: np.ndarray) -> np.ndarray:
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    return ((h >= BLU_TONO[0]) & (h <= BLU_TONO[1]) & (s > BLU_SAT) & (v > BLU_VAL))


def blu_nuovo(riferimento: np.ndarray, ora: np.ndarray) -> tuple[float, float] | None:
    """Il centro del blu che c'e' in ``ora`` e non c'era nel ``riferimento``: la
    caramella nella pinza, arrivata li'. Le caramelle blu ferme sul tavolo sono
    in tutte e due le foto e si annullano: non serve toglierle."""
    prima = cv2.dilate(maschera_blu(riferimento).astype(np.uint8), np.ones((7, 7), np.uint8))
    nuovo = (maschera_blu(ora) & (prima == 0)).astype(np.uint8)
    nuovo = cv2.morphologyEx(nuovo, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(nuovo)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < 40:
        return None
    return float(cen[i][0]), float(cen[i][1])


@dataclass(frozen=True)
class DuePiani:
    """Due calibrazioni a quote diverse: ogni pixel diventa un **raggio**.

    Una omografia vale per un piano. Con la camera di sbieco, una caramella
    tenuta a 3 cm e una appoggiata sul tavolo, nello **stesso pixel**, stanno in
    punti del tavolo diversi di centimetri (parallasse). Con due piani ogni
    pixel da' due punti, e la retta che li unisce e' il raggio della camera: lo si
    prolunga fino alla quota che serve. E' il *modello a due piani* (two-plane
    camera model): niente focale ne' orientamento della camera da stimare.
    """

    basso: Calibrazione
    alto: Calibrazione

    def a_tavolo(self, uv: np.ndarray, quota: float) -> np.ndarray:
        """Il punto (x, y) alla ``quota`` data, per ogni pixel (N x 2)."""
        p1, p2 = self.basso.a_tavolo(uv), self.alto.a_tavolo(uv)
        t = (quota - self.basso.quota) / (self.alto.quota - self.basso.quota)
        return p1 + t * (p2 - p1)

    @property
    def errore_mm(self) -> float:
        return max(self.basso.errore_mm, self.alto.errore_mm)

    def salva(self, file: Path) -> None:
        file.parent.mkdir(parents=True, exist_ok=True)
        dati = {nome: {"H": c.H.tolist(), "quota": c.quota, "errori_mm": list(c.errori_mm)}
                for nome, c in (("basso", self.basso), ("alto", self.alto))}
        file.write_text(json.dumps(dati, indent=1))

    @staticmethod
    def carica(file: Path) -> DuePiani:
        dati = json.loads(file.read_text())
        piani = [Calibrazione(np.array(dati[n]["H"]), dati[n]["quota"],
                              tuple(dati[n]["errori_mm"])) for n in ("basso", "alto")]
        return DuePiani(*piani)


def calibra(pixel: np.ndarray, punti: np.ndarray, quota: float) -> Calibrazione:
    """H dalle coppie (pixel, punto del piano), e l'errore lasciando fuori un punto.

    Servono almeno 5 coppie: 4 per calcolare H, una da lasciare fuori.
    """
    pixel, punti = np.asarray(pixel, float), np.asarray(punti, float)[:, :2]
    if len(pixel) < 5:
        raise ValueError(f"servono almeno 5 punti, ne ho {len(pixel)}")
    H, _ = cv2.findHomography(punti, pixel)
    if H is None:
        raise ValueError("omografia non trovata: punti allineati o ripetuti?")
    errori = []
    for k in range(len(pixel)):
        altri = np.arange(len(pixel)) != k
        Hk, _ = cv2.findHomography(punti[altri], pixel[altri])
        stimato = _applica(np.linalg.inv(Hk), pixel[k:k + 1])[0]
        errori.append(float(np.linalg.norm(stimato - punti[k]) * 1000))
    return Calibrazione(H, quota, tuple(errori))
