"""Cosa ha giocato la persona, da cio' che la camera ha visto.

Dalla camera (``lerobot/camera/lettore_camera.py``, MediaPipe su Windows)
arrivano per ogni fotogramma due cose:

- **i gesti** che MediaPipe conosce, coi suoi nomi. Tre sono i nostri (pugno,
  palmo, V), ma il suo classificatore ha imparato su mani **dritte, dita in
  su**. A sasso-carta-forbici la mano si tiene **orizzontale, puntata verso
  l'avversario**: misurato il 2026-10-10, mano trovata in 94 fotogrammi su 94
  e gesto riconosciuto in **zero** (70 "None", 24 "Thumb_Down").
- **la forma della mano**: 21 punti in metri, centrati sulla mano. Da qui si
  contano le **dita tese** con la geometria, e una piega e' una piega da
  qualunque parte la si guardi. E' cio' che decide quando MediaPipe non sa.

**Si legge su piu' fotogrammi, a voti**, non su uno: la mano si sta ancora
aprendo quando il conto finisce. E **"non l'ho vista" e' una risposta**: la
mano diventa nulla e si rigioca. Tirare a indovinare falserebbe la partita e,
peggio, la memoria del predittore, che imparerebbe abitudini che la persona
non ha.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from robotica.games.rps.gioco import Mossa

#: Nome MediaPipe -> mossa. Solo questi tre contano.
GESTO_A_MOSSA: dict[str, Mossa] = {
    "Closed_Fist": Mossa.SASSO,
    "Open_Palm": Mossa.CARTA,
    "Victory": Mossa.FORBICI,
}

#: Sotto questa sicurezza un gesto di MediaPipe non vota (decide la geometria).
SOGLIA = 0.6

#: Voti validi minimi per dire qualcosa.
VOTI_MINIMI = 3

#: Quota dei voti che il vincitore deve avere: sotto, la mano e' indecisa.
MAGGIORANZA = 0.7

#: I quattro punti di ogni dito (nocca, due articolazioni, punta), numerazione
#: MediaPipe. Il pollice no: e' il piu' difficile da leggere e il gioco non lo usa.
DITA: dict[str, tuple[int, int, int, int]] = {
    "indice": (5, 6, 7, 8),
    "medio": (9, 10, 11, 12),
    "anulare": (13, 14, 15, 16),
    "mignolo": (17, 18, 19, 20),
}

#: Un dito e' teso se piega meno di cosi', sommando le due articolazioni
#: centrali. Teso: qualche decina di gradi; chiuso nel pugno: ben oltre 100.
#: **Da misurare sulle mani vere**: e' un punto a meta', non un numero noto.
PIEGA_MAX_TESO = np.radians(80)


class Osservazione(Protocol):
    """Un fotogramma, per quanto serve qui (``reale/camera.Osservazione``)."""

    gesti: Sequence[tuple[str, float]]
    mondo: np.ndarray | None


class Memoria(Protocol):
    """Cio' che la camera ricorda (``reale/camera.CameraRete``)."""

    def osservazioni_fra(self, da: float, a: float) -> Sequence[Osservazione]: ...

    def ricorda_da(self) -> float | None: ...


#: Finestra di lettura della mano, in secondi dopo il via: prima la mano si
#: sta ancora muovendo, dopo potresti aver visto il robot e cambiato idea.
LETTURA = (0.2, 1.2)

#: Secondi dopo la finestra in cui una mano che compare e' "in ritardo".
RITARDO = 1.0

#: Perche' una mano e' nulla. Il gioco decide su questo, non sul testo del motivo.
SCADUTA, NON_VISTA, INDECISA, IN_RITARDO = "scaduta", "non vista", "indecisa", "ritardo"


@dataclass(frozen=True)
class Lettura:
    """La mossa letta, o ``None`` col motivo per cui non si e' letta."""

    mossa: Mossa | None
    motivo: str
    voti: dict[Mossa, int]
    causa: str | None = None  # SCADUTA, NON_VISTA, INDECISA, IN_RITARDO; None se letta


def _piega(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Di quanto il segmento b->c devia dalla direzione a->b (0 = dritto)."""
    u, v = b - a, c - b
    coseno = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
    return float(np.arccos(np.clip(coseno, -1.0, 1.0)))


def dita_tese(mondo: np.ndarray) -> dict[str, bool]:
    """Quali delle quattro dita sono tese, dai 21 punti della mano."""
    p = np.asarray(mondo, dtype=float)
    tese = {}
    for nome, (nocca, a1, a2, punta) in DITA.items():
        piega = _piega(p[nocca], p[a1], p[a2]) + _piega(p[a1], p[a2], p[punta])
        tese[nome] = piega < PIEGA_MAX_TESO
    return tese


def mossa_da_forma(mondo: np.ndarray) -> Mossa | None:
    """La mossa dalla forma della mano, o ``None`` se non e' nessuna delle tre."""
    t = dita_tese(mondo)
    n = sum(t.values())
    if n <= 1:
        return Mossa.SASSO
    if n == 4:
        return Mossa.CARTA
    if t["indice"] and t["medio"] and not t["anulare"] and not t["mignolo"]:
        return Mossa.FORBICI
    return None  # tre dita, o due che non sono indice e medio: non e' un gesto del gioco


def mossa_di(oss: Osservazione) -> Mossa | None:
    """Il voto di un fotogramma.

    Prima MediaPipe, se e' sicuro e il gesto e' dei nostri; altrimenti decide la
    forma della mano, **qualunque nome** abbia dato MediaPipe: un pugno
    orizzontale che lui chiama "Thumb_Down" e' un pugno.
    """
    if oss.gesti:
        nome, punteggio = oss.gesti[0]
        if punteggio >= SOGLIA and nome in GESTO_A_MOSSA:
            return GESTO_A_MOSSA[nome]
    if oss.mondo is None:
        return None
    return mossa_da_forma(oss.mondo)


def leggi_mano(fotogrammi: Sequence[Osservazione]) -> Lettura:
    """La mossa di una finestra di fotogrammi."""
    voti = Counter(m for m in map(mossa_di, fotogrammi) if m is not None)
    totale = sum(voti.values())
    if totale < VOTI_MINIMI:
        return Lettura(None, f"mano non vista ({totale} voti su {len(fotogrammi)} fotogrammi)",
                       dict(voti), NON_VISTA)
    mossa, n = voti.most_common(1)[0]
    if n / totale < MAGGIORANZA:
        dettaglio = ", ".join(f"{m} {k}" for m, k in voti.most_common())
        return Lettura(None, f"indecisa: {dettaglio}", dict(voti), INDECISA)
    return Lettura(mossa, f"{n} voti su {totale}", dict(voti))


def leggi_tiro(camera: Memoria, via: float) -> Lettura:
    """La mossa mostrata al via, dalla memoria della camera.

    Si chiama **dopo** ``via + LETTURA[1]``, e il braccio puo' ancora muoversi:
    la camera ricorda, la lettura non aspetta il gesto. Prima si controlla che
    quei fotogrammi ci siano ancora: una finestra cancellata non e' una mano
    assente, e dirlo "non vista" accusa la camera di un errore del gioco.
    """
    da, a = via + LETTURA[0], via + LETTURA[1]
    inizio = camera.ricorda_da()
    if inizio is None or inizio > da:
        quando = "nessun fotogramma" if inizio is None else f"ricorda da {inizio - via:+.1f} s"
        return Lettura(None, f"finestra scaduta: serve da {LETTURA[0]:+.1f} s dal via, "
                       f"la camera {quando}", {}, SCADUTA)
    return leggi_mano(camera.osservazioni_fra(da, a))


#: Secondi di mano in vista per giocare, e senza mano per finire: si gioca
#: senza toccare il portatile. Scelti, non misurati: si regolano giocando
#: (3 s, 2026-10-11: "poco reattivo").
PER_PARTIRE = 0.7
PER_FINIRE = 10.0

#: Quota dei fotogrammi con la mano perche' "in vista" valga: qualche
#: fotogramma perso (mano mossa, sfocata) non deve azzerare l'attesa.
IN_VISTA = 0.8

#: Sotto questa quota la mano "non c'e'": un falso riconoscimento isolato non
#: deve tenere viva una partita che il giocatore ha lasciato.
ASSENTE = 0.1

#: Fotogrammi minimi per dire che la mano c'e': meno, e la camera non manda.
#: In 0,7 s a 20 fps ne arrivano 14: 5 lascia margine se la camera rallenta.
FOTOGRAMMI_MINIMI = 5


def quota_con_mano(fotogrammi: Sequence[Osservazione]) -> float:
    if not fotogrammi:
        return 0.0
    return sum(o.mondo is not None for o in fotogrammi) / len(fotogrammi)


def aspetta_mano(
    camera: Memoria, finire: bool, ora: Callable[[], float] = time.time,
    dormi: Callable[[float], None] = time.sleep,
) -> bool:
    """``True`` quando la mano resta in vista ``PER_PARTIRE`` secondi; ``False`` se
    (con ``finire``) per ``PER_FINIRE`` secondi non c'e'.

    I secondi si contano **da adesso**: la mano tenuta a fine mano precedente
    non vale, cosi' c'e' il tempo di guardare l'esito. ``ora`` e ``dormi`` si
    passano per provarla senza aspettare davvero.
    """
    t0 = ora()
    while True:
        t = ora()
        if t - t0 >= PER_PARTIRE:
            visti = camera.osservazioni_fra(t - PER_PARTIRE, t)
            if len(visti) >= FOTOGRAMMI_MINIMI and quota_con_mano(visti) >= IN_VISTA:
                return True
        if finire and t - t0 >= PER_FINIRE:
            if quota_con_mano(camera.osservazioni_fra(t - PER_FINIRE, t)) < ASSENTE:
                return False
        dormi(0.1)


def in_ritardo(camera: Memoria, via: float, lettura: Lettura) -> Lettura:
    """Se la mano non c'era nella finestra ma compare subito dopo, la lettura lo dice.

    Si chiama dopo ``via + LETTURA[1] + RITARDO``. Non cambia l'esito (la mano
    resta nulla: leggerla dopo vorrebbe dire lasciar vedere il robot), cambia
    il consiglio: "mostrala prima", non "la camera non ti vede".
    """
    if lettura.causa != NON_VISTA:
        return lettura
    a = via + LETTURA[1]
    dopo = leggi_mano(camera.osservazioni_fra(a, a + RITARDO))
    if dopo.mossa is None:
        return lettura
    return Lettura(None, f"in ritardo: {dopo.mossa} mostrato dopo la finestra "
                   f"({LETTURA[1]:.1f} s dal via)", lettura.voti, IN_RITARDO)
