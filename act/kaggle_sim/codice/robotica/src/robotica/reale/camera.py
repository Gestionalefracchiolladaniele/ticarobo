"""La camera vera, letta su Windows e ricevuta qui: tiene l'ultimo fotogramma.

Dall'altra parte c'e' `lerobot/camera/lettore_camera.py`. La camera non si
legge da WSL perche' via usbipd arriva a pezzi (vedi quel file e
`esperimenti/so101/hardware.md`).

Si tiene **solo l'ultimo** fotogramma: chi lo chiede vuole l'adesso, e una coda
di immagini vecchie farebbe decidere il robot sul passato. Ogni fotogramma ha
l'**eta'**: dall'ora dello scatto (orologio di Windows) all'ora in cui lo si
chiede (orologio di WSL). WSL sincronizza l'orologio con Windows, ma non e'
garantito al millisecondo: l'eta' e' una misura, e va guardata, non assunta.

Con l'immagine arriva cio' che MediaPipe ha visto, **su Windows**: i gesti che
conosce (nomi suoi, ``Closed_Fist`` e simili) e i 21 punti della mano. Cosa sia
"sasso" non si decide qui ma nel gioco (``games/rps/mano.py``).
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

#: ora dello scatto, lunghezza del JSON, lunghezza del JPEG (lettore_camera.py).
INTESTAZIONE = struct.Struct(">dII")

#: Secondi senza dati dopo i quali un lettore e' dato per morto.
SILENZIO_MAX = 3.0


@dataclass(frozen=True)
class Osservazione:
    """Cio' che serve per leggere una mossa da un fotogramma, senza l'immagine."""

    gesti: tuple[tuple[str, float], ...]
    mondo: np.ndarray | None


@dataclass(frozen=True)
class Fotogramma:
    img: np.ndarray  # BGR, come OpenCV
    scatto: float  # time.time() di Windows
    numero: int
    gesti: tuple[tuple[str, float], ...] = ()  # (nome MediaPipe, punteggio), il migliore prima
    mano: np.ndarray | None = None  # 21 x 3, coordinate normalizzate all'immagine
    ms_riconoscimento: float = 0.0
    mondo: np.ndarray | None = None  # 21 x 3, metri, centrate sulla mano: la sua forma

    @property
    def eta(self) -> float:
        """Secondi dallo scatto ad adesso."""
        return time.time() - self.scatto

    @property
    def osservazione(self) -> Osservazione:
        return Osservazione(self.gesti, self.mondo)


class CameraRete:
    """Ascolta su ``porta`` e decomprime in un thread. ``ultimo()`` non aspetta mai."""

    #: Secondi di gesti ricordati: per leggere una finestra **gia' passata**
    #: ("cosa hai visto subito dopo il via?"). Solo i gesti, non le immagini:
    #: 15 s a 20 fps sono 300 forme di mano, niente. Con 5 s la finestra della
    #: carta era gia' cancellata quando il gesto finiva (2026-10-11).
    MEMORIA = 15.0

    def __init__(self, porta: int = 5600) -> None:
        self._server = socket.create_server(("0.0.0.0", porta))
        self._lock = threading.Lock()
        self._ultimo: Fotogramma | None = None
        self._storia: deque[tuple[float, Osservazione]] = deque()
        self._numero = 0
        self._chiuso = False
        threading.Thread(target=self._ascolta, daemon=True).start()

    def ultimo(self) -> Fotogramma | None:
        with self._lock:
            return self._ultimo

    def osservazioni_fra(self, da: float, a: float) -> list[Osservazione]:
        """Cio' che si e' visto nei fotogrammi **scattati** fra ``da`` e ``a`` (``time.time()``).

        Si confronta l'ora di Windows con quella di WSL: sono sincronizzate,
        e l'eta' misurata (mediana 34 ms) dice che lo scarto e' piccolo.
        """
        with self._lock:
            return [g for t, g in self._storia if da <= t <= a]

    def ricorda_da(self) -> float | None:
        """L'ora di scatto del fotogramma piu' vecchio ancora ricordato.

        Serve a distinguere "non c'era nessuna mano" da "quei fotogrammi sono
        gia' stati cancellati": per ``osservazioni_fra`` sono la stessa lista
        vuota, e scambiarli dice "mano non vista" quando la camera l'ha vista.
        """
        with self._lock:
            return self._storia[0][0] if self._storia else None

    def chiudi(self) -> None:
        self._chiuso = True
        self._server.close()

    def _ascolta(self) -> None:
        while not self._chiuso:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            # Un lettore ucciso di colpo lascia la connessione mezza aperta (wslrelay
            # la tiene): niente dati, niente chiusura. Servite una alla volta, ~110 di
            # queste in coda hanno tenuto fuori il lettore vivo (2026-10-12). Quindi
            # ognuna nel suo thread, con una scadenza: vince chi manda fotogrammi.
            # Il lettore ne manda ~10 al secondo: 3 s di silenzio vogliono dire morto.
            conn.settimeout(SILENZIO_MAX)
            threading.Thread(target=self._servi, args=(conn,), daemon=True).start()

    def _servi(self, conn: socket.socket) -> None:
        with conn:
            try:
                self._ricevi(conn)
            except OSError:  # anche TimeoutError: la connessione e' morta
                pass

    def _ricevi(self, conn: socket.socket) -> None:
        while True:
            testa = _leggi(conn, INTESTAZIONE.size)
            if testa is None:
                return  # il lettore si e' fermato: si aspetta il prossimo
            scatto, n_meta, n_jpeg = INTESTAZIONE.unpack(testa)
            testo, dati = _leggi(conn, n_meta), _leggi(conn, n_jpeg)
            if testo is None or dati is None:
                return
            img = cv2.imdecode(np.frombuffer(dati, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            meta = json.loads(testo)
            mano = np.array(meta["mano"]) if meta.get("mano") else None
            mondo = np.array(meta["mondo"]) if meta.get("mondo") else None
            gesti = tuple((nome, float(p)) for nome, p in meta.get("gesti", []))
            with self._lock:
                self._storia.append((scatto, Osservazione(gesti, mondo)))
                while self._storia and self._storia[0][0] < scatto - self.MEMORIA:
                    self._storia.popleft()
                self._numero += 1
                self._ultimo = Fotogramma(
                    img, scatto, self._numero, gesti, mano,
                    meta.get("ms_riconoscimento", 0.0), mondo,
                )


def _leggi(conn: socket.socket, n: int) -> bytes | None:
    """Esattamente ``n`` byte, o ``None`` se la connessione si chiude."""
    pezzi, resta = [], n
    while resta:
        pezzo = conn.recv(resta)
        if not pezzo:
            return None
        pezzi.append(pezzo)
        resta -= len(pezzo)
    return b"".join(pezzi)


def righe_piatte(img: np.ndarray) -> float:
    """Frazione di righe con tutti i pixel uguali: il riempimento dei dati persi.

    Non "righe nere": in YUYV i dati mancanti escono **verdi**, e un controllo
    sul nero li conta come immagine (successo, 2026-10-10).
    """
    return float((img.astype(int).std(axis=1).max(axis=1) <= 2).mean())
