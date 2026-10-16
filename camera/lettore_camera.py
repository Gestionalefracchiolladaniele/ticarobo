"""Legge la camera del kit **su Windows**, riconosce la mano, e manda tutto a WSL.

    .venv\\Scripts\\python.exe lettore_camera.py --indice 1           con i gesti
    .venv\\Scripts\\python.exe lettore_camera.py --indice 1 --solo-immagini

Perche' su Windows: via usbipd la camera arriva a pezzi (14-23% del fotogramma,
poi si blocca) — i trasferimenti isocroni di una webcam non reggono il
passaggio a WSL. Il braccio si', perche' parla a pacchetti. Qui il driver e'
quello nativo: fotogrammi interi. Numeri: `robotica/esperimenti/so101/hardware.md`.

**La mano** la trova MediaPipe Gesture Recognizer (`modelli/gesture_recognizer.task`,
7,99 MB, Apache 2.0, nessun addestramento): 21 punti della mano e un gesto fra
quelli che conosce. Tre sono i nostri: `Closed_Fist`, `Open_Palm`, `Victory`.
Gira **qui** e non nel progetto: niente conflitto di numpy, e l'immagine non si
decomprime due volte. Cosa sia "sasso" lo decide il progetto, non questo file.

Il collegamento parte **da Windows verso `localhost`**: Windows inoltra gia'
quel traffico a WSL, e in questo verso il firewall non chiede niente. Dall'altra
parte ascolta `robotica/reale/camera.py`.

Un fotogramma sul filo: intestazione (ora dello scatto `time.time()` come
double, lunghezza del JSON, lunghezza del JPEG; big endian), poi il JSON, poi
il JPEG. Il JSON:

    {"gesti": [["Closed_Fist", 0.93], ...], "mano": [[x, y, z] x 21] | null,
     "mondo": [[x, y, z] x 21, metri] | null, "ms_riconoscimento": 31.2}

L'ora serve a sapere quanto e' **vecchio** un fotogramma quando lo si usa.
L'indice (0, 1, ...) e' l'ordine in cui Windows ha trovato le camere: puo'
cambiare se si ricollegano in un altro ordine. `trova_camera.py` lo dice.
"""

import argparse
import json
import math
import socket
import struct
import sys
import time
from pathlib import Path

import cv2

INTESTAZIONE = struct.Struct(">dII")
MODELLO = Path(__file__).parent / "modelli" / "gesture_recognizer.task"


def log(msg: str) -> None:
    print(f"[camera] {msg}", flush=True)


def apri_camera(indice: int, larghezza: int, altezza: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit(f"camera {indice} non si apre: e' collegata? un'altra app la sta usando?")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, larghezza)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altezza)
    log(f"camera {indice} aperta: {int(cap.get(3))}x{int(cap.get(4))}")
    return cap


class Riconoscitore:
    """MediaPipe in modalita' video: segue la mano da un fotogramma all'altro
    invece di cercarla ogni volta da zero."""

    def __init__(self) -> None:
        import mediapipe as mp  # solo se serve: e' l'import piu' lento
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        opzioni = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELLO)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self._rec = vision.GestureRecognizer.create_from_options(opzioni)
        self._ultimo_ms = -1
        log(f"riconoscitore pronto ({MODELLO.stat().st_size / 2**20:.2f} MB)")

    def __call__(self, img_bgr) -> dict:
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        immagine = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        # In modalita' video i tempi devono crescere sempre.
        ora_ms = max(int(time.monotonic() * 1000), self._ultimo_ms + 1)
        self._ultimo_ms = ora_ms
        res = self._rec.recognize_for_video(immagine, ora_ms)
        gesti, mano, mondo = [], None, None
        if res.gestures:
            gesti = [[c.category_name, round(c.score, 3)] for c in res.gestures[0]]
        if res.hand_landmarks:
            mano = [[round(p.x, 4), round(p.y, 4), round(p.z, 4)] for p in res.hand_landmarks[0]]
        # In metri, centrate sulla mano: la forma della mano, non dove sta
        # nell'immagine. Il progetto ci calcola le dita aperte (games/rps/mano.py).
        if res.hand_world_landmarks:
            mondo = [[round(p.x, 5), round(p.y, 5), round(p.z, 5)]
                     for p in res.hand_world_landmarks[0]]
        ms = (time.perf_counter() - t0) * 1000
        return {"gesti": gesti, "mano": mano, "mondo": mondo, "ms_riconoscimento": round(ms, 1)}


def collega(porta: int, scadenza: float = math.inf, host: str = "localhost") -> socket.socket:
    """Fino a quando WSL non ascolta, riprova: chi parte per primo non conta.
    ``scadenza`` (perf_counter): oltre, si smette di aspettare."""
    while time.perf_counter() < scadenza:
        try:
            s = socket.create_connection((host, porta), timeout=2)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            log(f"collegato a WSL su {host}:{porta}")
            return s
        except OSError:
            time.sleep(2)
    log("tempo scaduto, WSL non ascolta")
    sys.exit(0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--indice", type=int, default=1, help="numero della camera (trova_camera.py)")
    ap.add_argument("--porta", type=int, default=5600)
    # localhost passa dal ponte di Windows (wslrelay), che con molte connessioni
    # morte smette di inoltrare (2026-10-12): allora l IP di WSL, hostname -I.
    ap.add_argument("--host", default="localhost", help="dove ascolta WSL")
    ap.add_argument("--larghezza", type=int, default=640)
    ap.add_argument("--altezza", type=int, default=480)
    ap.add_argument("--qualita", type=int, default=85, help="JPEG, 0-100")
    ap.add_argument("--solo-immagini", action="store_true", help="senza riconoscere la mano")
    ap.add_argument("--secondi", type=float, default=0, help="0 = finche' non lo fermi")
    args = ap.parse_args()

    riconosci = None if args.solo_immagini else Riconoscitore()
    cap = apri_camera(args.indice, args.larghezza, args.altezza)
    t_inizio = t_report = time.perf_counter()
    scadenza = t_inizio + args.secondi if args.secondi else math.inf
    log("aspetto WSL...")
    sock = collega(args.porta, scadenza, args.host)
    parametri = [cv2.IMWRITE_JPEG_QUALITY, args.qualita]
    n, byte, ms_tot = 0, 0, 0.0
    try:
        while time.perf_counter() < scadenza:
            ok, img = cap.read()
            scatto = time.time()
            if not ok:
                log("la camera non da' fotogrammi")
                time.sleep(0.5)
                continue
            meta = riconosci(img) if riconosci else {"gesti": [], "mano": None}
            testo = json.dumps(meta).encode()
            ok, jpeg = cv2.imencode(".jpg", img, parametri)
            dati = jpeg.tobytes()
            try:
                sock.sendall(INTESTAZIONE.pack(scatto, len(testo), len(dati)) + testo + dati)
            except OSError:
                log("WSL non risponde piu': mi ricollego")
                sock.close()
                sock = collega(args.porta, scadenza, args.host)
                continue
            n, byte = n + 1, byte + len(dati)
            ms_tot += meta.get("ms_riconoscimento", 0.0)
            ora = time.perf_counter()
            if ora - t_report >= 5:
                log(f"{n / (ora - t_report):.1f} fps, {byte / n / 1024:.0f} kB a fotogramma, "
                    f"riconoscimento {ms_tot / n:.0f} ms")
                n, byte, ms_tot, t_report = 0, 0, 0.0, ora
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
