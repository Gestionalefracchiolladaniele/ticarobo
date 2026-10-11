"""Quale numero ha la camera del kit per Windows: una foto da ogni indice.

OpenCV su Windows (DirectShow) chiama le camere per numero, non per nome, e
qui ce ne sono due: quella del kit e quella del portatile.
"""

import sys
from pathlib import Path

import cv2

uscita = Path(sys.argv[1])
uscita.mkdir(parents=True, exist_ok=True)
for indice in range(3):
    cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"indice {indice}: nessuna camera")
        continue
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(15):  # esposizione automatica
        ok, img = cap.read()
    if ok:
        piatte = (img.astype(int).std(axis=1).max(axis=1) <= 2).mean()
        cv2.imwrite(str(uscita / f"indice_{indice}.png"), img)
        print(f"indice {indice}: {img.shape[1]}x{img.shape[0]}, righe piatte {piatte:.0%}")
    else:
        print(f"indice {indice}: si apre ma non da' fotogrammi")
    cap.release()
