"""Sola lettura del follower per il gemello: coppia spenta, posizioni su stdout.

    .venv/bin/python gemello_lettore.py --porta /dev/ttyACM0 | <chi le usa>

Il braccio diventa **morbido**: lo muovi tu con le mani, questo script legge e
basta. Nessun comando di posizione parte da qui.

Una riga JSON per lettura, in **gradi LeRobot** (pinza 0..100): la lingua del
driver. La conversione in radianti del modello sta nel progetto `robotica`
(`robotica/reale/so101.py`), perche' qui numpy e' 2 e la' e' 1 (D22): i due
processi si parlano con un tubo, non si importano.

    {"t": 12.345, "pos": {"shoulder_pan": -1.9, ...}, "lettura_ms": 8.1}

`lettura_ms` e' quanto costa **un** giro sul bus attraverso usbipd: la misura
della latenza, senza `max_relative_target` di mezzo.

**stdout e' il tubo**: ci va solo JSON. Tutto il resto va su stderr, che arriva
al terminale.

Lo zero dei gradi LeRobot e' il **centro del range calibrato**, (min+max)/2,
non la posa a meta' della calibrazione (`motors_bus.py`, `_normalize`).
"""

import argparse
import json
import sys
import time

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

# Gli stessi motori e gli stessi modi di SOFollower (so_follower.py): stessa
# lingua del driver che usera' l'esecutore.
MOTORI = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

CALIBRAZIONE = HF_LEROBOT_CALIBRATION / "robots" / "so_follower" / "follower.json"


def log(msg: str) -> None:
    print(f"[lettore] {msg}", file=sys.stderr, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--porta", required=True, help="porta del follower (da porte.sh)")
    ap.add_argument("--hz", type=float, default=30.0, help="letture al secondo volute")
    args = ap.parse_args()

    if not CALIBRAZIONE.is_file():
        log(f"calibrazione mancante: {CALIBRAZIONE}. Ricopiala da calibrazione/")
        return 1
    calib = {k: MotorCalibration(**v) for k, v in json.loads(CALIBRAZIONE.read_text()).items()}

    bus = FeetechMotorsBus(port=args.porta, motors=MOTORI, calibration=calib)
    bus.connect()
    # Gia' spenta dopo una teleoperazione, ma non lo si assume: e' l'unica
    # scrittura che questo script fa, e rende il braccio morbido.
    bus.disable_torque(num_retry=10)
    log(f"connesso su {args.porta}, coppia spenta: il braccio e' morbido, muovilo a mano")

    periodo = 1.0 / args.hz
    t_zero = time.perf_counter()
    n, somma_ms, peggiore_ms, t_report = 0, 0.0, 0.0, t_zero
    try:
        while True:
            t0 = time.perf_counter()
            pos = bus.sync_read("Present_Position", num_retry=10)
            lettura_ms = (time.perf_counter() - t0) * 1000
            riga = {
                "t": round(t0 - t_zero, 4),
                "pos": {k: round(float(v), 3) for k, v in pos.items()},
                "lettura_ms": round(lettura_ms, 2),
            }
            sys.stdout.write(json.dumps(riga) + "\n")
            sys.stdout.flush()

            n += 1
            somma_ms += lettura_ms
            peggiore_ms = max(peggiore_ms, lettura_ms)
            if t0 - t_report >= 5.0:
                log(f"{n / (t0 - t_report):.1f} letture/s, bus medio {somma_ms / n:.1f} ms, "
                    f"peggiore {peggiore_ms:.1f} ms")
                n, somma_ms, peggiore_ms, t_report = 0, 0.0, 0.0, t0
            time.sleep(max(0.0, periodo - (time.perf_counter() - t0)))
    except (BrokenPipeError, KeyboardInterrupt):
        # BrokenPipe: il gemello ha chiuso la finestra. Non e' un errore.
        pass
    finally:
        bus.disconnect(disable_torque=True)
        log("disconnesso")
    return 0


if __name__ == "__main__":
    sys.exit(main())
