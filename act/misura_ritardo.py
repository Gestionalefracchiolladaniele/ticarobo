"""Da dove viene il ritardo della teleoperazione: PC, limite di gradi o motori.

Fa la stessa teleoperazione di teleop.ps1 (stessa calibrazione, stesso limite)
e registra a ogni giro tre posizioni per giunto:

    comando  = dove il leader dice di andare
    mandato  = cio' che arriva al follower dopo il taglio di max_relative_target
    letto    = dove il follower e' davvero

Ritardo comando->mandato = colpa del limite. Ritardo mandato->letto = colpa
dei motori (P morbido). Hz del ciclo e ms per fase = colpa del PC.

    python misura_ritardo.py COM3 COM4 --limite 15 [--camera 1] [--secondi 30]

Bip: uno all'inizio (muovi), due a 20 s (ripiega il leader in basso), uno
lungo alla fine (il follower perde la coppia e si appoggia).
"""

from __future__ import annotations

import argparse
import logging
import time
import winsound

import numpy as np

# Zittisce solo il lungo avviso di torchcodec (README): gli errori veri restano.
logging.getLogger("lerobot.utils.import_utils").setLevel(logging.ERROR)

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

HZ = 60  # come lerobot-teleoperate
GIUNTI = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def ritardo_s(a: np.ndarray, b: np.ndarray, t: np.ndarray, max_s: float = 1.5) -> float | None:
    """Di quanto b insegue a: lo spostamento che le sovrappone meglio.

    Si confrontano le velocita', non le posizioni: un giunto che si sposta e
    resta li' sovrapporrebbe qualunque spostamento. Ricampionate a passo
    fisso, perche' i giri del ciclo non durano tutti uguale.
    """
    passo = 0.005
    tt = np.arange(t[0], t[-1], passo)
    va = np.diff(np.interp(tt, t, a))
    vb = np.diff(np.interp(tt, t, b))
    if np.std(va) < 1e-3:
        return None  # giunto mai mosso
    errori = [np.mean((va[: len(va) - k] - vb[k:]) ** 2) for k in range(int(max_s / passo))]
    return int(np.argmin(errori)) * passo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("follower")
    ap.add_argument("leader")
    ap.add_argument("--limite", type=float, default=15)
    ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--secondi", type=float, default=30)
    args = ap.parse_args()

    camere = {}
    if args.camera is not None:
        camere = {"polso": OpenCVCameraConfig(index_or_path=args.camera, width=640, height=480, fps=30)}
    robot = SO101Follower(SO101FollowerConfig(
        port=args.follower, id="follower", max_relative_target=args.limite,
        num_read_retries=10, cameras=camere))
    leader = SO101Leader(SO101LeaderConfig(port=args.leader, id="leader", num_read_retries=10))
    robot.connect()
    leader.connect()

    t, comando, mandato, letto, fasi = [], [], [], [], []
    input("\nBracci collegati. Mano sul leader, poi premi INVIO per iniziare... ")
    print("\n>>> MUOVI IL LEADER: movimenti ampi (base, spalla, gomito) <<<")
    winsound.Beep(1000, 300)
    inizio = time.perf_counter()
    avvisato = False
    ultimo_secondo = -1
    try:
        while (ora := time.perf_counter()) - inizio < args.secondi:
            mancano = int(args.secondi - (ora - inizio))
            if mancano != ultimo_secondo:
                ultimo_secondo = mancano
                print(f"   ancora {mancano} s", end="\r", flush=True)
            if not avvisato and ora - inizio > args.secondi - 10:
                print("\n>>> RIPIEGA IL LEADER IN BASSO e lascialo fermo <<<")
                winsound.Beep(800, 150)
                winsound.Beep(800, 150)
                avvisato = True
            t0 = time.perf_counter()
            obs = robot.get_observation()          # posizioni + camera
            t1 = time.perf_counter()
            azione = leader.get_action()
            t2 = time.perf_counter()
            inviato = robot.send_action(azione)    # rilegge e taglia, poi scrive
            t3 = time.perf_counter()
            t.append(t0 - inizio)
            letto.append([obs[f"{g}.pos"] for g in GIUNTI])
            comando.append([azione[f"{g}.pos"] for g in GIUNTI])
            mandato.append([inviato[f"{g}.pos"] for g in GIUNTI])
            fasi.append((t1 - t0, t2 - t1, t3 - t2))
            resto = 1 / HZ - (time.perf_counter() - t0)
            if resto > 0:
                time.sleep(resto)
    finally:
        winsound.Beep(600, 700)
        leader.disconnect()
        robot.disconnect()

    t = np.array(t)
    comando, mandato, letto = np.array(comando), np.array(mandato), np.array(letto)
    fasi = np.array(fasi) * 1000
    giri = np.diff(t)

    print(f"\n=== limite {args.limite} gradi, camera {'si' if camere else 'no'}, {len(t)} giri ===")
    print(f"Ciclo: {1 / giri.mean():.1f} Hz (obiettivo {HZ}), giro medio {giri.mean() * 1000:.1f} ms, "
          f"peggiore {giri.max() * 1000:.0f} ms")
    for nome, col in zip(("lettura follower+camera", "lettura leader", "taglio+scrittura"), fasi.T):
        print(f"  {nome:24s} media {col.mean():5.1f} ms   peggiore {col.max():5.0f} ms")
    print("Ritardo per giunto (s):   limite (comando->mandato)   motori (mandato->letto)   totale")
    for i, g in enumerate(GIUNTI):
        r1 = ritardo_s(comando[:, i], mandato[:, i], t)
        r2 = ritardo_s(mandato[:, i], letto[:, i], t)
        r3 = ritardo_s(comando[:, i], letto[:, i], t)
        f = lambda r: "   -  " if r is None else f"{r:6.3f}"
        print(f"  {g:14s}          {f(r1)}                    {f(r2)}             {f(r3)}")
    scarto = np.abs(comando - letto).max(axis=0)
    print("Scarto massimo comando-letto (gradi): "
          + ", ".join(f"{g} {s:.0f}" for g, s in zip(GIUNTI, scarto)))


if __name__ == "__main__":
    main()
