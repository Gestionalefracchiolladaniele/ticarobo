"""Rollout di ACT senza pause: il blocco nuovo si calcola mentre il braccio esegue.

Col rollout normale (rollout.ps1) ACT guarda, calcola (braccio fermo), esegue
100 azioni, guarda, calcola (fermo)... Qui il lavoro e' diviso in due processi
di LeRobot (``lerobot.async_inference``):

    policy server   fa girare la rete (lanciato a parte da rollout_async.ps1)
    robot client    muove il braccio a 30 Hz e, quando ha consumato meta' del
                    blocco (chunk_size_threshold=0.5), manda la foto al server

Il blocco nuovo arriva mentre restano azioni da fare; le azioni che si
sovrappongono si mediano (weighted_average: 0,3 vecchia + 0,7 nuova).

Il client di LeRobot non ha una durata e alla fine fa solo ``disconnect()``:
la coppia sparisce dove si trova il braccio, e il braccio cade. Questo
programma aggiunge:
- una durata;
- alla fine, il ritorno lento alla posa di partenza, POI la coppia tolta;
- i numeri del ciclo: Hz reali e quanti giri senza azioni (= braccio fermo).

SENZA safety/: eccezione dichiarata in D26, come rollout.ps1.

    python rollout_async.py COM3 MODELLO [--camera 1] [--secondi 120] [--limite 15]
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
import winsound

import numpy as np

# Zittisce solo il lungo avviso di torchcodec (README): gli errori veri restano.
logging.getLogger("lerobot.utils.import_utils").setLevel(logging.ERROR)

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101FollowerConfig

COMPITO = "Metti le caramelle gialle nel bicchiere a sinistra e quelle celesti-bianche a destra"
RITORNO_S = 4.0  # durata del ritorno alla posa di partenza


class ClientMisurato(RobotClient):
    """Il ciclo di LeRobot, identico, piu' un registro giro per giro."""

    def control_loop(self, task: str, verbose: bool = False):
        self.start_barrier.wait()
        self.giri: list[tuple[float, bool]] = []  # (durata del giro, azione eseguita?)
        while self.running:
            inizio = time.perf_counter()
            eseguita = self.actions_available()
            if eseguita:
                self.control_loop_action(verbose)
            if self._ready_to_send_observation():
                self.control_loop_observation(task, verbose)
            self.giri.append((time.perf_counter() - inizio, eseguita))
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - inizio)))


def torna(robot, partenza: dict[str, float], secondi: float, hz: int = 30) -> None:
    """Dalla posa attuale a quella di partenza, a passi piccoli e regolari."""
    ora = robot.get_observation()
    da = np.array([ora[k] for k in partenza])
    a = np.array(list(partenza.values()))
    for s in np.linspace(0, 1, int(secondi * hz))[1:]:
        robot.send_action(dict(zip(partenza, da + s * (a - da), strict=True)))
        time.sleep(1 / hz)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("porta")
    ap.add_argument("modello")
    ap.add_argument("--camera", type=int, default=1)
    ap.add_argument("--secondi", type=float, default=120)
    ap.add_argument("--limite", type=float, default=15)
    ap.add_argument("--server", default="127.0.0.1:8080")
    args = ap.parse_args()

    cfg = RobotClientConfig(
        policy_type="act",
        pretrained_name_or_path=args.modello,
        robot=SO101FollowerConfig(
            port=args.porta, id="follower", max_relative_target=args.limite, num_read_retries=10,
            cameras={"polso": OpenCVCameraConfig(index_or_path=args.camera, width=640, height=480, fps=30)}),
        actions_per_chunk=100,          # come l'addestramento (chunk_size 100)
        chunk_size_threshold=0.5,       # a meta' blocco chiede il prossimo
        aggregate_fn_name="weighted_average",
        task=COMPITO,
        server_address=args.server,
        policy_device="cpu",
        fps=30,
    )
    client = ClientMisurato(cfg)        # qui si collega il braccio
    obs = client.robot.get_observation()
    partenza = {k: obs[k] for k in client.robot.action_features}

    # Il server puo' metterci un po' a partire (carica torch): si riprova.
    for _ in range(60):
        if client.start():
            break
        time.sleep(1)
    else:
        client.robot.disconnect()
        raise SystemExit("Server non raggiungibile: e' partito? (finestra a parte)")

    ricevitore = threading.Thread(target=client.receive_actions, daemon=True)
    ricevitore.start()
    ciclo = threading.Thread(target=client.control_loop, args=(COMPITO,), daemon=True)
    print(f"\n>>> Il braccio parte da solo. {args.secondi:.0f} s. Mano sull'interruttore dei 12 V. <<<\n")
    winsound.Beep(1000, 300)
    ciclo.start()
    inizio = time.perf_counter()
    try:
        while ciclo.is_alive() and time.perf_counter() - inizio < args.secondi:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nFermato a mano.")
    finally:
        client.shutdown_event.set()     # ferma ciclo e ricevitore
        ciclo.join(timeout=2)
        print("Ritorno alla posa di partenza...")
        try:
            torna(client.robot, partenza, RITORNO_S)
        finally:
            winsound.Beep(600, 700)
            client.robot.disconnect()   # solo ora la coppia sparisce
            client.channel.close()

    giri = getattr(client, "giri", [])
    fatte = np.array([g[1] for g in giri], dtype=bool)
    if fatte.any():
        # Prima del primo blocco il braccio aspetta comunque: non e' una pausa fra blocchi.
        primo = int(np.argmax(fatte))
        print(f"\nAttesa del primo blocco: {primo / 30:.1f} s")
        durate = np.array([g[0] for g in giri])[primo:]
        fatte = fatte[primo:]
        tempo = len(fatte) / 30
        print(f"=== dal primo blocco: {len(fatte)} giri (~{tempo:.0f} s) ===")
        print(f"Azioni eseguite: {fatte.sum()} ({fatte.sum() / tempo:.1f} al secondo, obiettivo 30)")
        print(f"Giri SENZA azioni (braccio fermo): {(~fatte).sum()} "
              f"= {(~fatte).sum() / 30:.1f} s a 30 Hz")
        print(f"Giro medio {durate.mean() * 1000:.0f} ms, peggiore {durate.max() * 1000:.0f} ms")
        coda = client.action_queue_size
        if coda:
            print(f"Coda: minima {min(coda)}, media {np.mean(coda):.0f} (0 = pausa)")


if __name__ == "__main__":
    main()
