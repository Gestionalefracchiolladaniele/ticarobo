"""Episodi simulati per ACT, nello stesso formato LeRobot dei 50 veri.

Per ogni seme: l'episodio si prova **solo con la fisica** (secondi); se riesce
lo si rifa' identico con le immagini e lo si scrive. MuJoCo e' deterministico,
ma lo si controlla: se la seconda esecuzione non da' lo stesso esito e gli
stessi fotogrammi, l'episodio si scarta. Il render costa ~300 volte la fisica
(osmesa: 316 ms contro 0,7 a fotogramma): non lo si paga per episodi da buttare.

Gira dove ci sono **insieme** LeRobot 0.6.1 e MuJoCo 3.3.7 (numpy 2): un venv
suo (``lerobot\\sim``) o Kaggle. Il progetto (numpy 1) qui non serve: si
importa solo ``robotica.sim``.

    python scripts/genera_act_sim.py <cartella> --episodi 5 --seme 0
    python scripts/genera_act_sim.py <cartella> --prova-render --gpu

A blocchi (``--blocco 25``): ogni 25 episodi il dataset si chiude e se ne apre
un altro, ``<cartella>_b000``, ``_b001``... Un blocco chiuso e' un dataset
completo: un'interruzione perde al piu' quello aperto. ``--scadenza`` (epoch)
ferma prima di cominciare un episodio oltre quell'ora, e chiude puliti.

Perche' la prova di render: su Kaggle il log diceva "egl" e disegnava la CPU
(llvmpipe, 198 ms a fotogramma contro 1,3 della T4). "egl" nomina lo sportello,
non chi c'e' dietro: si chiede ``GL_RENDERER`` e si guarda un'immagine vera.

Video come i veri (``registra.ps1``): H.264 ultrafast, crf 23, in streaming.
La compressione lascia artefatti: devono essere gli stessi nei due mondi.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# Prima di importare MuJoCo, o non conta: senza schermo (Kaggle) si disegna con la GPU.
if "MUJOCO_GL" not in os.environ and sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from robotica.sim.so101_caramelle import costruisci  # noqa: E402
from robotica.sim.so101_esperto import FPS, Esperto  # noqa: E402
from robotica.sim.so101_rps import GIUNTI  # noqa: E402

COMPITO = "Metti le caramelle gialle nel bicchiere a sinistra e quelle celesti-bianche a destra"
CAMERA = "observation.images.polso"
NOMI = [f"{n}.pos" for n in GIUNTI]
CARATTERISTICHE = {
    "action": {"dtype": "float32", "shape": (6,), "names": NOMI},
    "observation.state": {"dtype": "float32", "shape": (6,), "names": NOMI},
    CAMERA: {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]},
}


def prova(seme: int) -> tuple[bool, int, list[str]]:
    """Solo fisica: (riuscito, fotogrammi, motivi)."""
    rng = np.random.default_rng(seme)
    m, d, ep = costruisci(rng)
    e = Esperto(m, d, ep, rng)
    try:
        ok = e.esegui()
    except ValueError as err:
        return False, len(e.registro.stato), [str(err)]
    return ok, len(e.registro.stato), e.esito


def registra(seme: int, dataset, fotogrammi_attesi: int) -> tuple[bool, dict]:
    rng = np.random.default_rng(seme)
    m, d, ep = costruisci(rng)
    e = Esperto(m, d, ep, rng)
    r = mujoco.Renderer(m, 480, 640)
    t_render = [0.0]

    def guarda(stato: np.ndarray, azione: np.ndarray) -> None:
        t = time.perf_counter()
        r.update_scene(d, camera="wrist_cam")
        img = r.render()
        t_render[0] += time.perf_counter() - t
        dataset.add_frame(
            {"observation.state": stato, "action": azione, CAMERA: img, "task": COMPITO}
        )

    e.guarda = guarda
    try:
        ok = e.esegui()
    finally:
        r.close()  # con osmesa un contesto lasciato aperto sporca il successivo
    n = len(e.registro.stato)
    info = {
        "seme": seme,
        "fotogrammi": n,
        "secondi": n / FPS,
        "caramelle": len(ep.caramelle),
        "render_ms": 1000 * t_render[0] / max(n, 1),
    }
    if not ok or n != fotogrammi_attesi:
        dataset.clear_episode_buffer()
        info["scartato"] = (
            f"non deterministico: riuscito {ok}, {n} fotogrammi invece di {fotogrammi_attesi}"
        )
        return False, info
    dataset.save_episode()
    return True, info


class _Basta(Exception):
    """Ferma l'esperto a meta' episodio: per la prova di render basta un fotogramma."""


def prova_render(cartella: Path, seme: int, gpu: bool, fotogramma: int = 600) -> None:
    """Chi disegna, quanto costa, e un'immagine vera della scena (camera sul tavolo).

    Esce con errore se l'immagine e' uniforme (nera o vuota) o, con ``gpu``,
    se il renderer non e' NVIDIA. Un render che non da' errore non prova niente.
    """
    from OpenGL import GL
    from PIL import Image

    rng = np.random.default_rng(seme)
    m, d, ep = costruisci(rng)
    e = Esperto(m, d, ep, rng)
    r = mujoco.Renderer(m, 480, 640)
    renderer = GL.glGetString(GL.GL_RENDERER).decode()
    tempi: list[float] = []
    immagine: list[np.ndarray] = []

    def guarda(stato: np.ndarray, azione: np.ndarray) -> None:
        t = time.perf_counter()
        r.update_scene(d, camera="wrist_cam")
        img = r.render()
        tempi.append(time.perf_counter() - t)
        if len(tempi) >= fotogramma:
            immagine.append(img)
            raise _Basta

    e.guarda = guarda
    try:
        e.esegui()
    except _Basta:
        pass
    finally:
        r.close()
    assert immagine, f"l'episodio e' finito prima del fotogramma {fotogramma}"
    img = immagine[0]
    cartella.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(cartella / "prova_render.png")
    ms = 1000 * float(np.median(tempi[10:]))
    print(
        f"GL_RENDERER: {renderer} | {ms:.1f} ms a fotogramma (mediana) | "
        f"media pixel {img.mean():.1f}, scarto {img.std():.1f} | {cartella / 'prova_render.png'}"
    )
    assert img.std() > 10, "immagine uniforme: nera o vuota, il render non disegna la scena"
    if gpu:
        assert "nvidia" in renderer.lower() or "tesla" in renderer.lower(), (
            f"chiesta la GPU, disegna {renderer}"
        )


def _nuovo_dataset(root: Path, repo: str):
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo,
        fps=FPS,
        features=CARATTERISTICHE,
        root=root,
        robot_type="so_follower",
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="ultrafast", crf=23),
        streaming_encoding=True,
        encoder_threads=1,
    )


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("cartella", type=Path)
    a.add_argument("--episodi", type=int, default=0)
    a.add_argument(
        "--seme", type=int, default=0, help="primo seme; ogni processo il suo intervallo"
    )
    a.add_argument("--repo", default="so101/caramelle_sim")
    a.add_argument("--blocco", type=int, default=0, help="episodi per blocco; 0 = uno solo")
    a.add_argument("--scadenza", type=float, default=0.0, help="epoch: oltre, non si comincia")
    a.add_argument("--prova-render", action="store_true")
    a.add_argument("--gpu", action="store_true", help="con --prova-render: esige NVIDIA")
    args = a.parse_args()

    if args.prova_render:
        prova_render(args.cartella, args.seme, args.gpu)
        return
    assert args.episodi > 0, "--episodi"

    def radice(k: int) -> Path:
        if not args.blocco:
            return args.cartella
        return args.cartella.with_name(f"{args.cartella.name}_b{k:03d}")

    # Il blocco si apre quando c'e' un episodio da scrivere e si chiude appena pieno:
    # mai un dataset aperto e vuoto da chiudere.
    blocco, nel_blocco, dataset = 0, 0, None
    registro = args.cartella.parent / f"{args.cartella.name}_generazione.jsonl"
    registro.parent.mkdir(parents=True, exist_ok=True)
    fatti, seme, t0 = 0, args.seme, time.perf_counter()
    with open(registro, "a") as log:
        while fatti < args.episodi:
            if args.scadenza and time.time() > args.scadenza:
                print(f"scadenza: fermo a {fatti} episodi", flush=True)
                break
            t = time.perf_counter()
            ok, n, motivi = prova(seme)
            riga: dict = {"seme": seme, "prova_s": round(time.perf_counter() - t, 1)}
            if ok:
                if dataset is None:
                    dataset = _nuovo_dataset(radice(blocco), args.repo)
                t = time.perf_counter()
                ok, info = registra(seme, dataset, n)
                riga |= info | {"registra_s": round(time.perf_counter() - t, 1)}
                fatti += ok
                nel_blocco += ok
                if args.blocco and nel_blocco == args.blocco:
                    dataset.finalize()
                    print(f"blocco {blocco} chiuso: {radice(blocco)}", flush=True)
                    blocco, nel_blocco, dataset = blocco + 1, 0, None
            else:
                riga["fallito"] = motivi
            log.write(json.dumps(riga) + "\n")
            log.flush()
            print(
                f"[{fatti}/{args.episodi}] seme {seme}: "
                f"{'scritto' if ok else 'scartato'} ({time.perf_counter() - t0:.0f} s)",
                flush=True,
            )
            seme += 1
    if dataset is not None:
        dataset.finalize()
    print(
        f"fatto: {fatti} episodi in {args.cartella}, "
        f"{time.perf_counter() - t0:.0f} s; registro {registro}"
    )


if __name__ == "__main__":
    main()
