"""Rendering offscreen delle camere.

Su questa macchina (GPU integrata, WSL) il rendering usa **OSMesa**, cioe'
software: ~2-5 fps a 640x480. Va bene per guardare cosa sta succedendo, non
per generare dataset. Per quello serve una GPU (es. Modal).

``MUJOCO_GL=osmesa`` va impostato **prima** di importare mujoco, per questo
la variabile si imposta qui in cima al modulo.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from robotica.paths import OUT  # noqa: E402

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


class CameraRenderer:
    """Rende immagini RGB e mappe di profondita' da una camera della scena.

    Il renderer e' costoso da creare e va riusato: crearne uno per frame
    rende il rendering diverse volte piu' lento.
    """

    def __init__(
        self, model: mujoco.MjModel, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT
    ) -> None:
        self.model = model
        self.width = width
        self.height = height
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def rgb(self, data: mujoco.MjData, camera: str = "overhead") -> np.ndarray:
        """Immagine RGB come array uint8 (H, W, 3)."""
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(data, camera=camera)
        return self._renderer.render()

    def depth(self, data: mujoco.MjData, camera: str = "overhead") -> np.ndarray:
        """Mappa di profondita' in metri, come array float32 (H, W).

        La depth trasferisce al reale molto meglio della RGB: una camera
        stereo o ToF misura distanze, e le distanze in simulazione sono
        corrette. E' il motivo per cui conviene appoggiarsi alla geometria
        piuttosto che ai colori.
        """
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(data, camera=camera)
        out = self._renderer.render()
        self._renderer.disable_depth_rendering()
        return out

    def body_mask(
        self,
        data: mujoco.MjData,
        prefixes: tuple[str, ...],
        camera: str = "overhead",
    ) -> np.ndarray:
        """Maschera ``True`` sui pixel coperti da corpi il cui nome inizia per
        uno dei ``prefixes``.

        Serve al **self-filtering**: il braccio del robot sporge dal tavolo
        esattamente dove stanno gli oggetti — la pinza a riposo sta a 211 mm sul
        piano — e nessun volume di lavoro puo' separarli, perche' e' li' che il
        braccio lavora.

        **Sul transfer, per non lasciarlo implicito.** Qui la maschera viene dal
        renderer, che sa quale geom copre ogni pixel: e' informazione che una
        camera vera non ha. Ma l'informazione *che se ne estrae* — "questi pixel
        sono il mio braccio" — sul robot reale si ottiene lo stesso,
        proiettando il modello del braccio nell'immagine a partire dalla
        cinematica, che e' nota. **Cambia chi la calcola, non se esiste.**

        Il riconoscimento e' per **prefisso del nome**, come ``obstacle_`` in
        ``safety/path.py`` e ``cube_``/``obj_`` in ``checks.py``: una convenzione
        nel nome evita una tabella da tenere aggiornata a mano, che prima o poi
        diverge dal modello.
        """
        self._renderer.enable_segmentation_rendering()
        self._renderer.update_scene(data, camera=camera)
        seg = self._renderer.render()
        self._renderer.disable_segmentation_rendering()

        # Canale 0 = id del geom; -1 dove non c'e' nulla.
        geom_ids = seg[:, :, 0]
        maschera = np.zeros(geom_ids.shape, dtype=bool)

        for gid in np.unique(geom_ids):
            if gid < 0 or gid >= self.model.ngeom:
                continue
            bid = int(self.model.geom_bodyid[int(gid)])
            nome = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if nome and nome.startswith(prefixes):
                maschera |= geom_ids == gid

        return maschera

    def save(
        self, data: mujoco.MjData, filename: str, camera: str = "overhead"
    ) -> str:
        """Salva un'immagine RGB in ``out/`` e restituisce il percorso."""
        from PIL import Image

        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / filename
        Image.fromarray(self.rgb(data, camera)).save(path)
        return str(path)

    def close(self) -> None:
        self._renderer.close()

    def __enter__(self) -> CameraRenderer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def camera_pose(model: mujoco.MjModel, data: mujoco.MjData, camera: str) -> np.ndarray:
    """Pose 4x4 della camera nel frame mondo.

    In simulazione e' nota esattamente. Sul robot reale va **calibrata**: e' la
    calibrazione mano-occhio, e sbagliarla e' la causa numero uno di robot che
    mancano il bersaglio di qualche centimetro senza motivo apparente.
    """
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cid < 0:
        raise KeyError(f"camera '{camera}' non trovata")
    T = np.eye(4)
    T[:3, :3] = data.cam_xmat[cid].reshape(3, 3)
    T[:3, 3] = data.cam_xpos[cid]
    return T


def intrinsics(model: mujoco.MjModel, camera: str, width: int, height: int) -> np.ndarray:
    """Matrice intrinseca K (3x3) della camera.

    Serve a convertire fra pixel e raggi nello spazio. Sul robot reale si
    ottiene dalla calibrazione con scacchiera, o dal driver della camera.
    """
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cid < 0:
        raise KeyError(f"camera '{camera}' non trovata")
    fovy = float(np.radians(model.cam_fovy[cid]))
    f = 0.5 * height / np.tan(fovy / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=float)
