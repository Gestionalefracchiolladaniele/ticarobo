"""Da pixel a punti nel mondo: la geometria della camera.

E' il passaggio che rende utile un'immagine. Finche' si hanno pixel, il robot
non puo' andarci: un pixel e' una **direzione**, non un luogo.

Un punto dell'immagine da solo non basta, perche' tutti i punti allineati sul
raggio che parte dalla camera finiscono nello stesso pixel — un cubo vicino e
una parete lontana sono indistinguibili. La **depth** fornisce il pezzo
mancante: direzione piu' distanza fanno un punto unico.

E' anche il motivo per cui questa strada trasferisce al reale (D18): una camera
stereo o ToF **misura** quella distanza. Con la sola RGB andrebbe indovinata, ed
e' li' che i modelli sbagliano.

**Convenzioni, e perche' contano.** MuJoCo e la visione artificiale usano assi
opposti:

===================  ==================  ========================
                     MuJoCo (camera)     visione artificiale
===================  ==================  ========================
asse ottico          **-Z** in avanti    +Z in avanti
asse verticale       **+Y** in alto      +Y in basso
===================  ==================  ========================

Da qui i due segni meno in ``deproject`` e ``project``. Non sono aggiustamenti
empirici: sono la traduzione fra le due convenzioni. Un errore di frame non si
manifesta come errore — si manifesta come un robot che manca il bersaglio di
qualche centimetro con tutto che sembra funzionare (D16).
"""

from __future__ import annotations

import numpy as np

from robotica.transforms import Pose, inverse

__all__ = [
    "deproject",
    "deproject_many",
    "project",
]


def deproject(
    u: float,
    v: float,
    depth: float,
    K: np.ndarray,
    T_world_camera: Pose,
) -> np.ndarray:
    """Dal pixel ``(u, v)`` con la sua ``depth`` al punto 3D nel **mondo**.

    Args:
        u: colonna del pixel, da sinistra.
        v: riga del pixel, dall'alto.
        depth: distanza lungo l'asse ottico, in **metri** (non la distanza
            euclidea dal centro ottico: e' la componente lungo -Z, che e'
            quello che restituisce ``CameraRenderer.depth``).
        K: matrice intrinseca 3x3, da ``render.intrinsics``.
        T_world_camera: posa della camera nel mondo, da ``render.camera_pose``.
            **E' l'unico termine che sul robot vero non si conosce**: va
            calibrato. In simulazione e' esatto, ed e' cio' che permette di
            verificare una calibrazione invece di doversene fidare.

    Returns:
        Il punto ``(x, y, z)`` nel frame mondo.

    La ``depth`` non si ricava dall'immagine: si **legge**. Stimarla dalla
    dimensione apparente di un oggetto noto funziona finche' l'oggetto e'
    davvero quello che si crede.
    """
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # Dal pixel al raggio: (u-cx)/fx e' la tangente dell'angolo orizzontale.
    # Il segno meno su y traduce "v cresce verso il basso" (immagine) in
    # "+Y e' in alto" (MuJoCo).
    x_cam = (u - cx) / fx * depth
    y_cam = -(v - cy) / fy * depth
    z_cam = -depth  # -Z e' la direzione in cui la camera guarda

    p_cam = np.array([x_cam, y_cam, z_cam, 1.0], dtype=float)
    return (np.asarray(T_world_camera, dtype=float) @ p_cam)[:3]


def deproject_many(
    uv: np.ndarray,
    depth_map: np.ndarray,
    K: np.ndarray,
    T_world_camera: Pose,
) -> np.ndarray:
    """``deproject`` su molti pixel insieme.

    Args:
        uv: array ``(N, 2)`` di coppie ``(u, v)`` intere.
        depth_map: la mappa di profondita' completa ``(H, W)``.

    Returns:
        Array ``(N, 3)`` di punti nel mondo.

    Esiste separato perche' su 2 core la differenza fra un ciclo Python e
    un'operazione vettoriale su qualche migliaio di pixel e' fra decimi di
    secondo e secondi — e la percezione gira dentro un ciclo che ha gia' un
    budget stretto (D17: 19 s per ciclo di presa).
    """
    uv = np.asarray(uv, dtype=int)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv deve essere (N, 2), ricevuto {uv.shape}")

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u = uv[:, 0].astype(float)
    v = uv[:, 1].astype(float)
    d = depth_map[uv[:, 1], uv[:, 0]].astype(float)

    p_cam = np.stack(
        [(u - cx) / fx * d, -(v - cy) / fy * d, -d, np.ones_like(d)], axis=1
    )
    return (np.asarray(T_world_camera, dtype=float) @ p_cam.T).T[:, :3]


def project(
    point_world: np.ndarray,
    K: np.ndarray,
    T_world_camera: Pose,
) -> tuple[float, float, float]:
    """L'inverso: da un punto del mondo al pixel in cui appare.

    Returns:
        ``(u, v, depth)``. La ``depth`` e' negativa se il punto sta **dietro**
        la camera, caso in cui ``(u, v)`` non ha significato: un punto alle
        spalle proietta comunque dei numeri, e usarli senza controllare il
        segno e' un modo silenzioso di sbagliare.

    Serve a **verificare**: noto dove sta un oggetto, si controlla che appaia
    dove la geometria dice. E' il controllo che ha rivelato una camera rivolta
    al soffitto quando il conteggio dei pixel non neri diceva 100%.
    """
    p_cam = (inverse(np.asarray(T_world_camera, dtype=float)) @ np.append(point_world, 1.0))[:3]
    depth = -float(p_cam[2])
    if abs(depth) < 1e-9:
        raise ValueError("punto sul piano della camera: la proiezione non esiste")

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = fx * (p_cam[0] / depth) + cx
    v = -fy * (p_cam[1] / depth) + cy
    return float(u), float(v), depth
