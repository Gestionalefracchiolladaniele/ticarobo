"""Scrive il notebook di diagnosi del render su Kaggle (chi disegna davvero?).

    python so101-egl-diagnosi.ipynb.py
"""

import json
from pathlib import Path

QUI = Path(__file__).parent
NOME = "so101-egl-diagnosi"

MISURA = r'''
import os, sys, time, glob
sys.path.insert(0, glob.glob('/kaggle/input/**/robotica/src', recursive=True)[0])
import mujoco, numpy as np
from robotica.sim.so101_caramelle import costruisci
m, d, ep = costruisci(np.random.default_rng(3))
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 480, 640)
from OpenGL import GL
print("GL_RENDERER:", GL.glGetString(GL.GL_RENDERER), "| GL_VENDOR:", GL.glGetString(GL.GL_VENDOR))
for nome, ombre in (("ombre accese", True), ("ombre spente", False)):
    r.update_scene(d, camera="wrist_cam")
    r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = ombre
    r.render()
    t = time.perf_counter()
    for _ in range(30):
        r.update_scene(d, camera="wrist_cam")
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = ombre
        img = r.render()
    print(f"{nome}: {1000 * (time.perf_counter() - t) / 30:.1f} ms a fotogramma, media pixel {img.mean():.1f}")
    from PIL import Image
    Image.fromarray(img).save(f"/kaggle/working/{os.environ.get('ETICHETTA', 'prova')}_{'ombre' if ombre else 'senza'}.png")
r.close()
'''

celle = [
    ("code", "!nvidia-smi --query-gpu=name,driver_version --format=csv; nproc"),
    ("code", "!ls /usr/lib/x86_64-linux-gnu/ | grep -i -E 'nvidia|libEGL' | head -30; ls /usr/share/glvnd/egl_vendor.d/ 2>&1; ls /usr/lib/x86_64-linux-gnu/libnvidia* 2>/dev/null | head"),
    ("code", """!pip install -q uv
!UV_CACHE_DIR=/tmp/uvcache uv venv -q -p 3.12 /tmp/venv
!UV_CACHE_DIR=/tmp/uvcache uv pip install -q --python /tmp/venv/bin/python "mujoco==3.3.7" scipy pyopengl pillow"""),
    ("code", f"open('/tmp/misura.py', 'w').write({MISURA!r})\n!ETICHETTA=base MUJOCO_GL=egl PYOPENGL_PLATFORM=egl /tmp/venv/bin/python /tmp/misura.py 2>&1 | tail -15"),
    ("code", """# Piano A: le librerie GL di NVIDIA da apt. Valgono solo se la versione coincide col driver.
import subprocess
ver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True).stdout.split()[0]
major = ver.split('.')[0]
print('driver', ver)
subprocess.run('apt-get -qq update > /dev/null 2>&1', shell=True)
for pacchetto in (f'libnvidia-gl-{major}', f'libnvidia-gl-{major}-server'):
    print(subprocess.run(f'apt-cache policy {pacchetto} | head -3', shell=True, capture_output=True, text=True).stdout)
    r = subprocess.run(f'apt-get -qq install -y --no-install-recommends {pacchetto}', shell=True, capture_output=True, text=True)
    print(pacchetto, 'esito', r.returncode, r.stderr[-300:])
    if r.returncode == 0:
        break
!ls /usr/share/glvnd/egl_vendor.d/; ls /usr/lib/x86_64-linux-gnu/ | grep -i -E 'libEGL_nvidia|libnvidia-eglcore'"""),
    ("code", "!ETICHETTA=apt MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json /tmp/venv/bin/python /tmp/misura.py 2>&1 | tail -15"),
    ("code", """# Piano B: dall'installer NVIDIA della versione ESATTA del driver, estratto e non installato.
# Solo le librerie GL/EGL, in una cartella privata: il sistema non si tocca.
!cd /tmp && (wget -q https://us.download.nvidia.com/tesla/{ver}/NVIDIA-Linux-x86_64-{ver}.run -O nv.run || wget -q https://us.download.nvidia.com/XFree86/Linux-x86_64/{ver}/NVIDIA-Linux-x86_64-{ver}.run -O nv.run); ls -la /tmp/nv.run
!cd /tmp && sh nv.run --extract-only --target /tmp/nvx > /dev/null && mkdir -p /tmp/nvlib && cp /tmp/nvx/libEGL_nvidia.so.{ver} /tmp/nvx/libnvidia-eglcore.so.{ver} /tmp/nvx/libnvidia-glsi.so.{ver} /tmp/nvlib/ && (cp /tmp/nvx/libnvidia-gpucomp.so.{ver} /tmp/nvlib/ 2>/dev/null; true) && ln -sf libEGL_nvidia.so.{ver} /tmp/nvlib/libEGL_nvidia.so.0 && ls /tmp/nvlib
open('/tmp/nvlib/10_nvidia.json', 'w').write('{"file_format_version": "1.0.0", "ICD": {"library_path": "/tmp/nvlib/libEGL_nvidia.so.0"}}')"""),
    ("code", "!ETICHETTA=run LD_LIBRARY_PATH=/tmp/nvlib:$$LD_LIBRARY_PATH MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES=/tmp/nvlib/10_nvidia.json /tmp/venv/bin/python /tmp/misura.py 2>&1 | tail -15"),
]


def cella(tipo: str, testo: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": testo.splitlines(keepends=True)}


nb = {"cells": [cella(t, s) for t, s in celle],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}},
      "nbformat": 4, "nbformat_minor": 4}
(QUI / f"{NOME}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
(QUI / "kernel-metadata.json").write_text(json.dumps({
    "id": f"danielefracchiolla/{NOME}", "title": NOME, "code_file": f"{NOME}.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4", "dataset_sources": ["danielefracchiolla/so101-sim-codice"],
    "competition_sources": [], "kernel_sources": [], "model_sources": []}, indent=2), encoding="utf-8")
print("scritto")
