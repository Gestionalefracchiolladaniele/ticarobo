"""Genera act_caramelle.ipynb: addestramento ACT su Kaggle (GPU T4), LeRobot 0.6.1.

Il notebook si scrive da qui perche' un .ipynb a mano e' JSON fragile.

    python scrivi_notebook.py                                     run A (qui)
    python scrivi_notebook.py --passi 60000 --salva 10000 --zip 30000 60000 \\
        --cartella ..\\kaggle_50                                   run B

Passi: il minimo degli autori di ACT e' 5000 epoche *loro* (un campione per
episodio, utils.py di tonyzhaozh/act) = 5000 x episodi / batch.
- Run del 15/10: 5 episodi, 20k.
- Run A (16/10): 35 episodi, minimo ~21,9k -> 25k (fine entro le 17:35).
- Run B (16/10): 50 episodi, minimo ~31k -> 60k, zip a 30k e 60k. Learning
  rate costante (get_scheduler_preset -> None): il checkpoint a 30k vale una
  run da 30k, e costa una run sola.
"""

import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--passi", type=int, default=25_000)
ap.add_argument("--salva", type=int, default=5_000, help="checkpoint ogni N passi")
ap.add_argument("--zip", type=int, nargs="*", default=[], help="passi da zippare (vuoto: l'ultimo)")
ap.add_argument("--cartella", type=Path, default=Path(__file__).parent)
args = ap.parse_args()
PASSI = args.passi
ZIP = args.zip or [PASSI]

celle = [
    ("md", """# ACT sul SO-101 — smistare caramelle

Impostazioni (gia' in `kernel-metadata.json`, lanciato con `kaggle kernels push`):
- **Accelerator: GPU T4 x2** (`machine_shape: NvidiaTeslaT4`). `lerobot-train`
  ne usa **una**: per due servirebbe `accelerate launch`, mai provato qui
- **Internet: On** (serve il telefono verificato sull'account)
- **Input**: il dataset privato in `dataset_sources` di `kernel-metadata.json`

Poi **Save Version → Save & Run All**: gira da solo, puoi chiudere il browser.
Il modello finito e' in **Output** → `modello_act.zip`."""),
    ("md", "## 1. La GPU c'e'?"),
    ("code", "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    ("md", """## 2. Il dataset

Kaggle lo monta in sola lettura sotto `/kaggle/input/`: lo si cerca dal file
`meta/info.json` (non dipende dal nome scelto al caricamento) e lo si copia in
`/tmp`, fuori dall'output (che Kaggle salva e fa scaricare)."""),
    ("code", """import glob, json, shutil, os
trovati = glob.glob('/kaggle/input/**/meta/info.json', recursive=True)
assert trovati, 'Dataset non trovato: e\\' aggiunto in Input?'
sorgente = os.path.dirname(os.path.dirname(trovati[0]))
shutil.copytree(sorgente, '/tmp/caramelle', dirs_exist_ok=True)
info = json.load(open('/tmp/caramelle/meta/info.json'))
print(sorgente, '->', '/tmp/caramelle')
print('episodi', info['total_episodes'], 'fotogrammi', info['total_frames'], 'fps', info['fps'])"""),
    ("md", """## 3. LeRobot 0.6.1

Vuole Python >= 3.12, e Kaggle puo' averne uno piu' vecchio: `uv` crea un
ambiente 3.12 a parte in `/tmp` (fuori dall'output: torch pesa GB), con torch
per la GPU che trova (`--torch-backend=auto`). Stessa versione del portatile."""),
    ("code", """!pip install -q uv
!UV_CACHE_DIR=/tmp/uvcache uv venv -q -p 3.12 /tmp/venv
!UV_CACHE_DIR=/tmp/uvcache uv pip install -q --python /tmp/venv/bin/python --torch-backend=auto "lerobot[training]==0.6.1"
!/tmp/venv/bin/python -c "import lerobot, torch; print('lerobot', lerobot.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())\""""),
    ("md", f"""## 4. Addestramento

- `--steps={PASSI}`: il minimo consigliato dagli autori di ACT per questi episodi
  (default LeRobot 100k)
- `--dataset.image_transforms.enable=true`: luminosita', contrasto, colore
  cambiati a caso, la rete non si lega alla luce della stanza
- `--policy.push_to_hub=false`: niente account Hugging Face
- ogni 200 passi stampa la `loss` (deve scendere) e i secondi per passo:
  da li' il tempo totale"""),
    ("code", f"""!/tmp/venv/bin/lerobot-train \\
    --dataset.repo_id=so101/caramelle \\
    --dataset.root=/tmp/caramelle \\
    --dataset.image_transforms.enable=true \\
    --policy.type=act \\
    --policy.device=cuda \\
    --policy.push_to_hub=false \\
    --output_dir=/kaggle/working/act_caramelle \\
    --job_name=act_caramelle \\
    --steps={PASSI} \\
    --save_freq={args.salva} \\
    --batch_size=8 \\
    --wandb.enable=false"""),
    ("md", f"""## 5. I modelli da scaricare

Solo i pesi (`pretrained_model`) dei checkpoint a {ZIP} passi, uno zip
ciascuno (`modello_act_<passi>.zip`). Le cartelle si cercano per numero, non
per nome: LeRobot le numera con gli zeri davanti. Gli altri checkpoint restano
nell'output nel caso servano."""),
    ("code", f"""import shutil, os
base = '/kaggle/working/act_caramelle/checkpoints'
numeri = {{int(n): n for n in os.listdir(base) if n.isdigit()}}
print('checkpoint:', sorted(numeri))
for passi in {ZIP}:
    cartella = os.path.join(base, numeri[passi], 'pretrained_model')
    uscita = shutil.make_archive(f'/kaggle/working/modello_act_{{passi}}', 'zip', cartella)
    print('pronto:', uscita, os.path.getsize(uscita) / 1e6, 'MB')"""),
]


def cella(tipo: str, testo: str) -> dict:
    righe = testo.splitlines(keepends=True)
    if tipo == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": righe}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": righe}


nb = {
    "cells": [cella(t, s) for t, s in celle],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "gpu", "isInternetEnabled": True},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
out = args.cartella / "act_caramelle.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("scritto", out)
