"""Scrive i notebook Kaggle del co-training (run C e D).

    python scrivi_notebook.py

- ``gen\\``: genera gli episodi simulati (GPU T4: EGL di NVIDIA, installato e
  verificato; senza, EGL disegna sulla CPU), 4 processi a blocchi, poi un
  merge solo -> output ``caramelle_sim``.
- ``run_c\\``: 50 veri x4 + simulati mescolati dall'inizio, ACT.
- ``run_d1\\``: solo simulati. ``run_d2\\``: riparte da D1, stessi dati di C.
  C contro D: conta *come* si usano i simulati, a parita' di dati.

Perche' i veri x4: LeRobot 0.6.1 non pesa due dataset (``MultiLeRobotDataset``
disattivato), pesca a caso i fotogrammi; 93k veri contro 844k simulati
farebbero 10% di veri. A parita' di passi, ogni simulato in piu' toglie
esposizione ai veri (run B: ogni fotogramma vero visto 5,1 volte; x2: 1,6;
x4: 2,6, ~31% veri, vicino all'unico ACT su SO-101 con numeri). Perche' copie:
lo stesso dataset nominato due volte nel merge si rompe, e ``caramelle_50`` ha
``meta/episodes/file_index`` sbagliati (``scripts/ripara_meta_episodi.py``).
"""

import json
from pathlib import Path

QUI = Path(__file__).parent
UTENTE = "danielefracchiolla"
EPISODI = 500
PROCESSI = 4
PASSI, SALVA = 100_000, 25_000  # run C
COPIE_VERI = 4  # C e D2: ~31% veri (ricerca del 18/10 in act.md)
D1_PASSI, D1_SALVA = 60_000, 20_000  # D fase 1, solo simulati (come so101_act_cotrain)
D2_PASSI, D2_SALVA = 20_000, 10_000  # D fase 2, dalla fase 1, misti
D2B_SIM = 25  # D2b: simulati tenuti in fase 2 (~10%): dopo D1 servono a ricordare, non a insegnare
BLOCCO = 25  # episodi per blocco: un'interruzione perde al piu' quelli in corso
ORE_MAX = 10  # stima oltre questa: ci si ferma dopo la prova
SCADENZA_H = 10.5  # dall'avvio: oltre, nessun episodio nuovo (Kaggle uccide a 12)


def cella(tipo: str, testo: str) -> dict:
    righe = testo.splitlines(keepends=True)
    if tipo == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": righe}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": righe}


def scrivi(cartella: Path, nome: str, celle: list, metadati: dict) -> None:
    cartella.mkdir(parents=True, exist_ok=True)
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
    (cartella / f"{nome}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {
        "id": f"{UTENTE}/{nome}", "title": nome, "code_file": f"{nome}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "competition_sources": [], "model_sources": [], **metadati,
    }
    (cartella / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("scritto", cartella / f"{nome}.ipynb")


TROVA_CODICE = """import glob, os, shutil
s = glob.glob('/kaggle/input/**/robotica/scripts/genera_act_sim.py', recursive=True)
assert s, 'pacchetto codice non trovato: e\\' in Input?'
shutil.copytree(os.path.dirname(os.path.dirname(s[0])), '/tmp/robotica', dirs_exist_ok=True)
print('codice:', s[0])"""

AMBIENTE = """!pip install -q uv
!UV_CACHE_DIR=/tmp/uvcache uv venv -q -p 3.12 /tmp/venv
!UV_CACHE_DIR=/tmp/uvcache uv pip install -q --python /tmp/venv/bin/python --torch-backend={torch} "lerobot[{extra}]==0.6.1" "mujoco==3.3.7" scipy
!/tmp/venv/bin/python -c "import lerobot, mujoco, numpy; print('lerobot', lerobot.__version__, '| mujoco', mujoco.__version__, '| numpy', numpy.__version__)\""""

gen = [
    ("md", f"""# Episodi simulati per ACT — SO-101, caramelle

{EPISODI} episodi **riusciti** (i falliti si scartano e restano nel registro col motivo),
{PROCESSI} processi in parallelo con semi diversi, poi un merge solo.
Codice: `robotica` (dataset `so101-sim-codice`), stesso commit del portatile.
Output: `caramelle_sim/` (dataset LeRobot) e i registri `*_generazione.jsonl`."""),
    ("code", "import time; INIZIO = time.time()\n!nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; nproc; free -g | head -2; df -h /tmp /kaggle/working | tail -2"),
    ("code", TROVA_CODICE),
    ("code", AMBIENTE.format(torch="cpu", extra="dataset")),
    ("md", """## Chi disegna?

Il contenitore ha EGL solo di Mesa: senza `libEGL_nvidia`, "egl" disegna sulla
CPU (llvmpipe, 181 ms a fotogramma) senza dire niente. La T4 vera: 1,3 ms.
Le librerie GL di NVIDIA devono avere la versione **esatta** del driver: da apt
se coincide, altrimenti dall'installer NVIDIA estratto in una cartella privata.
Poi si disegna la **nostra** scena a meta' episodio e si esige NVIDIA e
un'immagine non uniforme. Un render che non da' errore non prova niente."""),
    ("code", """import subprocess, os, json
ver = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True).stdout.split()[0]
subprocess.run('apt-get -qq update > /dev/null 2>&1', shell=True)
cand = subprocess.run(f"apt-cache policy libnvidia-gl-{ver.split('.')[0]} | grep Candidate", shell=True, capture_output=True, text=True).stdout
print('driver', ver, '| apt', cand.strip())
AMB = {**os.environ, 'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl'}
if ver in cand:
    r = subprocess.run(f"apt-get -qq install -y --no-install-recommends libnvidia-gl-{ver.split('.')[0]}", shell=True, capture_output=True, text=True)
    print('apt esito', r.returncode, r.stderr[-300:])
    AMB['__EGL_VENDOR_LIBRARY_FILENAMES'] = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
else:  # piano B, provato nella diagnosi: stesse prestazioni
    for url in (f'https://us.download.nvidia.com/tesla/{ver}/NVIDIA-Linux-x86_64-{ver}.run',
                f'https://us.download.nvidia.com/XFree86/Linux-x86_64/{ver}/NVIDIA-Linux-x86_64-{ver}.run'):
        if subprocess.run(['wget', '-q', url, '-O', '/tmp/nv.run']).returncode == 0:
            break
    subprocess.run('sh /tmp/nv.run --extract-only --target /tmp/nvx > /dev/null && mkdir -p /tmp/nvlib', shell=True, check=True)
    for nome in ('libEGL_nvidia', 'libnvidia-eglcore', 'libnvidia-glsi', 'libnvidia-gpucomp'):
        if os.path.exists(f'/tmp/nvx/{nome}.so.{ver}'):
            shutil.copy(f'/tmp/nvx/{nome}.so.{ver}', '/tmp/nvlib/')
    os.symlink(f'libEGL_nvidia.so.{ver}', '/tmp/nvlib/libEGL_nvidia.so.0')
    json.dump({'file_format_version': '1.0.0', 'ICD': {'library_path': '/tmp/nvlib/libEGL_nvidia.so.0'}}, open('/tmp/nvlib/10_nvidia.json', 'w'))
    AMB['__EGL_VENDOR_LIBRARY_FILENAMES'] = '/tmp/nvlib/10_nvidia.json'
    AMB['LD_LIBRARY_PATH'] = '/tmp/nvlib:' + AMB.get('LD_LIBRARY_PATH', '')
    shutil.rmtree('/tmp/nvx'); os.remove('/tmp/nv.run')
r = subprocess.run(['/tmp/venv/bin/python', '/tmp/robotica/scripts/genera_act_sim.py', '/kaggle/working/prova', '--prova-render', '--gpu', '--seme', '7'],
                   env=AMB, capture_output=True, text=True)
print(r.stdout[-600:], r.stderr[-1500:] if r.returncode else '')
assert r.returncode == 0, 'il render non e\\' sulla GPU o non disegna la scena'
from IPython.display import Image, display
display(Image('/kaggle/working/prova/prova_render.png'))"""),
    ("md", f"""## Prova: 2 episodi a blocchi da 1, il loro merge, e la stima

Prova anche il meccanismo dei blocchi. Se la stima supera **{ORE_MAX} h** ci si
ferma qui: meglio saperlo ora che dopo 12 h."""),
    ("code", f"""t = time.time()
r = subprocess.run(['/tmp/venv/bin/python', '/tmp/robotica/scripts/genera_act_sim.py', '/tmp/prova/p', '--episodi', '2', '--seme', '7', '--blocco', '1'],
                   env=AMB, capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-1500:] if r.returncode else '')
assert r.returncode == 0
prova_s = time.time() - t
righe = [json.loads(x) for x in open('/tmp/prova/p_generazione.jsonl')]
for x in righe:
    print(x)
blocchi = sorted(glob.glob('/tmp/prova/p_b*'))
r = subprocess.run(['/tmp/venv/bin/lerobot-edit-dataset', '--new_repo_id', 'so101/prova', '--new_root', '/tmp/prova_unita',
                    '--operation.type', 'merge', '--operation.repo_ids', json.dumps([f'so101/b{{i}}' for i in range(len(blocchi))]),
                    '--operation.roots', json.dumps(blocchi)], capture_output=True, text=True)
info = json.load(open('/tmp/prova_unita/meta/info.json'))
print('blocchi', len(blocchi), '-> merge:', info['total_episodes'], 'episodi', info['total_frames'], 'fotogrammi')
assert len(blocchi) == 2 and info['total_episodes'] == 2, r.stderr[-800:]
# Un processo solo: con {PROCESSI} insieme i core sono condivisi (fisica e video sono CPU), si conta 1,5x.
a_episodio = prova_s / 2
ore = a_episodio * {EPISODI} / {PROCESSI} * 1.5 / 3600
print(f'{{a_episodio:.0f}} s a episodio scritto (falliti compresi) -> stima {{ore:.1f}} h per {EPISODI} con {PROCESSI} processi')
assert ore < {ORE_MAX}, f'stima {{ore:.1f}} h oltre {ORE_MAX}: ci si ferma'
shutil.rmtree('/tmp/prova'); shutil.rmtree('/tmp/prova_unita')"""),
    ("md", f"""## Tutti: {PROCESSI} processi, {EPISODI} episodi, a blocchi da {BLOCCO}

Blocchi e registri direttamente in `/kaggle/working/blocchi`: un blocco chiuso
e' un dataset completo. Scadenza a {SCADENZA_H} h dall'avvio del notebook: oltre,
nessun episodio nuovo, e si unisce quello che c'e'."""),
    ("code", f"""import math
quanti = [math.ceil({EPISODI} / {PROCESSI})] * {PROCESSI}
quanti[-1] = {EPISODI} - sum(quanti[:-1])
B = '/kaggle/working/blocchi'
os.makedirs(B, exist_ok=True)
scadenza = INIZIO + {SCADENZA_H} * 3600
processi = []
for k, n in enumerate(quanti):
    log = open(f'{{B}}/parte_{{k}}.log', 'w')
    processi.append(subprocess.Popen(['/tmp/venv/bin/python', '/tmp/robotica/scripts/genera_act_sim.py', f'{{B}}/parte_{{k}}',
                                      '--episodi', str(n), '--seme', str(1_000_000 * (k + 1)), '--blocco', '{BLOCCO}',
                                      '--scadenza', str(scadenza)], env=AMB, stdout=log, stderr=subprocess.STDOUT))
t0 = time.time()
while any(p.poll() is None for p in processi):
    time.sleep(300)
    fatti = provati = 0
    for k in range(len(quanti)):
        try:
            righe = [json.loads(x) for x in open(f'{{B}}/parte_{{k}}_generazione.jsonl')]
        except FileNotFoundError:
            continue
        provati += len(righe)
        fatti += sum('registra_s' in x and 'scartato' not in x for x in righe)
    ore = (time.time() - t0) / 3600
    fine = ore * {EPISODI} / fatti if fatti else float('nan')
    print(f'{{ore * 60:.0f}} min: {{fatti}} scritti su {{provati}} provati, fine prevista a {{fine:.1f}} h', flush=True)
for k, p in enumerate(processi):
    print('parte', k, 'esito', p.returncode, open(f'{{B}}/parte_{{k}}.log').read()[-300:])
assert all(p.returncode == 0 for p in processi)"""),
    ("md", "## Un dataset solo, e i conti"),
    ("code", f"""blocchi = sorted(glob.glob(f'{{B}}/parte_*_b*'))
blocchi = [b for b in blocchi if os.path.exists(b + '/meta/info.json')]
r = subprocess.run(['/tmp/venv/bin/lerobot-edit-dataset', '--new_repo_id', 'so101/caramelle_sim', '--new_root', '/tmp/caramelle_sim',
                    '--operation.type', 'merge', '--operation.repo_ids', json.dumps([f'so101/b{{i}}' for i in range(len(blocchi))]),
                    '--operation.roots', json.dumps(blocchi)], capture_output=True, text=True)
print(r.stdout[-800:], r.stderr[-800:])
info = json.load(open('/tmp/caramelle_sim/meta/info.json'))
attesi = sum(json.load(open(b + '/meta/info.json'))['total_episodes'] for b in blocchi)
print('blocchi', len(blocchi), '| episodi', info['total_episodes'], 'su', attesi, 'nei blocchi | fotogrammi', info['total_frames'])
assert info['total_episodes'] == attesi
if attesi < {EPISODI}:
    print(f'ATTENZIONE: {{attesi}} episodi invece di {EPISODI} (scadenza)')
righe = [json.loads(x) for k in range({PROCESSI}) for x in open(f'{{B}}/parte_{{k}}_generazione.jsonl')]
scritti = [x for x in righe if 'registra_s' in x and 'scartato' not in x]
print(f'provati {{len(righe)}}, scritti {{len(scritti)}} ({{100 * len(scritti) / len(righe):.0f}}%), non deterministici {{sum("scartato" in x for x in righe)}}')
print('render ms/fotogramma mediana', sorted(x['render_ms'] for x in scritti)[len(scritti) // 2])
from collections import Counter
print('motivi di scarto:', Counter(m.split(' ')[1] if ' ' in m else m for x in righe for m in x.get('fallito', [])).most_common(6))
with open('/kaggle/working/generazione.jsonl', 'w') as f:
    f.writelines(json.dumps(x) + '\\n' for x in righe)
# Solo ora, col merge verificato, si buttano i blocchi: lo spazio in /kaggle/working non raddoppia mai.
shutil.rmtree(B)
shutil.move('/tmp/caramelle_sim', '/kaggle/working/caramelle_sim')
print(f'fatto in {{(time.time() - INIZIO) / 3600:.1f}} h')"""),
]

TROVA_SIM = """import json, subprocess
sim = [os.path.dirname(os.path.dirname(p)) for p in glob.glob('/kaggle/input/**/caramelle_sim/meta/info.json', recursive=True)]
assert len(sim) == 1, sim
es = json.load(open(sim[0] + '/meta/info.json'))['total_episodes']
SIM = sim[0]
print('simulati:', SIM, es, 'episodi')
if es != %d:
    print(f'ATTENZIONE: {es} simulati invece di %d')""" % (EPISODI, EPISODI)


def unisci(copie: int) -> str:
    """Veri x copie (metadati riparati) + simulati in /tmp/dati: identico per C e D2."""
    lettere = "abcdefgh"[:copie]
    return f"""veri = [os.path.dirname(os.path.dirname(p)) for p in glob.glob('/kaggle/input/**/meta/info.json', recursive=True)
        if json.load(open(p))['total_episodes'] == 50]
assert len(veri) == 1, veri
copie = [f'/tmp/veri_{{x}}' for x in '{lettere}']
for copia in copie:
    shutil.copytree(veri[0], copia, copy_function=shutil.copyfile)  # /kaggle/input e' in sola lettura: niente permessi
    print(subprocess.run(['/tmp/venv/bin/python', '/tmp/robotica/scripts/ripara_meta_episodi.py', copia], capture_output=True, text=True).stdout)
ids = [f'so101/veri_{{x}}' for x in '{lettere}'] + ['so101/sim']
r = subprocess.run(['/tmp/venv/bin/lerobot-edit-dataset', '--new_repo_id', 'so101/misto', '--new_root', '/tmp/dati',
                    '--operation.type', 'merge', '--operation.repo_ids', json.dumps(ids),
                    '--operation.roots', json.dumps(copie + [SIM])], capture_output=True, text=True)
print(r.stdout[-500:], r.stderr[-500:])
info = json.load(open('/tmp/dati/meta/info.json'))
fs = json.load(open(SIM + '/meta/info.json'))['total_frames']
print('episodi', info['total_episodes'], 'fotogrammi', info['total_frames'], f'-> veri {{100 * (1 - fs / info["total_frames"]):.0f}}% dei campioni')
assert info['total_episodes'] == 50 * {copie} + es
for copia in copie:
    shutil.rmtree(copia)
REPO = 'so101/misto'"""


SOLO_SIM = """shutil.copytree(sim[0], '/tmp/dati', copy_function=shutil.copyfile)
info = json.load(open('/tmp/dati/meta/info.json'))
print('solo simulati: episodi', info['total_episodes'], 'fotogrammi', info['total_frames'])
REPO = 'so101/caramelle_sim'"""

# D2 riparte dal checkpoint finale di D1. --policy.path carica i pesi; LeRobot 0.6.1
# ricalcola la normalizzazione sul dataset nuovo (lerobot_train.py, righe 347-377).
TROVA_D1 = """modelli = {int(m.split('/')[-2]): m for m in glob.glob('/kaggle/input/**/act_caramelle/checkpoints/*/pretrained_model', recursive=True)
           if m.split('/')[-2].isdigit()}
assert modelli, 'checkpoint di D1 non trovato: e\\' in Input?'
print('checkpoint di D1:', sorted(modelli))
assert max(modelli) == %d, 'D1 non e\\' arrivata in fondo'
PARTENZA = modelli[max(modelli)]
print('si parte da', PARTENZA)""" % D1_PASSI


def addestra(nome: str, passi: int, salva: int, da_d1: bool = False) -> str:
    politica = "--policy.path=$PARTENZA" if da_d1 else "--policy.type=act"
    return f"""!/tmp/venv/bin/lerobot-train \\
    --dataset.repo_id=$REPO \\
    --dataset.root=/tmp/dati \\
    --dataset.image_transforms.enable=true \\
    {politica} \\
    --policy.device=cuda \\
    --policy.push_to_hub=false \\
    --output_dir=/kaggle/working/act_caramelle \\
    --job_name=act_{nome} \\
    --steps={passi} \\
    --save_freq={salva} \\
    --batch_size=8 \\
    --wandb.enable=false"""


ZIP = """base = '/kaggle/working/act_caramelle/checkpoints'
numeri = {int(n): n for n in os.listdir(base) if n.isdigit()}
print('checkpoint:', sorted(numeri))
for passi in sorted(numeri):
    uscita = shutil.make_archive(f'/kaggle/working/modello_act_{passi}', 'zip', os.path.join(base, numeri[passi], 'pretrained_model'))
    print('pronto:', uscita, os.path.getsize(uscita) / 1e6, 'MB')"""

INIZIO = [
    ("code", "!nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; df -h /tmp | tail -1"),
    ("code", TROVA_CODICE),
    ("code", AMBIENTE.format(torch="auto", extra="training,dataset")),
    ("code", TROVA_SIM),
]

run_c = [
    ("md", f"""# ACT run C — 50 veri x{COPIE_VERI} + {EPISODI} simulati, mescolati dall'inizio

Veri: `caramelle-so101-50`, {COPIE_VERI} copie con i metadati riparati (lo stesso
dataset nominato piu' volte nel merge si rompe): ~31% dei campioni, ogni
fotogramma vero visto ~2,6 volte (run B: 5,1). Simulati: output di
`act-caramelle-sim-gen`. `lerobot-train` come la run B, {PASSI} passi,
checkpoint ogni {SALVA}. Il checkpoint si sceglie **sul braccio**."""),
    *INIZIO,
    ("code", unisci(COPIE_VERI)),
    ("code", addestra("run_c", PASSI, SALVA)),
    ("code", ZIP),
]

run_d1 = [
    ("md", f"""# ACT run D, fase 1 — solo i {EPISODI} simulati

La rete impara la struttura del compito dove i dati costano poco. {D1_PASSI} passi,
checkpoint ogni {D1_SALVA}. Poi `act-caramelle-run-d2` riparte da qui coi veri
dentro. Provata da sola sul braccio dice quanto la simulazione e' vicina al vero:
non ha mai visto un'immagine vera."""),
    *INIZIO,
    ("code", SOLO_SIM),
    ("code", addestra("run_d1", D1_PASSI, D1_SALVA)),
    ("code", ZIP),
]

run_d2 = [
    ("md", f"""# ACT run D, fase 2 — dalla fase 1, coi veri x{COPIE_VERI} + simulati

Stessi dati della run C (stessa cella di merge); cambia solo il punto di
partenza: i pesi di D1 invece che casuali. Ancora misti, non solo veri: coi soli
veri la rete dimenticherebbe la varieta' della simulazione. {D2_PASSI} passi,
checkpoint ogni {D2_SALVA}."""),
    *INIZIO,
    ("code", TROVA_D1),
    ("code", unisci(COPIE_VERI)),
    ("code", addestra("run_d2", D2_PASSI, D2_SALVA, da_d1=True)),
    ("code", ZIP),
]

POCHI_SIM = f"""# Uno ogni {{es // {D2B_SIM}}}: sparsi su tutti i processi e i blocchi. Provato in locale (venv lerobot sim):
# delete_episodes lascia gli episodi giusti, azioni identiche.
tieni = list(range(0, es, es // {D2B_SIM}))[:{D2B_SIM}]
via = [i for i in range(es) if i not in tieni]
r = subprocess.run(['/tmp/venv/bin/lerobot-edit-dataset', '--repo_id', 'so101/caramelle_sim', '--root', SIM,
                    '--new_repo_id', 'so101/sim_pochi', '--new_root', '/tmp/sim_pochi',
                    '--operation.type', 'delete_episodes', '--operation.episode_indices', json.dumps(via)], capture_output=True, text=True)
print(r.stdout[-300:], r.stderr[-300:])
SIM = '/tmp/sim_pochi'
es = json.load(open(SIM + '/meta/info.json'))['total_episodes']
print('simulati tenuti:', es, 'episodi', tieni)
assert es == {D2B_SIM}"""

run_d2b = [
    ("md", f"""# ACT run D, fase 2b — dalla fase 1, veri x{COPIE_VERI} + solo {D2B_SIM} simulati

Come D2, ma i simulati in fase 2 sono {D2B_SIM} invece di {EPISODI} (~10% dei campioni
invece di ~69%). Domanda: dopo la fase 1 i simulati servono solo a non dimenticare
(per non dimenticare ne basta poco), e il resto del tempo va meglio sui veri?
Idea di Daniele. {D2_PASSI} passi, checkpoint ogni {D2_SALVA}."""),
    *INIZIO,
    ("code", TROVA_D1),
    ("code", POCHI_SIM),
    ("code", unisci(COPIE_VERI)),
    ("code", addestra("run_d2b", D2_PASSI, D2_SALVA, da_d1=True)),
    ("code", ZIP),
]

FONTI_ADDESTRA = [f"{UTENTE}/caramelle-so101-50", f"{UTENTE}/so101-sim-codice"]
scrivi(QUI / "gen", "act-caramelle-sim-gen", gen,
       {"dataset_sources": [f"{UTENTE}/so101-sim-codice"], "kernel_sources": []})
scrivi(QUI / "run_c", "act-caramelle-run-c", run_c,
       {"dataset_sources": FONTI_ADDESTRA, "kernel_sources": [f"{UTENTE}/act-caramelle-sim-gen"]})
scrivi(QUI / "run_d1", "act-caramelle-run-d1", run_d1,
       {"dataset_sources": [f"{UTENTE}/so101-sim-codice"], "kernel_sources": [f"{UTENTE}/act-caramelle-sim-gen"]})
for cartella, nome, celle in (("run_d2", "act-caramelle-run-d2", run_d2), ("run_d2b", "act-caramelle-run-d2b", run_d2b)):
    scrivi(QUI / cartella, nome, celle,
           {"dataset_sources": FONTI_ADDESTRA,
            "kernel_sources": [f"{UTENTE}/act-caramelle-sim-gen", f"{UTENTE}/act-caramelle-run-d1"]})
