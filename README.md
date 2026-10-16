# SO-101 vero — LeRobot

Il braccio fisico, comandato con LeRobot. **Separato dal progetto `robotica`**
perche' LeRobot vuole `numpy>=2` e il progetto `<2` (mplib): nello stesso venv
uno dei due si rompe. Perche' e come: `robotica/docs/decisioni.md`, **D22**.
Numeri e cose andate storte: `robotica/esperimenti/so101/hardware.md`.

## L'hardware

| | Follower | Leader |
|---|---|---|
| Colore | bianco | nero |
| Cosa fa | esegue | lo muovi tu a mano |
| Motori | 6x STS3215 12 V, 1/345 | 6x STS3215 7,4 V, 1/191 e 1/147 |
| Alimentatore | **12 V 2 A** (Qingliupower) | **5 V 3 A** (rettangolare) |
| Scheda | basetta bianca, seriale **5B90101906** | basetta nera, seriale **5B79082554** |
| Adesivi motori | F1 base … F6 pinza | L1 base … L6 grilletto |

**Gli alimentatori non si scambiano:** il 12 V sul leader brucia i motori.

## Usarlo

**1. PowerShell** — collega le schede a WSL e apre un terminale Ubuntu pronto:
```powershell
& "F:\cartella prova\lerobot\braccio.ps1"
```
La prima volta per una scheda nuova chiede la conferma UAC (bind).

**2. Ubuntu** (prompt `(.venv) danie@...`):
```bash
bash teleop.sh        # muovi il nero, il bianco copia (limite 5 gradi/comando)
bash teleop.sh 15     # piu' reattivo
bash calibra.sh follower   # solo se serve ricalibrare
bash calibra.sh leader
.venv/bin/python leggi_motori.py /dev/ttyACM0   # chi risponde, tensione, temperatura
```
Le porte le trovano gli script dal seriale (`porte.sh`): non serve sapere chi e'
ttyACM0 e chi ttyACM1, e **cambiano** a ogni ricollegamento.

**Per fermare la teleoperazione:** leader ripiegato in basso, poi Ctrl+C. Il
follower perde la coppia e **cade**: da in basso non va da nessuna parte.

**Quando non lavori: stacca il 12 V.** Il follower si scalda anche da fermo.

## Calibrazione

Gia' fatta per entrambi (2026-10-09). File in
`~/.cache/huggingface/lerobot/calibration/` dentro Ubuntu, copia in
`calibrazione/` qui. Se la cache si svuota, si ricopia invece di rifarla.

Se va rifatta: **posa a meta'** = "lampada da scrivania", vista **di lato**:
prima Y (F2) dritta in su, piega in **F3** (non in F4!), avambraccio
orizzontale, polso dritto, pinza a meta'. Invio. Poi ogni giunto da un fine
corsa all'altro, F5 escluso. Invio.

- **Non premere Invio mentre carica**: gli Invio restano in coda e vengono
  consumati tutti insieme (risultato: MIN = MAX, LeRobot rifiuta).
- **Il follower e' duro da girare** (1/345): la durezza non e' il fine corsa.
  Il fine corsa e' un blocco netto.

## Quando qualcosa non va

| Sintomo | Causa probabile |
|---|---|
| `/dev/ttyACM*` non esiste | WSL si e' spento o il cavo si e' mosso: rilancia `braccio.ps1` |
| `Permission denied` sulla porta | rilancia `braccio.ps1` (la distro non ha systemd: permessi a mano) |
| "Missing motor IDs", trovati `{}` | **ai motori non arriva corrente**: alimentatore, o filo rosso uscito dal morsetto verde |
| "There is no status packet" | risposta persa via usbipd: `teleop.sh` ritenta 10 volte; se torna su movimenti ampi, e' l'alimentatore da 2 A |
| il follower insegue con 1-2 s di ritardo | il limite gradi/comando: prova `bash teleop.sh 15` |
| col peso non arriva a braccio steso | leva: lavora vicino alla base, braccio piegato |

## File

| File | Cosa fa |
|---|---|
| `braccio.ps1` | Windows: bind/attach delle schede, apre Ubuntu col venv |
| `prepara_porte.sh` | Linux: driver `cdc_acm`, permessi, stampa porta e seriale |
| `porte.sh` | trova la porta di ogni braccio dal seriale |
| `teleop.sh`, `calibra.sh` | i comandi LeRobot, con le porte giuste |
| `gemello_lettore.py` | sola lettura a coppia spenta, una riga JSON per lettura: la usa `robotica\gemello.ps1` (D23) |
| `esecutore.py` | **muove il follower**: riceve pose gia' approvate da `safety/` e le esegue coi suoi cancelli (partenza nel range, range vero, letture fresche, inseguimento, passi piccoli, velocita' con tetto 60°/s, **90 per il polso**). Esegue anche **coreografie**, pose di fila senza fermarsi: le battute del gioco (D24). La usano `robotica\gesto.ps1` e `robotica\rps_vero.ps1` (D23) |
| `leggi_motori.py` | sola lettura: ID, tensione, temperatura, posizione |
| `camera/` | **la camera del kit, letta da Windows** (via usbipd arriva a pezzi): `lettore_camera.py` la manda a WSL con la mano trovata da MediaPipe (`modelli/gesture_recognizer.task`, 7,99 MB, **non in git**: se manca si riscarica da `https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task`), `trova_camera.py` dice quale indice e'. Venv suo, **per Windows**, con Python e cache uv su F:. Si prova con `robotica\camera.ps1` |
| `calibrazione/` | i file di calibrazione: copia di quelli di WSL, e **la sorgente diretta** per LeRobot su Windows (`act\`, via `HF_LEROBOT_CALIBRATION`). Se si ricalibra, si aggiornano tutti e due |
| `act/` | **ACT, imitazione col leader** — LeRobot 0.6.1 **su Windows** (venv suo), registrazione, dataset, notebook Kaggle. Guida nel suo `README.md`, **D26** |
| `.venv/` | LeRobot + torch CPU (`uv pip install --torch-backend=cpu "lerobot[feetech]"`) |

## Git

Questa cartella ha un **git suo** (dal 2026-10-11), separato da quello del
progetto: esecutore, camera, calibrazioni. Fuori da git: `.venv/` e il modello
MediaPipe (si riscarica). Fine riga: LF, CRLF solo per i `.ps1`
(`.gitattributes`): un `.sh` con CRLF rompe bash in WSL senza dire perche'.
