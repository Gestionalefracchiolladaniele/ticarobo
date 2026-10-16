# ACT sul SO-101 — smistare caramelle imitando

Il braccio impara guardando: tu lo guidi col leader N volte, una rete (ACT)
impara a fare lo stesso da sola. Perche' le scelte: `robotica/docs/decisioni.md`,
**D26**. Numeri: `robotica/esperimenti/so101/act.md`.

**Gira su Windows, non in WSL.** ACT vuole camera e bracci letti dallo stesso
processo, e la camera via usbipd arriva a pezzi (14-23% del fotogramma,
misurato). Le schede quindi restano a Windows: se `braccio.ps1` le ha date a
WSL, si liberano con `usbipd detach --all`.

## Il compito

- Caramelle **gialle** → bicchiere a **sinistra**
- Caramelle **celesti-bianche** → bicchiere a **destra**
- Un **altro colore** → **mai toccato**

## Il materiale

| | |
|---|---|
| venv | `.venv\`, LeRobot **0.6.1** (come WSL), torch CPU, Python 3.12 su F: |
| Calibrazione | **la stessa di WSL**: `HF_LEROBOT_CALIBRATION` → `lerobot\calibrazione` |
| Dati | `dati\so101\<dataset>\` (fuori da git) |
| Porte | dal seriale: follower `5B90101906`, leader `5B79082554` (oggi COM3 / COM4) |
| Camera polso | indice **1** (0 e' quella dello schermo) |

All'avvio compare un lungo errore di `torchcodec` seguito da *"Falling back to
'pyav'"*: **e' normale**. Serve FFmpeg "shared" che qui non c'e', e LeRobot
ripiega da solo. Registrare usa PyAV comunque; addestrare si fa su Kaggle.

## Passo dopo passo

| # | Comando | Cosa guardi |
|---|---|---|
| 1 | `.\camere.ps1` | la foto della camera 1 e' quella del polso? |
| 2 | `.\teleop.ps1 15 -Finestra` | dalla posa di partenza la camera vede **tutte** le caramelle? |
| 3 | `.\misura.ps1` | ritardo leader→follower per giunto, Hz del ciclo (aspetta INVIO) |
| 4 | `.\registra.ps1 -Prova -Episodi 1` | si salva subito? Pochi fotogrammi persi? (la prova vecchia si cancella da sola) |
| 5 | `.\rivedi.ps1 -Prova` | il follower rifa' il movimento dell'episodio 0? |
| 6 | `.\registra.ps1 -Episodi N` | gli episodi veri; `-Riprendi` per aggiungerne N in coda |
| 7 | Kaggle | addestramento: `kaggle\` (sotto) |
| 8 | `.\rollout.ps1 -Secondi 60` | il braccio da solo (`-Modello <cartella>`, `-Limite 5` piu' lento). **Senza `safety/`** (D26): mano sull'interruttore dei 12 V, **non** sotto il braccio |
| 9 | `.\rollout_async.ps1 -Modello <cartella>` | come il rollout, **senza pause**: server (la rete) in una finestra minimizzata, client (il braccio) qui; a fine prova azioni/s e giri senza azioni. **Mai provato sul braccio** |

**Rollout:** ACT guarda una volta ed esegue 100 azioni (3,3 s) alla cieca;
fra un blocco e l'altro una pausa (inferenza su CPU) e' normale. A fine tempo
torna piano alla posa di partenza, poi perde la coppia. Il modello e' salvato
con `device: cuda`: lo script forza la CPU. `rollout_async` chiede il blocco
nuovo a meta' di quello in corso (serve `lerobot[async]`, cioe' gRPC, gia'
installato); il client di LeRobot da solo toglierebbe la coppia di colpo a
fine prova, `rollout_async.py` riporta prima il braccio alla partenza.

**Bicchieri segnati col nastro, sempre li'.** Nei 50 episodi si spostavano:
ACT, che non ha memoria e col polso spesso non li vede, mira alla media.

**Se si ferma da solo:** `Failed to write 'Torque_Enable' on id_=1` = il
cavetto del motore della base si e' staccato; `Porte non trovate` = USB del
leader staccata; `Timed out waiting for frame` = camera (altra app, USB,
indice). Gli episodi gia' salvati restano integri: si riparte con
`-Riprendi`, contando **gli episodi da aggiungere**.

**Episodi da togliere o dataset da unire:** `lerobot-edit-dataset`
(`--operation.type delete_episodes` / `merge`), sempre in una **cartella
nuova** (`--new_root`), verificata prima di sostituire l'originale.

Lanciati da PowerShell. Se Windows blocca gli script:
`powershell -ExecutionPolicy Bypass -File "F:\cartella prova\lerobot\act\teleop.ps1"`.

**Senza finestra rerun di default:** su 2 core rallentava il braccio. La
camera registra lo stesso (rerun e' solo una copia a schermo); `-Finestra`
per riaverla. **Video H.264 ultrafast compresso durante l'episodio**: zero
attesa dopo `→` (AV1, il default di LeRobot, impiegava oltre 4 minuti).

## Kaggle (`kaggle\`)

| File | |
|---|---|
| `scrivi_notebook.py` | genera `act_caramelle.ipynb`: si cambia questo, non il notebook. `--passi`, `--salva`, `--zip` (checkpoint da zippare), `--cartella` |
| `kernel-metadata.json` | notebook privato, **GPU T4 x2** (`NvidiaTeslaT4`), internet on, dataset in input |
| `dataset\` | `caramelle.zip` + `dataset-metadata.json` (fuori da git) |
| `output\` | i modelli scaricati (fuori da git) |

Due run, due cartelle: `kaggle\` = `act-caramelle` (dataset
`caramelle-so101`, run A: 35 episodi, 25k), `kaggle_50\` = `act-caramelle-50`
(dataset `caramelle-so101-50`, run B: 50 episodi, 60k, zip a 30k e 60k).
Cartelle e notebook separati: un `push` sullo stesso notebook non deve
toccare una run in corso.

**Passi:** il minimo degli autori di ACT e' 5000 epoche *loro*, e la loro
epoca e' **un campione per episodio**: 5000 x episodi / batch. ~3,1 passi/s
su una T4 (200 passi in 64 s), piu' ~3 min di installazione.

CLI `kaggle` nel venv; la chiave **solo** in `$env:KAGGLE_API_TOKEN`, mai in
un file; `$env:KAGGLE_CONFIG_DIR='F:\caches\kaggle'`.

```powershell
kaggle datasets version -p kaggle\dataset -m "nuovi episodi"    # dataset aggiornato
kaggle kernels push -p kaggle --accelerator NvidiaTeslaT4      # lancia
kaggle kernels status danielefracchiolla/act-caramelle
kaggle kernels output danielefracchiolla/act-caramelle -p kaggle\output --file-pattern '^modello_act.*\.zip$'   # a fine corsa
kaggle datasets create -p kaggle_50\dataset                      # dataset nuovo (run B)
```

`kernels output` senza `--file-pattern` scarica anche tutti i checkpoint
(GB). L'errore `'charmap' codec` alla fine e' il log che Windows non sa
scrivere: lo zip c'e' lo stesso. **Lo stato `COMPLETE` non basta:** nel log
si controllano passi fatti, `End of training`, `num_episodes`.

Il log si legge solo a fine corsa (`kernels logs`); dal vivo, dal browser.

## Come si registra un episodio

**Un episodio = tutte le caramelle sul tavolo.** 2-5 caramelle buone
mescolate (2 gialle + 1 celeste, 1 gialla + 3 celesti...), quasi sempre anche
quella da evitare, in posizioni sempre diverse.

1. Braccio nella **posa di partenza** (segnata). Caramelle disposte.
2. Prendi **sempre per prima quella piu' a sinistra** (vista dal braccio).
   ACT non sceglie: imita. Se l'ordine cambia a caso, fa la media e esita.
3. **Ritmo costante, mai fermarsi a meta'**: le pause insegnano a stare fermi.
4. Finite le buone: braccio nella posa di partenza e **subito →**.
5. Riordino: svuota i bicchieri, ridisponi in modo diverso, poi **→**.
   Nessun limite di tempo (10 minuti): episodio e riordino li chiudi tu.
6. **La pinza guarda sempre il tavolo**: se la camera vede la stanza, la rete
   impara anche quella.

| Tasto | |
|---|---|
| **→** o `n` | salva, prossimo |
| **←** o `r` | butta e rifai — **ogni episodio sbagliato va rifatto**: ACT copia anche gli errori |
| **Esc** o `q` | chiudi |

Bicchieri **sempre nello stesso punto** (scotch). Luce il piu' possibile uguale.

7. **Ultimi 2-3 cm lenti** e caramella centrata fra le dita: e' la parte che
   ACT ha sbagliato col primo modello.
8. **Pinza del leader chiusa fino in fondo**, oltre il contatto: ACT impara
   una posizione bersaglio, e la stretta nasce dalla differenza.
9. **Camera del polso fissata e segnata** col pennarello: si sposta sul
   supporto, e una camera che deriva cambia cio' che la rete vede.
10. Dopo `→` aspetta il messaggio del riordino: un doppio `→` salva episodi
    vuoti (successo: 3 s e 0 s).
