# Registra episodi per ACT: tu guidi il leader, si salvano camera del polso,
# giunti del follower e comandi.
#
#   .\registra.ps1 -Prova                3 episodi nel dataset "prova" (da buttare)
#   .\registra.ps1 -Prova -Episodi 1     uno solo
#   .\registra.ps1                       50 episodi nel dataset "caramelle"
#   .\registra.ps1 -Episodi 20 -Riprendi altri 20 in coda a "caramelle"
#   ... -Finestra                        con la finestra rerun: rallenta
#
# Senza finestra la camera registra lo stesso: la finestra e' solo una copia
# a schermo, e su 2 core il braccio con la finestra andava visibilmente piu'
# lento (2026-10-15). La posa di partenza si sceglie prima con teleop -Finestra.
#
# Nessun limite di tempo pratico (10 min): episodio e riordino li chiudi tu.
# Tasti: -> (o n) salva e passa al prossimo, <- (o r) butta e rifai,
#        Esc (o q) chiudi. La finestra del terminale deve avere il fuoco.
#
# Un episodio = tutte le caramelle sul tavolo. Regole (README.md):
#   - prima sempre la piu' a SINISTRA (vista dal braccio)
#   - gialla -> bicchiere sinistro, celeste-bianca -> destro, l'altro colore mai
#   - ritmo costante, nessuna pausa a meta'
#   - finito: braccio nella posa di partenza e subito ->
#   - episodio sbagliato: <- e si rifa' (ACT copia anche gli errori)
#
# Video: H.264 ultrafast compresso DURANTE l'episodio (streaming), 1 thread.
# AV1 (il default di LeRobot) a fine episodio: dopo 4 minuti su 2 core non
# aveva finito 90 s di video (2026-10-15). A fine sessione leggi il "Cadence
# summary": se e' sotto 30 Hz il compressore ruba il ciclo e il dataset mente.

param(
    [int]$Episodi = 50,
    [switch]$Riprendi,
    [switch]$Prova,
    [int]$Limite = 15,
    [switch]$Finestra
)

. "$PSScriptRoot\comune.ps1"
Set-Location $ACT

$nome = if ($Prova) { "prova" } else { "caramelle" }
if ($Prova -and -not $PSBoundParameters.ContainsKey("Episodi")) { $Episodi = 3 }
$cartella = "$ACT\dati\so101\$nome"

# La prova e' da buttare: si riparte sempre da zero. I dati veri no.
if ($Prova -and (Test-Path $cartella) -and -not $Riprendi) {
    Remove-Item -Recurse -Force $cartella
}
if ((Test-Path $cartella) -and -not $Riprendi) {
    Write-Host "Il dataset '$nome' esiste gia' in $cartella."
    Write-Host "Per aggiungere episodi: -Riprendi. Per ricominciare: cancella la cartella a mano."
    exit 1
}

$p = Get-Porte
Write-Host "follower $($p.Follower)   leader $($p.Leader)   camera $CAMERA_POLSO"
Write-Host "dataset $nome   episodi $Episodi   riprendi $([bool]$Riprendi)"

& "$VENV\lerobot-record.exe" `
    --robot.type=so101_follower --robot.port=$($p.Follower) --robot.id=follower `
    --robot.max_relative_target=$Limite --robot.num_read_retries=10 `
    --robot.cameras="$CAMERE" `
    --teleop.type=so101_leader --teleop.port=$($p.Leader) --teleop.id=leader `
    --teleop.num_read_retries=10 `
    --dataset.repo_id="so101/$nome" `
    --dataset.root="$cartella" `
    --dataset.single_task="Metti le caramelle gialle nel bicchiere a sinistra e quelle celesti-bianche a destra" `
    --dataset.num_episodes=$Episodi `
    --dataset.episode_time_s=600 `
    --dataset.reset_time_s=600 `
    --dataset.push_to_hub=false `
    --dataset.rgb_encoder.vcodec=h264 `
    --dataset.rgb_encoder.preset=ultrafast `
    --dataset.rgb_encoder.crf=23 `
    --dataset.streaming_encoding=true `
    --dataset.encoder_threads=1 `
    --display_data=$(if ($Finestra) { "true" } else { "false" }) `
    --play_sounds=false `
    --resume=$(if ($Riprendi) { "true" } else { "false" })
