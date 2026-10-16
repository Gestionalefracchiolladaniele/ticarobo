# ACT senza pause fra un blocco e l'altro (inferenza asincrona di LeRobot).
#
#   .\rollout_async.ps1 -Modello <cartella>               120 s
#   .\rollout_async.ps1 -Modello <cartella> -Secondi 60
#
# Due processi: il policy server (la rete) in una finestra a parte, il client
# (il braccio) qui. Il client chiede il blocco nuovo a meta' di quello in
# corso: arriva mentre il braccio si muove ancora. Spiegazione in
# rollout_async.py. A fine prova: numeri del ciclo (Hz, giri senza azioni).
#
# SENZA safety/ (D26), come rollout.ps1: mano sull'interruttore dei 12 V,
# NON sotto il braccio. A fine tempo (o Ctrl+C) torna piano alla posa di
# partenza, poi perde la coppia.
#
# Su 2 core server e client si contendono la CPU: guarda gli Hz a fine prova.

param(
    [Parameter(Mandatory = $true)][string]$Modello,
    [int]$Secondi = 120,
    [int]$Limite = 15,
    [int]$Porta = 8080
)

. "$PSScriptRoot\comune.ps1"
Set-Location $ACT

if (-not (Test-Path "$Modello\model.safetensors")) {
    Write-Host "Modello non trovato in $Modello"
    exit 1
}
$f = Get-PortaDi $FOLLOWER_SERIALE
if (-not $f) {
    Write-Host "Follower non trovato. Cavo USB e 12 V collegati? Schede in WSL? usbipd detach --all"
    exit 1
}
Write-Host "follower $f   camera $CAMERA_POLSO   $Secondi s   limite $Limite"
Write-Host "Modello: $Modello"

Write-Host "Avvio il policy server (finestra a parte)..."
$server = Start-Process -FilePath "$VENV\python.exe" -PassThru -WindowStyle Minimized `
    -ArgumentList "-m", "lerobot.async_inference.policy_server", "--host=127.0.0.1", "--port=$Porta", `
                  "--fps=30", "--inference_latency=0.033", "--obs_queue_timeout=1"
try {
    & "$VENV\python.exe" "$ACT\rollout_async.py" $f "$Modello" `
        --camera $CAMERA_POLSO --secondi $Secondi --limite $Limite --server "127.0.0.1:$Porta"
}
finally {
    if (-not $server.HasExited) { Stop-Process -Id $server.Id -Force }
    Write-Host "Server fermato."
}
