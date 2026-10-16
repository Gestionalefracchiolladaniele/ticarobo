# Il follower fa da solo: ACT guarda la camera del polso e muove il braccio.
# Il leader non serve (puo' restare collegato, non si usa).
#
#   .\rollout.ps1                 60 s col modello di Kaggle
#   .\rollout.ps1 -Secondi 30     piu' corto
#   .\rollout.ps1 -Limite 5       piu' lento (piu' tempo per reagire)
#
# SENZA safety/: eccezione dichiarata in D26, scelta di Daniele (2026-10-16).
# Uniche protezioni: max_relative_target (gradi per ciclo) e il range di
# calibrazione. La rete di sicurezza e' la mano sull'interruttore dei 12 V,
# NON sotto il braccio. Togliendo corrente il braccio perde la coppia e cade.
#
# A fine tempo (o Ctrl+C) il braccio torna piano alla posa di partenza, poi
# perde la coppia: partire dalla posa ripiegata degli episodi.
#
# ACT esegue 100 azioni (3,3 s) a ogni sguardo, senza riguardare la camera:
# fra un blocco e l'altro una pausa (inferenza su CPU) e' normale.

param(
    [int]$Secondi = 60,
    [int]$Limite = 15,
    [string]$Modello = "$PSScriptRoot\kaggle\output\modello_act"
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

# Il modello e' stato salvato con device=cuda (Kaggle): qui si forza la CPU.
& "$VENV\lerobot-rollout.exe" `
    --strategy.type=base `
    --policy.path="$Modello" `
    --policy.device=cpu `
    --device=cpu `
    --robot.type=so101_follower --robot.port=$f --robot.id=follower `
    --robot.max_relative_target=$Limite --robot.num_read_retries=10 `
    --robot.cameras="$CAMERE" `
    --task="Metti le caramelle gialle nel bicchiere a sinistra e quelle celesti-bianche a destra" `
    --duration=$Secondi `
    --fps=30 `
    --display_data=false `
    --play_sounds=false
