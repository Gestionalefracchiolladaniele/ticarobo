# Impostazioni comuni agli script di ACT. Si include con:  . "$PSScriptRoot\comune.ps1"
#
# LeRobot gira su Windows (non in WSL) perche' la camera via usbipd arriva a
# pezzi (esperimenti/so101/hardware.md): ACT vuole camera e bracci letti dallo
# stesso processo. Le schede quindi NON vanno collegate a WSL: se braccio.ps1
# le ha prese, Windows non vede piu' le COM.

$ErrorActionPreference = "Stop"

$ACT = $PSScriptRoot
$VENV = "$ACT\.venv\Scripts"
# Come attivare il venv: LeRobot cerca rerun.exe (la finestra) nel PATH.
$env:PATH = "$VENV;$env:PATH"

# Stessa calibrazione di WSL, non una copia: una sola verita'.
$env:HF_LEROBOT_CALIBRATION = "F:\cartella prova\lerobot\calibrazione"
# Dataset e modelli dentro questa cartella, su F:.
$env:HF_LEROBOT_HOME = "$ACT\dati"

$FOLLOWER_SERIALE = "5B90101906"   # scheda bianca, 12 V
$LEADER_SERIALE = "5B79082554"     # scheda nera, 5 V

# Camera del polso. Indice 1 su questo portatile (0 e' quella dello schermo):
# verificato con camere.ps1 il 2026-10-15. Se cambia, si rilancia camere.ps1.
$CAMERA_POLSO = 1
$CAMERE = "{ polso: {type: opencv, index_or_path: $CAMERA_POLSO, width: 640, height: 480, fps: 30} }"

# La COM si trova dal numero di serie, come porte.sh in WSL: COM3/COM4 possono
# scambiarsi a ogni ricollegamento.
function Get-PortaDi([string]$seriale) {
    $dev = Get-PnpDevice -Class Ports -PresentOnly -ErrorAction SilentlyContinue |
        Where-Object { $_.InstanceId -match $seriale }
    if ($dev -and $dev.FriendlyName -match "\((COM\d+)\)") { return $Matches[1] }
    return $null
}

function Get-Porte {
    $f = Get-PortaDi $FOLLOWER_SERIALE
    $l = Get-PortaDi $LEADER_SERIALE
    if (-not $f -or -not $l) {
        Write-Host "Porte non trovate (follower: $f, leader: $l)."
        Write-Host "Cavi USB e alimentatori collegati? Se le schede sono in WSL, liberale:"
        Write-Host "  usbipd detach --all"
        exit 1
    }
    return @{ Follower = $f; Leader = $l }
}
