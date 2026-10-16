# Misura da dove viene il ritardo della teleoperazione (misura_ritardo.py).
#
#   .\misura.ps1                 limite 15, camera accesa (come in registrazione)
#   .\misura.ps1 -Limite 5
#   .\misura.ps1 -SenzaCamera    per vedere quanto pesa la camera
#
# Prima: bracci nella stessa posa ripiegata. Il programma aspetta INVIO.

param([int]$Limite = 15, [switch]$SenzaCamera)

. "$PSScriptRoot\comune.ps1"
Set-Location $ACT
$p = Get-Porte
$cam = if ($SenzaCamera) { @() } else { @("--camera", $CAMERA_POLSO) }

& "$VENV\python.exe" "$ACT\misura_ritardo.py" $p.Follower $p.Leader --limite $Limite @cam
