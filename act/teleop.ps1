# Teleoperazione, con la camera del polso accesa.
#
#   .\teleop.ps1                  limite 5 gradi per comando (prudente)
#   .\teleop.ps1 15               piu' reattivo
#   .\teleop.ps1 15 -Finestra     con la finestra rerun (camera e giunti a
#                                 schermo): rallenta il braccio su 2 core
#
# Serve a provare la posa di partenza: da li' la camera deve vedere TUTTE le
# caramelle. Regola LeRobot: devi poter fare il compito guardando solo lo
# schermo. Prima: bracci nella stessa posa, ripiegati, polso roll a circa +100.
# Per fermare: leader ripiegato in basso, poi Ctrl+C (il follower si appoggia).

param([int]$Limite = 5, [switch]$Finestra)

. "$PSScriptRoot\comune.ps1"
Set-Location $ACT
$p = Get-Porte
Write-Host "follower $($p.Follower)   leader $($p.Leader)   camera $CAMERA_POLSO   limite $Limite gradi/comando"

& "$VENV\lerobot-teleoperate.exe" `
    --robot.type=so101_follower --robot.port=$($p.Follower) --robot.id=follower `
    --robot.max_relative_target=$Limite --robot.num_read_retries=10 `
    --robot.cameras="$CAMERE" `
    --teleop.type=so101_leader --teleop.port=$($p.Leader) --teleop.id=leader `
    --teleop.num_read_retries=10 `
    --display_data=$(if ($Finestra) { "true" } else { "false" })
