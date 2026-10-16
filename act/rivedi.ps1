# Il follower rifa' un episodio registrato: verifica che i dati siano giusti.
#
#   .\rivedi.ps1 -Prova             episodio 0 del dataset "prova"
#   .\rivedi.ps1 -Episodio 7        episodio 7 di "caramelle"
#
# Leader non serve. Prima: follower nella posa di partenza. Rimetti le
# caramelle come in quell'episodio se vuoi vedere se le prende davvero.

param(
    [int]$Episodio = 0,
    [switch]$Prova,
    [int]$Limite = 15
)

. "$PSScriptRoot\comune.ps1"
Set-Location $ACT

$nome = if ($Prova) { "prova" } else { "caramelle" }
$f = Get-PortaDi $FOLLOWER_SERIALE
if (-not $f) { Write-Host "Follower non trovato: USB e 12 V collegati? Schede in WSL? (usbipd detach --all)"; exit 1 }
Write-Host "follower $f   dataset $nome   episodio $Episodio"

& "$VENV\lerobot-replay.exe" `
    --robot.type=so101_follower --robot.port=$f --robot.id=follower `
    --robot.max_relative_target=$Limite --robot.num_read_retries=10 `
    --dataset.repo_id="so101/$nome" `
    --dataset.root="$ACT\dati\so101\$nome" `
    --dataset.episode=$Episodio
