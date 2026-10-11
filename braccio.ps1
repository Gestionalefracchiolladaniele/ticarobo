# Collega le schede del SO-101 a WSL e apre un terminale pronto per LeRobot.
#
#   .\braccio.ps1             collega e apre il terminale Ubuntu (venv attivo)
#   .\braccio.ps1 -SoloPorte  collega e basta
#
# Ogni scheda e' un convertitore CH343 (1a86:55d3). Il bind si fa una volta per
# scheda e chiede UAC; l'attach va rifatto a ogni ricollegamento e ogni volta
# che WSL si spegne per inattivita'. Finche' il terminale aperto da questo
# script resta aperto, WSL non si spegne e la scheda resta collegata.

param(
    [switch]$SoloPorte
)

$usbipd = "C:\Program Files\usbipd-win\usbipd.exe"
$schede = & $usbipd list | Select-String "1a86:55d3"
if (-not $schede) {
    Write-Host "Nessuna scheda CH343 trovata: il cavo USB e' collegato?"
    exit 1
}

# L'attach fallisce su una VM spenta, e WSL si spegne appena l'ultimo comando
# finisce: un 'wsl -- true' non basta (la conferma UAC dura piu' di lui).
# Questo processo nascosto tiene accesa la VM per tutta la durata dello script.
$tienimiAcceso = Start-Process wsl -ArgumentList "-d robotica -- sleep 60" -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 2

foreach ($riga in $schede) {
    $busid = ($riga.Line.Trim() -split "\s+")[0]
    if ($riga.Line -match "Not shared") {
        Write-Host "Scheda $busid mai condivisa: bind (conferma UAC)"
        Start-Process -FilePath $usbipd -ArgumentList "bind --busid $busid" -Verb RunAs -Wait
    }
    if ($riga.Line -notmatch "Attached") {
        & $usbipd attach --wsl --busid $busid
    }
}

wsl -d robotica -- bash "/mnt/f/cartella prova/lerobot/prepara_porte.sh" @($schede).Count
if ($LASTEXITCODE -ne 0) { exit 1 }

if (-not $SoloPorte) {
    wsl -d robotica --cd "/mnt/f/cartella prova/lerobot" -- bash --rcfile .venv/bin/activate -i
}
