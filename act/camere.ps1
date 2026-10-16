# Trova le camere, salva una foto per ciascuna e le apre.
#
#   .\camere.ps1
#
# Serve a sapere quale indice e' la camera del polso ($CAMERA_POLSO in
# comune.ps1). Si guarda la foto, non il nome: "OpenCV Camera @ 1" non dice
# dove e' montata. Chiudi l'app Fotocamera prima: una camera, un'app.

. "$PSScriptRoot\comune.ps1"
Set-Location $ACT

# Via le foto vecchie: non si guarda per errore quella di un'altra volta.
Remove-Item "$ACT\outputs\captured_images\*" -ErrorAction SilentlyContinue

& "$VENV\lerobot-find-cameras.exe" opencv 2>&1 |
    Select-String -Pattern "Camera #|Id:|Width|Height|Fps|Format" | ForEach-Object { $_.Line }

Get-ChildItem "$ACT\outputs\captured_images\*.png" | ForEach-Object {
    Write-Host "Apro $($_.Name)"
    Start-Process $_.FullName
}
