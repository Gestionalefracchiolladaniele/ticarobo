#!/bin/bash
# Parte Linux di braccio.ps1. La distro non ha systemd: modules-load.d e le
# regole udev vengono ignorate, quindi driver e permessi si rimettono a mano.
sudo -n modprobe cdc_acm || { echo "modprobe cdc_acm fallito"; exit 1; }

# Si aspettano tutte le schede (argomento 1, default 1): aspettarne una sola
# lasciava la seconda, arrivata un attimo dopo, senza permessi.
attese=${1:-1}
for i in $(seq 1 10); do
    [ "$(ls /dev/ttyACM* 2>/dev/null | wc -l)" -ge "$attese" ] && break
    sleep 1
done
if ! ls /dev/ttyACM* >/dev/null 2>&1; then
    echo "Nessuna porta /dev/ttyACM*: la scheda e' collegata e attaccata a WSL?"
    exit 1
fi

# ttyACM0/1 possono scambiarsi a ogni collegamento: il numero di serie no.
for tty in /dev/ttyACM*; do
    sudo -n chmod 666 "$tty"
    usb_dir=$(dirname "$(readlink -f "/sys/class/tty/$(basename "$tty")/device")")
    echo "$tty  seriale $(cat "$usb_dir/serial" 2>/dev/null || echo '?')"
done
