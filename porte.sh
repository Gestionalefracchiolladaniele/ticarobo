#!/bin/bash
# Trova la porta di ogni braccio dal numero di serie della sua scheda.
# ttyACM0/1 si scambiano a ogni ricollegamento, il seriale no.
# Si usa con: source porte.sh  ->  $FOLLOWER_PORT, $LEADER_PORT

FOLLOWER_SERIALE=5B90101906   # scheda bianca, 12 V
LEADER_SERIALE=5B79082554     # scheda nera, 5 V

porta_di() {
    for tty in /dev/ttyACM*; do
        [ -e "$tty" ] || continue
        usb_dir=$(dirname "$(readlink -f "/sys/class/tty/$(basename "$tty")/device")")
        [ "$(cat "$usb_dir/serial" 2>/dev/null)" = "$1" ] && { echo "$tty"; return; }
    done
}

FOLLOWER_PORT=$(porta_di "$FOLLOWER_SERIALE")
LEADER_PORT=$(porta_di "$LEADER_SERIALE")
