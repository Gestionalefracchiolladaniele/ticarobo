#!/bin/bash
# Calibrazione di un braccio: bash calibra.sh follower | leader
# Sovrascrive la calibrazione esistente: rifarla solo se serve (vedi README).
cd "$(dirname "$0")"
source porte.sh

case "$1" in
    follower)
        [ -n "$FOLLOWER_PORT" ] || { echo "Follower non trovato"; exit 1; }
        .venv/bin/lerobot-calibrate --robot.type=so101_follower \
            --robot.port="$FOLLOWER_PORT" --robot.id=follower ;;
    leader)
        [ -n "$LEADER_PORT" ] || { echo "Leader non trovato"; exit 1; }
        .venv/bin/lerobot-calibrate --teleop.type=so101_leader \
            --teleop.port="$LEADER_PORT" --teleop.id=leader ;;
    *)
        echo "uso: bash calibra.sh follower | leader"; exit 1 ;;
esac
