#!/bin/bash
# Teleoperazione: muovi il leader (nero), il follower (bianco) ti copia.
#
#   bash teleop.sh        limite 5 gradi per comando (prudente, il follower insegue)
#   bash teleop.sh 15     limite 15 gradi (piu' reattivo)
#
# Prima: i due bracci nella stessa posa, ripiegati. Per fermare: leader
# ripiegato in basso, poi Ctrl+C (il follower perde la coppia e si appoggia).
cd "$(dirname "$0")"
source porte.sh
LIMITE=${1:-5}

[ -n "$FOLLOWER_PORT" ] || { echo "Follower non trovato: 12 V e USB collegati? braccio.ps1 lanciato?"; exit 1; }
[ -n "$LEADER_PORT" ]   || { echo "Leader non trovato: 5 V e USB collegati? braccio.ps1 lanciato?"; exit 1; }
echo "follower $FOLLOWER_PORT   leader $LEADER_PORT   limite ${LIMITE} gradi/comando"

# num_read_retries=10: via usbipd ogni tanto una risposta arriva tardi, e con
# i 2 tentativi di default la teleoperazione si ferma ("no status packet").
.venv/bin/lerobot-teleoperate \
    --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id=follower \
    --robot.max_relative_target="$LIMITE" --robot.num_read_retries=10 \
    --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id=leader \
    --teleop.num_read_retries=10
