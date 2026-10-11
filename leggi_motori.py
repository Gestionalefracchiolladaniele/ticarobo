"""Sola lettura: chi risponde sul bus e cosa vede. Non scrive nulla, il braccio non si muove.

Registri STS3215 (tabella di controllo Feetech):
  56-57  posizione attuale (0..4095 = un giro, 2048 = centro)
  62     tensione attuale, in decimi di volt
  63     temperatura attuale, gradi C
"""

import sys

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler

PORTA = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
BAUD = 1_000_000  # quello usato da LeRobot per gli STS3215

porta = PortHandler(PORTA)
if not porta.openPort() or not porta.setBaudRate(BAUD):
    sys.exit(f"Non riesco ad aprire {PORTA} a {BAUD} baud")
bus = PacketHandler(0)  # protocollo Feetech serie STS

trovati = 0
for motore_id in range(0, 21):
    modello, esito, _ = bus.ping(porta, motore_id)
    if esito != COMM_SUCCESS:
        continue
    trovati += 1
    tensione, _, _ = bus.read1ByteTxRx(porta, motore_id, 62)
    temperatura, _, _ = bus.read1ByteTxRx(porta, motore_id, 63)
    posizione, _, _ = bus.read2ByteTxRx(porta, motore_id, 56)
    print(
        f"ID {motore_id:2d}  modello {modello}  tensione {tensione / 10:4.1f} V  "
        f"temp {temperatura:2d} C  posizione {posizione:4d}"
    )

porta.closePort()
print(f"\nMotori che rispondono: {trovati}")
