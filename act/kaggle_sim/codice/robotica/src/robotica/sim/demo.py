"""Demo di verifica: la catena completa su un pick-and-place.

Esegue e stampa ogni passo — verifica di sicurezza, presa, trasporto, rilascio
— e ripete il compito piu' volte per mostrare che e' ripetibile, non riuscito
per caso.

    uv run python -m robotica.sim.demo
    uv run python -m robotica.sim.demo --runs 5
"""

from __future__ import annotations

import argparse

import numpy as np

from robotica.sim.env import WorkspaceEnv, table_surface_z
from robotica.sim.robot import OSCRobot

SEPARATORE = "=" * 62


def run(runs: int = 3) -> bool:
    """Esegue la demo. True se il compito riesce in tutte le esecuzioni."""
    print(SEPARATORE)
    print("  Robotica — pick and place con controller OSC")
    print(SEPARATORE)

    env, robot = _build()

    print(f"\nScena: {env.action_dim} comandi, oggetti: {', '.join(env.object_names())}")
    print(f"Piano del tavolo: z={table_surface_z():.3f}")
    print(f"Pinza: {robot.tcp_pos.round(3)}, apertura {robot.gripper_width() * 1000:.0f} mm")

    _mostra_sicurezza(robot)
    riusciti = _mostra_pick_and_place(env, robot, runs)

    print("\n" + SEPARATORE)
    ok = riusciti == runs
    print(f"  Risultato: {riusciti}/{runs} — {'OK' if ok else 'FALLITO'}")
    print(SEPARATORE)
    return ok


def _build() -> tuple[WorkspaceEnv, OSCRobot]:
    """Scena robosuite con controller OSC."""
    from robosuite.controllers import load_composite_controller_config

    config = load_composite_controller_config(controller="BASIC", robot="Panda")
    env = WorkspaceEnv(robots="Panda", controller_configs=config)
    env.reset()
    return env, OSCRobot(env)


def _mostra_sicurezza(robot: OSCRobot) -> None:
    """Cosa viene rifiutato, e con quale motivo.

    E' il pezzo che vale (D2): non che il comando venga bloccato, ma che il
    motivo basti a correggersi senza indovinare.
    """
    print("\n[1] Verifica di sicurezza")
    z = table_surface_z()

    casi = [
        ("sopra il tavolo, raggiungibile", np.array([0.0, -0.1, z + 0.15])),
        ("dentro il piano del tavolo", np.array([0.0, 0.0, z - 0.10])),
        ("fuori dal workspace in y", np.array([0.0, -0.90, z + 0.15])),
        ("troppo in alto", np.array([0.0, 0.0, z + 0.80])),
    ]

    for etichetta, punto in casi:
        report = robot.safety.verify_position(punto)
        stato = "AMMESSO" if report.ok else "RIFIUTATO"
        print(f"  {stato:9} {etichetta}")
        for violazione in report.errors:
            print(f"            -> {violazione.rule}: {violazione.detail}")


def _mostra_pick_and_place(env: WorkspaceEnv, robot: OSCRobot, runs: int) -> int:
    """Il compito, ripetuto. Restituisce quante volte e' riuscito.

    Ripeterlo non e' pedanteria: una presa che riesce una volta puo' essere
    fortuna, e la Fase 1 chiede un pick-and-place *ripetibile*.
    """
    print(f"\n[2] Pick and place: cube_red -> zona verde ({runs} esecuzioni)")

    zona = env.green_zone_center()
    riusciti = 0

    for i in range(runs):
        if i > 0:
            env.reset()
            robot = OSCRobot(env)

        esito = _un_ciclo(env, robot, zona)
        if esito is None:
            riusciti += 1
            scarto = np.linalg.norm(env.object_pos("cube_red")[:2] - zona[:2])
            print(f"  ok  esecuzione {i + 1}: cubo nella zona, scarto {scarto * 1000:.0f} mm")
        else:
            print(f"  NO  esecuzione {i + 1}: {esito}")

    return riusciti


def _un_ciclo(env: WorkspaceEnv, robot: OSCRobot, zona: np.ndarray) -> str | None:
    """Un pick-and-place. None se riuscito, altrimenti il motivo del fallimento."""
    res = robot.pick("cube_red")
    if not res:
        return res.reason

    res = robot.place(zona, "cube_red")
    if not res:
        return res.reason

    for _ in range(40):
        env.step(np.zeros(env.action_dim))

    if not env.object_in_zone("cube_red"):
        pos = env.object_pos("cube_red")
        return f"cubo posato fuori zona, in {pos.round(3)}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo pick-and-place")
    parser.add_argument("--runs", type=int, default=3, help="quante volte ripetere il compito")
    args = parser.parse_args()
    return 0 if run(args.runs) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
