"""Percorsi del progetto. Unico posto dove si risolvono path su disco."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCENES = ROOT / "scenes"
OUT = ROOT / "out"
MENAGERIE = ROOT / "third_party" / "mujoco_menagerie"

# Roba generata e riutilizzabile, non versionata. Ci finisce l'URDF preparato
# per mplib e l'SRDF che ne deriva: calcolarli costa ~40 secondi e il risultato
# non cambia mai, quindi si fa una volta sola.
CACHE = ROOT / "out" / "cache"

# Robot disponibili: nome breve -> cartella in mujoco_menagerie.
# Sono i modelli ufficiali dei produttori, la cinematica e' quella reale.
ROBOTS = {
    "panda": "franka_emika_panda",
    "fr3": "franka_fr3",
    "ur5e": "universal_robots_ur5e",
    "ur10e": "universal_robots_ur10e",
    # Non trs_so_arm100: e' il SO-ARM100, un braccio diverso (giunti e quote).
    "so101": "robotstudio_so101",
}


def robot_dir(name: str) -> Path:
    """Cartella del modello robot. Solleva se il nome o il clone mancano."""
    if name not in ROBOTS:
        raise KeyError(f"robot '{name}' sconosciuto; disponibili: {sorted(ROBOTS)}")
    path = MENAGERIE / ROBOTS[name]
    if not path.is_dir():
        raise FileNotFoundError(
            f"modello non trovato in {path}. "
            "Clonare mujoco_menagerie in third_party/ (vedi scripts/setup.sh)"
        )
    return path
