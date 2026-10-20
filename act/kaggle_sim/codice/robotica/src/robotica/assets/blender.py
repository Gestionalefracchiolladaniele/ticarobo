"""Invocare Blender per preparare gli asset 3D.

Blender qui serve a **produrre e ripulire mesh**, non a generare dataset di
training: i modelli foundation zero-shot (FoundationPose, SAM-6D) hanno reso
superata quella strada. Vedi docs/percezione.md.

Gli usi che restano:

- convertire CAD (STEP, OBJ, FBX) nei formati che MuJoCo accetta (STL, OBJ)
- **semplificare** mesh troppo dense: una mesh CAD da 200k triangoli rallenta
  la simulazione senza migliorarla
- generare geometria da regole, quando serve una scena che non esiste

Blender gira in modalita' ``--background``: nessuna finestra, tutto da script.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from robotica.paths import CACHE, ROOT

BLENDER = ROOT / "tools" / "blender-5.2.1-windows-x64" / "blender.exe"
TIMEOUT = 600


def to_windows_path(path: Path | str) -> str:
    """Traduce un percorso WSL in uno Windows.

    Blender qui e' un eseguibile Windows: WSL lo avvia (interop), ma il
    processo gira lato Windows e non sa cosa sia ``/mnt/f/...``. Ogni percorso
    passato come argomento va tradotto in ``F:\\...``.
    """
    s = str(Path(path))
    if s.startswith("/mnt/") and len(s) > 6 and s[6] == "/":
        return f"{s[5].upper()}:\\" + s[7:].replace("/", "\\")
    return s


def available() -> bool:
    """Blender e' presente e invocabile?"""
    return BLENDER.is_file()


def version() -> str:
    """Versione di Blender, o una stringa d'errore."""
    if not available():
        return f"non trovato in {BLENDER}"
    out = _run(["--version"])
    return out.splitlines()[0] if out else "sconosciuta"


def run_script(script: Path | str, *args: str, blend_file: Path | str | None = None) -> str:
    """Esegue uno script Python dentro Blender, senza interfaccia.

    Gli argomenti dopo ``--`` arrivano allo script in ``sys.argv``: e' la
    convenzione di Blender per distinguerli dai propri.
    """
    cmd: list[str] = []
    if blend_file is not None:
        cmd.append(to_windows_path(blend_file))
    cmd += ["--python", to_windows_path(script)]
    if args:
        cmd += ["--", *args]
    return _run(cmd)


def convert_mesh(source: Path | str, dest: Path | str, max_faces: int | None = None) -> Path:
    """Converte una mesh e opzionalmente la semplifica.

    ``max_faces`` applica un decimate: indispensabile per le mesh CAD, che
    hanno spesso centinaia di migliaia di triangoli. Per la collisione in
    MuJoCo ne bastano poche migliaia, e la differenza in velocita' e' enorme.
    """
    source, dest = Path(source), Path(dest)
    if not source.is_file():
        raise FileNotFoundError(source)
    dest.parent.mkdir(parents=True, exist_ok=True)

    script = Path(__file__).parent / "_convert_mesh.py"
    run_script(
        script,
        to_windows_path(source), to_windows_path(dest), str(max_faces or 0),
    )
    if not dest.is_file():
        raise RuntimeError(f"conversione fallita: {dest} non prodotto")
    return dest


def crea_caramelle(misure: dict, dest: Path | str) -> list[Path]:
    """Caramelle e bicchiere per la scena ACT, in OBJ, dalle ``misure`` (metri).

    Le misure le dichiara ``robotica.sim.so101_caramelle``: qui si costruisce
    solo la forma da vedere. Il JSON passa da ``out/cache``, su F: come tutto.
    """
    import json

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    argomenti = CACHE / "misure_caramelle.json"
    argomenti.write_text(json.dumps(misure, indent=1))
    out = run_script(
        Path(__file__).parent / "_caramelle.py", to_windows_path(argomenti), to_windows_path(dest)
    )
    file = [dest / f"{nome}.obj" for nome in misure]
    mancanti = [f for f in file if not f.is_file()]
    if mancanti:
        raise RuntimeError(f"Blender non ha prodotto {mancanti}:\n{out[-2000:]}")
    return file


def _run(args: list[str]) -> str:
    """Lancia Blender in background e restituisce lo stdout."""
    if not available():
        raise FileNotFoundError(f"Blender non trovato in {BLENDER}")
    proc = subprocess.run(
        [str(BLENDER), "--background", *args],
        capture_output=True, text=True, timeout=TIMEOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Blender exit {proc.returncode}:\n{proc.stderr[-2000:]}")
    return proc.stdout
