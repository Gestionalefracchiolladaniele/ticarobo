"""Ripara ``meta/episodes/file_index`` di un dataset LeRobot nato da un merge.

``caramelle_50`` (5 + 45, merge del 16/10) dice che i suoi 50 episodi sono
descritti in 5 file di metadati (indici 0-4), ma il file e' uno solo e li
contiene tutti: il merge ha copiato gli indici delle registrazioni originali
(fatte in piu' sessioni, ``-Riprendi``). L'addestramento non usa quel campo;
``lerobot-edit-dataset merge`` si', e cade su ``file-001.parquet``.

Si lavora **su una copia**, mai sull'originale: qui si scrive solo dove le
righe stanno davvero. Verifica prima di scrivere che i file esistenti siano
esattamente quelli dichiarati dopo la riparazione.

    python scripts/ripara_meta_episodi.py <copia del dataset>
"""

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    radice = Path(sys.argv[1])
    file = sorted((radice / "meta" / "episodes").rglob("*.parquet"))
    if len(file) != 1:
        raise SystemExit(f"attesi un solo file di metadati episodi, trovati {len(file)}: {file}")
    percorso = file[0]
    chunk = int(percorso.parent.name.split("-")[1])
    indice = int(percorso.stem.split("-")[1])
    e = pd.read_parquet(percorso)
    prima = e["meta/episodes/file_index"].value_counts().sort_index().to_dict()
    e["meta/episodes/chunk_index"] = chunk
    e["meta/episodes/file_index"] = indice
    e.to_parquet(percorso, index=False)
    print(f"{radice.name}: {len(e)} episodi, meta/episodes/file_index {prima} -> tutti {indice}")


if __name__ == "__main__":
    main()
