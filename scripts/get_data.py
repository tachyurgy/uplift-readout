"""Download the Criteo Uplift v2.1 file (if absent), verify it, convert to parquet.

Source: Criteo AI Lab, https://ailab.criteo.com/criteo-uplift-prediction-dataset/
Mirror used here: https://huggingface.co/datasets/criteo/criteo-uplift
Licence: CC BY-NC-SA 4.0. Paper: Diemert, Betlei, Renaudin, Amini (2018),
"A Large Scale Benchmark for Uplift Modeling", AdKDD workshop at KDD 2018.
"""
import hashlib
import subprocess
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GZ = DATA / "criteo-research-uplift-v2.1.csv.gz"
PQ = DATA / "criteo.parquet"
URLS = [
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-research-uplift-v2.1.csv.gz",
    "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz",
]
SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    DATA.mkdir(exist_ok=True)
    if not GZ.exists():
        for u in URLS:
            if subprocess.run(["curl", "-fL", "-o", str(GZ), u]).returncode == 0:
                break
    got = sha(GZ)
    if got != SHA256:
        raise SystemExit(f"checksum mismatch: {got}")
    if not PQ.exists():
        c = duckdb.connect()
        c.execute("SET memory_limit='3GB'; SET preserve_insertion_order=true")
        # rid = position in the source file; fold = deterministic 50/50 split used for cross-fitting
        c.execute(f"""COPY (SELECT rid, (hash(rid) % 2)::TINYINT AS fold, * EXCLUDE (rid)
                     FROM (SELECT row_number() OVER () AS rid, * FROM read_csv_auto('{GZ}')))
                     TO '{PQ}' (FORMAT parquet)""")
    n = duckdb.sql(f"SELECT count(*) FROM '{PQ}'").fetchone()[0]
    print(f"{PQ.name}: {n:,} rows")


if __name__ == "__main__":
    main()
