"""Cross-fitted nuisance models and per-row AIPW scores for the full table.

Why this exists: the released sample passes the SRM check but fails covariate
balance (see readout.json), consistent with the dataset card's note that the
data pools several incrementality tests and was sub-sampled non-uniformly for
privacy. So assignment is not exactly independent of the features, and the
analysis treats it as "randomized conditional on X" instead.

For each of the two folds (fixed by hash of the source row id) we fit, on the
OTHER fold only:
  e(x)      propensity P(T=1 | X)           1.5M-row sample, both arms
  mu1_y(x)  P(Y=1 | X, T=1) per outcome     1.5M treated rows
  mu0_y(x)  P(Y=1 | X, T=0) per outcome     every control row in the fold (~1.05M)
and score this fold. The AIPW (doubly robust) score
  psi = mu1 - mu0 + T (Y - mu1) / e - (1 - T)(Y - mu0) / (1 - e)
has mean equal to the ATE if either the propensity or the outcome models are
right, and its sample variance gives the standard error. Averaging psi inside
a segment gives that segment's effect, which is how heterogeneity is read.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "criteo.parquet"
DST = ROOT / "data" / "nuisance.parquet"
FEATS = [f"f{i}" for i in range(12)]
OUTCOMES = ["visit", "conversion"]
E_CLIP = (0.02, 0.98)
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=500,
              subsample=0.8, subsample_freq=1, n_jobs=4, verbose=-1)


def fetch(c, q):
    a = c.execute(q).fetchnumpy()
    return np.column_stack([a[f] for f in FEATS]).astype(np.float32), a


def main():
    if DST.exists():
        print("nuisance.parquet exists; delete it to refit")
        return
    c = duckdb.connect()
    c.execute("SET memory_limit='2GB'; SET threads=4")
    c.execute(f"CREATE VIEW d AS SELECT * FROM read_parquet('{PARQUET}')")
    writer = None
    for fold in (0, 1):
        src = 1 - fold
        X, a = fetch(c, f"SELECT {','.join(FEATS)}, treatment FROM d WHERE fold={src} USING SAMPLE 1500000 ROWS (reservoir, 11)")
        prop = lgb.LGBMClassifier(random_state=fold, **PARAMS).fit(X, a["treatment"])
        X1, a1 = fetch(c, f"SELECT {','.join(FEATS)}, visit, conversion FROM d WHERE fold={src} AND treatment=1 USING SAMPLE 1500000 ROWS (reservoir, 12)")
        X0, a0 = fetch(c, f"SELECT {','.join(FEATS)}, visit, conversion FROM d WHERE fold={src} AND treatment=0")
        mu = {}
        for y in OUTCOMES:
            mu[(1, y)] = lgb.LGBMClassifier(random_state=fold, **PARAMS).fit(X1, a1[y])
            mu[(0, y)] = lgb.LGBMClassifier(random_state=fold, **PARAMS).fit(X0, a0[y])
        del X, X1, X0, a, a1, a0
        c.execute(f"SELECT rid, {','.join(FEATS)}, treatment, visit, conversion FROM d WHERE fold={fold}")
        while True:
            b = c.fetch_df_chunk(20)
            if len(b) == 0:
                break
            Xb = b[FEATS].to_numpy(np.float32)
            t = b["treatment"].to_numpy(np.float64)
            e = np.clip(prop.predict_proba(Xb)[:, 1], *E_CLIP)
            cols = {"rid": b["rid"].to_numpy(), "e_hat": e}
            for y in OUTCOMES:
                yy = b[y].to_numpy(np.float64)
                m1 = mu[(1, y)].predict_proba(Xb)[:, 1]
                m0 = mu[(0, y)].predict_proba(Xb)[:, 1]
                cols[f"mu1_{y}"] = m1
                cols[f"mu0_{y}"] = m0
                cols[f"psi_{y}"] = m1 - m0 + t * (yy - m1) / e - (1 - t) * (yy - m0) / (1 - e)
            tbl = pa.table(cols)
            writer = writer or pq.ParquetWriter(DST, tbl.schema)
            writer.write_table(tbl)
        print(f"fold {fold} scored", flush=True)
    writer.close()


if __name__ == "__main__":
    main()
