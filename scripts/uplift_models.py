"""Part 3: uplift modeling with honest held-out evaluation.

Rows are split into 5 disjoint folds by a hash of their source row id. For
each fold: train every learner on a 1.5M-row random sample of the OTHER four
folds, then score all ~2.8M rows of the held-out fold. Curves, AUUC and policy
values are computed only on held-out rows, and the spread across the 5 folds
gives the error bars.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np

from uplift_readout import uplift as U

ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "criteo.parquet"
OUT = ROOT / "results"
FEATS = [f"f{i}" for i in range(12)]
K = 5
N_TRAIN = 1_500_000
GRID = np.round(np.linspace(0, 1, 101), 4)
DECILES = 10


def load(c, where, sample=None):
    q = f"SELECT {','.join(FEATS)}, treatment, visit, conversion FROM d WHERE {where}"
    if sample:
        q += f" USING SAMPLE {sample} ROWS (reservoir, 7)"
    a = c.execute(q).fetchnumpy()
    X = np.column_stack([a[f] for f in FEATS]).astype(np.float32)
    return X, a["treatment"].astype(np.int8), {y: a[y].astype(np.int8) for y in ("visit", "conversion")}


def decile_calibration(tau_hat, t, y):
    """Observed uplift (difference in means) inside each predicted-uplift decile."""
    edges = np.quantile(tau_hat, np.linspace(0, 1, DECILES + 1))
    b = np.clip(np.searchsorted(edges, tau_hat, side="right") - 1, 0, DECILES - 1)
    out = []
    for i in range(DECILES):
        m = b == i
        tt, yy = t[m].astype(bool), y[m]
        out.append({"decile": i + 1, "pred": float(tau_hat[m].mean()),
                    "obs": float(yy[tt].mean() - yy[~tt].mean()),
                    "se": float(np.sqrt(yy[tt].var() / tt.sum() + yy[~tt].var() / (~tt).sum()))})
    return out


def main():
    t0 = time.time()
    c = duckdb.connect()
    c.execute("SET memory_limit='2GB'; SET threads=4")
    c.execute(f"CREATE VIEW d AS SELECT *, (hash(rid * 7919 + 13) % {K})::INTEGER AS k5 FROM read_parquet('{PARQUET}')")
    e = c.execute("SELECT avg(treatment) FROM d").fetchone()[0]
    res = {"k": K, "n_train": N_TRAIN, "propensity": e, "grid": GRID.tolist(), "folds": []}
    for k in range(K):
        Xtr, ttr, ytr = load(c, f"k5 <> {k}", N_TRAIN)
        Xte, tte, yte = load(c, f"k5 = {k}")
        fold = {"fold": k, "n_test": int(len(tte)), "outcomes": {}}
        for y in ("visit", "conversion"):
            seed = 100 * k + (y == "conversion")
            preds = {}
            preds["S-learner"] = U.s_learner(Xtr, ttr, ytr[y], Xte, seed)
            tau_t, m1, m0 = U.t_learner(Xtr, ttr, ytr[y], Xte, seed, return_models=True)
            preds["T-learner"] = tau_t
            preds["X-learner"] = U.x_learner(Xtr, ttr, ytr[y], Xte, e, seed, outcome_models=(m1, m0))
            preds["Transformed outcome"] = U.transformed_outcome(Xtr, ttr, ytr[y], Xte, e, seed)
            preds["Random"] = np.random.default_rng(seed).random(len(tte))
            ores = {}
            for name, tau in preds.items():
                g, curve = U.uplift_curve(tau, tte, yte[y], GRID)
                ores[name] = {"curve": curve.tolist(), "auuc": U.auuc(g, curve),
                              "qini": U.qini_coefficient(g, curve),
                              "pred_mean": float(tau.mean()), "pred_share_negative": float((tau < 0).mean()),
                              "calibration": decile_calibration(tau, tte, yte[y]) if name != "Random" else None}
            fold["outcomes"][y] = ores
            print(f"fold {k} {y}: " + ", ".join(f"{n} qini={v['qini']:.3f}" for n, v in ores.items()),
                  f"[{time.time() - t0:.0f}s]", flush=True)
        res["folds"].append(fold)
        del Xtr, Xte
    res["runtime_s"] = time.time() - t0
    (OUT / "uplift.json").write_text(json.dumps(res))


if __name__ == "__main__":
    sys.exit(main())
