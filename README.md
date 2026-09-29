# Uplift readout: should everyone keep seeing this campaign?

An experiment readout and uplift-targeting analysis of the **Criteo Uplift Prediction
Dataset v2.1** (13,979,592 users; randomized ad-eligibility tests with visit and
conversion outcomes and 12 anonymized pre-treatment features), written as a decision
memo for a growth team.

**Report:** https://uplift.levelbrook.com

## What is in it

1. **Readout**: sample-ratio-mismatch test against the 85/15 design, covariate balance
   (it fails, and the report says why and what to do about it), ATE on visit and
   conversion with analytic and bootstrap CIs, relative lift, an exposure (Wald) ratio,
   variance reduction with CUPED, Lin regression adjustment and CUPAC, a doubly robust
   AIPW headline estimate, and segment effects with Holm and Benjamini-Hochberg correction.
2. **Power and design**: MDE / sample-size calculator anchored to the observed control
   conversion rate, the 85/15 allocation penalty, a peeking simulation, and the fixes
   (Lan-DeMets O'Brien-Fleming alpha spending, mSPRT).
3. **Uplift modeling**: S-, T-, X-learner and transformed-outcome learners on LightGBM,
   evaluated on 5 disjoint held-out folds with uplift curves, Qini coefficient and AUUC
   vs random, calibration by decile, and the policy value of "treat the top k%" as a
   function of the cost-to-value ratio.
4. **Recommendation and limitations** in plain English.

## Run it

```bash
make setup      # uv venv + pinned requirements
make data       # download (~300 MB), sha256 check, convert to parquet
make analysis   # ~35 min on a laptop, peak memory < 2 GB
make site       # renders site/index.html from results/*.json
make test       # pytest on the statistics functions
```

Everything that touches all 14M rows runs as DuckDB aggregates or streams in chunks;
nothing loads the full table into pandas. Every number on the report page is read from
`results/*.json` by `scripts/build_site.py`; the template contains no typed figures.

## Layout

```
src/uplift_readout/stats.py       SRM, diff-in-means, delta-method lift, bootstrap, CUPED/Lin, Holm/BH, power
src/uplift_readout/sequential.py  peeking simulation, alpha-spending boundaries, mSPRT
src/uplift_readout/uplift.py      S/T/X/transformed-outcome learners, uplift curves, Qini, policy value
scripts/get_data.py               download + checksum + parquet (stable row id, cross-fit fold)
scripts/nuisance.py               cross-fitted propensity + outcome models, per-row AIPW scores
scripts/readout.py                part 1
scripts/power.py                  part 2
scripts/uplift_models.py          part 3
scripts/build_site.py             charts + HTML
tests/test_stats.py               CUPED unbiasedness/coverage, SRM, bootstrap, AIPW, curves, sequential
```

## Data and licence

Criteo AI Lab, *Criteo Uplift Prediction Dataset v2.1*, CC BY-NC-SA 4.0
(https://ailab.criteo.com/criteo-uplift-prediction-dataset/, mirrored at
https://huggingface.co/datasets/criteo/criteo-uplift). Cite: E. Diemert, A. Betlei,
C. Renaudin, M.-R. Amini, "A Large Scale Benchmark for Uplift Modeling," AdKDD &
TargetAd Workshop, KDD 2018. The raw data is not committed; `make data` fetches it.

Code is MIT-licensed; the derived report and result files follow the dataset's
CC BY-NC-SA 4.0 terms.
