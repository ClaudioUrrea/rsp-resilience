# Code, raw episode data and figures for "An Axiomatic Resilience Functional for Fault-Tolerant Control of Redundant Manipulators"

This record archives everything needed to reproduce, line by line, the numerical
study of

> C. Urrea, *An Axiomatic Resilience Functional for Fault-Tolerant Control of
> Redundant Manipulators: Representation, Rank Invariance, and Sample
> Complexity*, **Mathematics** (MDPI), 2026.

Development repository: <https://github.com/ClaudioUrrea/rsp-resilience>
Release archived here: **v1.0.0**

## Files

| File | Contents |
|---|---|
| `rsp-resilience-v1.0.0-code.zip` | The complete source: dynamics, five plants, eight fault classes, eight control laws, the episode engine, the exact task margin, the scoring functional and the statistics, plus the seven scripts that regenerate the paper, verify it and assemble this record. |
| `rsp-resilience-v1.0.0-data.zip` | `results/raw_<plant>.npz` — the raw episode-level criteria of all 32,000 evaluation episodes — together with `meta.json` (calibrated detection thresholds, reference constants, execution times, full configuration, seed) and `summary.json` (every headline number quoted in the paper). |
| `rsp-resilience-v1.0.0-figures.zip` | The seven figures as published, plus `tables/tables.tex`, which carries the complete 28-row pairwise comparison of which Table 5 of the paper prints a selection. |
| `CHECKSUMS.sha256` | SHA-256 of each archive. |
| `ENVIRONMENT.txt` | Interpreter, platform and library versions under which the deposited results were produced. |

## Reproducing

```bash
unzip rsp-resilience-v1.0.0-code.zip && cd rsp-resilience-v1.0.0
pip install -r requirements.txt
python scripts/verify_dynamics.py                      # ~30 s
python scripts/run_experiments.py --episodes 100 --calib 48 --seed 20260811
python scripts/make_analysis.py
```

The campaign takes about 26 minutes on a single core and rewrites `results/`.
To skip it, unzip `...-data.zip` into `results/` and run `make_analysis.py`
directly: the raw criteria are stored, so scores, rankings and the rank-reversal
distance `kappa*` are recomputed from them without any further simulation, under
the nominal weights or under any others.

## Checking rather than trusting

Two scripts let a reader audit the record without re-running anything.

```bash
python scripts/verify_deposit.py         # is the archive complete and coherent?
python scripts/verify_paper_claims.py    # does the paper agree with the data?
```

The first validates the structure of every `raw_<plant>.npz` — keys, shapes,
balance of the eight fault classes, the range of the fault instant, the sharing
of the fault design across controllers — and recomputes the 40 medians, the 40
outright-failure rates and the paired dispersion, diffing them against
`summary.json`.

The second recomputes all **70 numerical assertions of the manuscript** from the
raw episodes — the seven rows of Table 5 including Cliff's delta and the
Holm-adjusted p-values, the captions of Figures 3 to 6, the episode budgets, the
sliding-mode fragility under variable latency, the sensitivity analysis of
Section 8.6 — and prints one PASS/FAIL line per claim. On the deposited data it
reports zero disagreements.

## What the data contains

Each `raw_<plant>.npz` holds, for every one of the eight controllers `c`
(`PID`, `CTC`, `SMC`, `FT1`, `FT2`, `ANN`, `AMPC`, `ADRC`):

| Key | Shape | Meaning |
|---|---|---|
| `X_<c>` | (800, 6) | raw criteria `x_1 … x_6`: post-fault tracking error [m], detection latency [s], reconfiguration transient [s], post-fault mean power [W], execution time as a fraction of the sampling period, fraction of the post-fault window in constraint violation. `+inf` marks outright failure. |
| `declared_<c>` | (800,) | the supervisor declared a fault |
| `degraded_<c>` | (800,) | the tracking error left its pre-fault envelope |
| `alive_<c>` | (800,) | the episode completed without divergence |
| `ratio_<c>` | (800,) | post/pre-fault error ratio, retained only as a diagnostic |
| `fid`, `t_f`, `idx` | (800,) | fault class (0–7 = F1–F8), fault instant [s] and faulty actuator index — identical across controllers, which is what makes the comparisons paired |

The 800 rows of each array are 8 fault classes × 100 episodes, in that order.
Reference constants and weights needed to turn `X` into a score are in
`meta.json` (`xstar`) and in `rsp/score.py` (`W_NOMINAL`). Two conventions
matter when scoring the raw values by hand: `xstar['x4']` is already the energy
budget, three times the pooled median nominal power, not the raw median (the
median itself is kept in `xstar_aux`); and raw detection latencies are censored
at 1.2 s, and set to zero for episodes in which the fault was never
consequential, before the normalization of Definition 2 is applied. Both steps
are performed by `scripts/make_analysis.py`.

## Scope

The study is entirely computational. No hardware, no human subjects and no
personal data are involved: the objects being validated are theorems, and the
simulations verify their bounds on concrete dynamics.

## Versions

**v1.0.0** — first public release, accompanying the article.

## License

MIT (code and data). Please cite the article and this record; `CITATION.cff` in
the code archive carries the machine-readable form.
