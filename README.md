# Axiomatic Resilience Scoring for Fault-Tolerant Control of Redundant Manipulators

Reference implementation of the theory and of the numerical study reported in

> C. Urrea, *An Axiomatic Resilience Functional for Fault-Tolerant Control of
> Redundant Manipulators: Representation, Rank Invariance, and Sample
> Complexity*, **Mathematics** (MDPI), 2026.

Everything in the paper is produced by this repository. There is no hardware in
the loop and none is required: the object being validated is a set of theorems,
and the simulations verify their bounds on concrete dynamics.

* Code: <https://github.com/ClaudioUrrea/rsp-resilience>
* Archived release, raw episode-level data and figures: Figshare,
  DOI [10.6084/m9.figshare.33214149](https://doi.org/10.6084/m9.figshare.33214149)

---

## 1. What the code does

| Module | Contents |
|---|---|
| `rsp/dynamics.py` | Batched rigid-body dynamics of open serial chains in standard DH form. Mass matrix from per-link body Jacobians; bias vector from one recursive Newton–Euler pass. All routines carry a leading batch axis so that a whole Monte-Carlo cell advances with one set of array operations. |
| `rsp/plants.py` | The five plants. P5 uses the manufacturer-published UR16e kinematic and dynamic parameters. P4 is a lumped reduced model of a four-arm delta with actuation redundancy. |
| `rsp/faults.py` | Fault classes F1–F8 of Definition 1, Latin-hypercube parameter sampling, and the communication channel (hold, loss, variable latency, missed cycle). |
| `rsp/controllers.py` | The eight control laws. They share the redundancy-resolution layer, the detection module and the actuator-allocation step, and differ only in the task-space acceleration command and in the fidelity of the model they may use. |
| `rsp/simulate.py` | The episode engine: integration, fault injection, residual-based detection, effectiveness estimation, criterion accumulation. |
| `rsp/margin.py` | The post-fault task margin of Section 4: the Chebyshev radius of the zonotope of attainable task velocities, computed **exactly** by enumerating the facet normals. |
| `rsp/score.py` | Normalization (Definition 2), the power means `R_p`, the rank-reversal margin `kappa` (sufficient bound and exact value), the empirical Lipschitz ratio. |
| `rsp/stats.py` | Hoeffding and Bernstein episode budgets, the Maurer–Pontil empirical Bernstein interval, Wilcoxon signed-rank, Holm correction, Cliff's delta with bootstrap CI. |

Why the margin is enumerated rather than sampled: `mu` is a *minimum* of the
support function over the unit sphere, so estimating it by sampling directions
can only overestimate it, and the overestimate may exceed the upper bound of
Proposition 1 that Figure 1 is meant to verify. Facet enumeration is exact and,
at these dimensions, cheaper.

## 2. Requirements

Python 3.10 or later.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy, scipy, matplotlib
```

No GPU, no compiler, no robotics toolbox is needed. The campaign fits in under
2 GB of RAM on a single core.

## 3. Reproducing the paper, step by step

### Step 1 — verify the dynamics

```bash
python scripts/verify_dynamics.py
```

Checks, on random configurations of every plant, that

* the mass matrix is symmetric (to better than `1e-10`) and positive definite;
* the recursive Newton–Euler torque equals `M(q) qdd + h(q,qd)` to a relative
  error below `1e-12` — the two are computed by independent algorithms, and the
  error observed on every plant is of order `3e-16`;
* the total energy of the unforced, frictionless system is conserved over 2 s
  at the integration step used. Explicit Euler at `dt = 1 ms` gives a relative
  drift of `4.1e-2` on P1, `9.8e-3` on P3, `6.0e-3` on P5 and `2.2e-3` on P2;
  the script fails above `5e-2`.

Nothing else should be run until this passes.

### Step 2 — run the Monte-Carlo campaign

```bash
python scripts/run_experiments.py --episodes 100 --calib 48 --seed 20260811
```

This is the only expensive step. It performs, for each of the five plants:

1. **Detector calibration.** 48 fault-free episodes per controller. The
   per-actuator residual thresholds are set to 1.30 times the 99th percentile of
   the peak filtered residual observed after the settling time, and the
   task-error threshold likewise. This fixes the pre-fault false-alarm rate at
   about 1% *by construction* and removes hand-tuned thresholds from the study.
2. **Reference constant `x_4^star`.** A *second, independent* set of 48
   fault-free episodes per controller, run with the calibrated thresholds
   already active. The constant is three times the median nominal power, pooled
   over controllers so that it does not depend on the controller being scored.
   Steps 1 and 2 together are the 3,840 calibration episodes reported in the
   paper (2 × 48 × 40 plant–controller pairs).
3. **Execution-time measurement.** 400 unbatched evaluations of each control
   law, used for criterion C5. This quantity is implementation- and
   hardware-dependent; the reference machine is declared in the paper.
4. **The campaign proper.** 8 fault classes × `--episodes` episodes per
   controller, i.e. 800 episodes per plant–controller pair at the default
   setting, which exceeds the 738 required by Theorem 4(i) for an absolute
   claim at `eps = delta = 0.05`. All controllers see identical episode
   realizations (common random numbers): the initial configuration, fault
   instant, fault class, severity, measurement noise, disturbance and
   communication events are derived from the episode index alone.

Outputs: `results/raw_<plant>.npz` (one row per episode, six raw criteria) and
`results/meta.json` (thresholds, reference constants, timings, full
configuration).

`meta.json` stores the reference constants **in the form used for scoring**:
`x4` is already the energy budget (three times the median nominal power) and
`x2` is 0.30 s. The raw pooled median power and the latency cap are kept
alongside in `xstar_aux` for traceability. `make_analysis.py` reads these
constants and rescales nothing.

Wall-clock time on the reference machine: about 26 minutes for the full default
budget (32,000 evaluation episodes plus 3,840 calibration episodes). To
rehearse the pipeline first, use `--episodes 5 --calib 12`, which takes a couple
of minutes and produces the same file structure.

### Step 3 — build the tables and figures

```bash
python scripts/make_analysis.py
```

Writes `tables/tables.tex`, `results/summary.json` and the figures

| File | Content |
|---|---|
| `fig_margin_bounds.pdf` | Post-fault margin against the smallest singular value of the reduced Jacobian, with the two-sided bounds of Proposition 1 drawn per plant (the upper bound `sqrt(m-1)` depends on `m`). |
| `fig_axioms.pdf` | `R_p` on a two-criterion slice; every `p > 0` stays positive when a criterion reaches zero. |
| `fig_lipschitz.pdf` | Distribution of the empirical Lipschitz ratio against the constant 1 of Theorem 2. |
| `fig_convergence.pdf` | Running mean with the Hoeffding band of Theorem 4(i) and the empirical Bernstein interval. |
| `fig_simplex.pdf` | A two-dimensional slice of the weight simplex with the rank-reversal boundary of Theorem 3(ii) and the kappa-ball. |
| `fig_failures.pdf` | Fraction of episodes with `R_0 = 0` per controller and fault class. |
| `fig_aggregators.pdf` | Median score under the excluded arithmetic mean against the score under the geometric mean. |

`summary.json` collects every headline number quoted in the text, including the
median *and* maximum paired dispersion `sigma_D` with the corresponding episode
budgets, and `margin_bounds_check`, which records the observed range of
`mu / sigma_min` per plant against `[1, sqrt(m-1)]`.

### Step 4 — check the deposit before archiving

```bash
python scripts/verify_deposit.py
```

Validates every `results/raw_<plant>.npz` — keys, shapes, dtypes, balance of the
eight fault classes, the range of the fault instant, the sharing of the fault
design across controllers — and then recomputes the 40 medians, the 40
outright-failure rates and the paired dispersion from the raw criteria and diffs
them against `results/summary.json`. Prints SHA-256 per file and exits non-zero
on any disagreement.

### Step 5 — audit the manuscript against the data

```bash
python scripts/verify_paper_claims.py
```

The complement of Step 4. Where `verify_deposit.py` asks whether the archive is
complete and coherent, this script asks whether the *text* agrees with the data:
it recomputes the seven rows of Table 2 (differences of medians, Cliff's delta,
Holm-adjusted p-values, `kappa*`), the captions of Figures 3–5, the sliding-mode
fragility of Section 8.4, the fault-class rates of Figure 6, the sensitivity
analysis of Section 8.5 and the campaign counts, and prints one PASS/FAIL line
per claim — 70 in all. Run it after any change to the weights, the reference
constants or the campaign.

### Step 6 — check the bibliography

```bash
python scripts/verify_references.py paper/Urrea_Mathematics_ResilienceFunctional_v8.tex
```

Queries Crossref for every `\bibitem` with a DOI and reports title/year
disagreements. Requires network access; nothing is rewritten automatically.

### Step 7 — rebuild the manuscript

```bash
cd paper && latexmk -pdf Urrea_Mathematics_ResilienceFunctional_v8.tex
```

The MDPI `Definitions/` folder from the official template must be present, and
the figures must be reachable at `figures/` relative to the manuscript.

### Typography

The figures are set in Palatino to match the body text of the MDPI class.
`make_analysis.py` looks for Palatino Linotype first (present on Windows), then
TeX Gyre Pagella (the metrically compatible clone shipped with TeX Live), and
falls back to any available serif face. Fonts are embedded as Type 42, so the
figures are editable text rather than outlines. No change is needed to
reproduce the published figures on a machine that has either face installed.

## 4. Changing the experiment

Everything that is not a theorem lives in two places.

* `rsp/simulate.py :: DEFAULT_CFG` — horizon, integration step, settling time,
  damping of the resolution and allocation, detector time constants, recovery
  window, noise levels, model uncertainty.
* `rsp/score.py :: W_NOMINAL, XSTAR_DEFAULT, X2_CAP, X4_BUDGET` — the nominal
  weight vector and the reference constants of Definition 2.

Changing the weights does **not** require re-running the campaign: the raw
criteria are stored, and `make_analysis.py` recomputes the scores, the rankings
and `kappa` from them. This is the practical content of Theorem 3 — the
sensitivity of a ranking to the weights is a post-processing question, not a
simulation question.

## 5. Determinism

Every random draw derives from `numpy.random.default_rng` seeded with the pair
`(seed, stream_id)`, where the stream identifiers separate the episode design,
the controller-internal randomness, the measurement and disturbance noise, and
the communication events. Re-running with the same `--seed` reproduces every
figure bit-for-bit on the same NumPy version. The seed used for the published
results is `20260811`, and the environment is recorded in the Figshare deposit.

## 6. Data

`results/raw_*.npz` is not tracked here: it is 40 arrays of 800 episodes and
belongs in the archive. Download it from the Figshare record above and drop it
into `results/` to run Step 3 without re-running Step 2.

Each `raw_<plant>.npz` contains, per controller `c`:

| Key | Shape | Meaning |
|---|---|---|
| `X_<c>` | (800, 6) | the raw criteria `x_1 … x_6` of Section 3.2, before normalization |
| `declared_<c>` | (800,) | whether the supervisor declared a fault |
| `degraded_<c>` | (800,) | whether the tracking error left its pre-fault envelope |
| `alive_<c>` | (800,) | whether the episode completed without divergence |
| `ratio_<c>` | (800,) | post/pre-fault error ratio, kept only as a diagnostic (Remark 1) |
| `fid`, `t_f`, `idx` | (800,) | fault class, fault instant and faulty actuator, shared by all controllers (common random numbers) |

## 7. Citing

Please cite both the article and the archived record; `CITATION.cff` carries the
machine-readable form.

## 8. License

MIT — see `LICENSE`.
