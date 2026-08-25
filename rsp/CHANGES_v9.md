# Corrections applied to Urrea_Mathematics_ResilienceFunctional_v8.tex

Eight edits, each verified against a rerun of the released code. Nothing else in
the manuscript was touched.

| # | Where | Was | Is now | Why |
|---|---|---|---|---|
| 1 | Appendix B | energy conserved "better than 3e-3 over 2 s"; RNEA vs `M qdd` "below 1e-13" | per-plant drift 4.1e-2 (P1), 9.8e-3 (P3), 6.0e-3 (P5), 2.2e-3 (P2), check fails above 5e-2; RNEA below 1e-15 | The 3e-3 claim is false for P1 and P3. Rerunning `verify_dynamics.py` gives the figures now quoted. The RNEA bound was true but two orders looser than the observed 3e-16. |
| 2 | Section 6, after Prop. 2 | "kappa* = 0.4 means the weights would have to be moved by 40% of their total mass, and kappa* >= 2 is unreachable" | kappa* = 0.4 means a mass of 0.2 changes hands; the range is bounded by 2(1 - min_k w_k) = 1.8 for the nominal weights | An l1 distance counts the mass that leaves *and* the mass that arrives, so 0.4 corresponds to transferring 0.2. The 2 is also loose: with w interior the supremum is 2(1 - min w). |
| 3 | Section 8.3 | "41% and 29% of the total weight mass"; "only 4.9%" | l1 perturbations of 0.412 and 0.287, i.e. transfers of 21% and 14%; ANN–CTC reverses at kappa* = 0.049, under 2.5% | Same factor of two as #2. |
| 4 | Figure 1 caption | "Every sample lies inside the cone" | states that the margin is computed exactly by facet enumeration, that the upper bound sqrt(m-1) is drawn per plant, and gives the observed ratio ranges [1.01,1.21] on P1, [1.01,1.79] on P3 and [1.00,1.66] on P5, as recorded in `summary.json :: margin_bounds_check` | The figure previously estimated mu by sampling directions on the sphere, which can only overestimate a minimum, and drew a single sqrt(6) line for all three plants. Against its own bound P5 showed a ratio of 2.418 > sqrt(5). With exact enumeration no sample violates its own cone. |
| 5 | Section 8.1 | reference constants listed only in the text | adds that all of them, including the calibrated x4*, are in `results/meta.json` in the form applied | The archived `meta.json` previously stored x2* = 0.10 and the raw median power, while the analysis silently used 0.30 and three times the median. Both the code and the file now carry the constants as used. |
| 6 | Section 8.1 | "injected at instants and severities drawn by Latin hypercube sampling" | adds the sampling range [0.80, 1.40] s for the fault instant | The range is a design choice a replicator needs and was only in the code. |
| 7 | Section 8.1 | "runs in 26 minutes on a single core" | adds "from the seed 20260811" | The seed appeared only in the repository README. |
| 8 | Data availability | generic | names release v1.0.0, the MIT license, what the archive contains, and that kappa* can be recomputed under any weights without re-running the campaign | Matches what is actually deposited. |

| 9 | Table 5, Cliff's delta column | +0.071, +0.048, +0.281, +0.055, -0.036, +0.191, +0.454 | +0.381, +0.373, +0.501, -0.134, +0.209, +0.418, +0.454 | Six of the seven published values are not reproduced by `rsp/stats.py`, nor by a paired sign statistic (0 of 7), nor by any variant tried: the column was stale. The regenerated values are the classical ordinal dominance statistic of Cliff (1993), which is what the text and the references describe. Only SMC-FT1 was already correct. |
| 10 | Section 8.4, the sign-disagreement paragraph | discussed CTC against PID | discusses ANN against CTC | With the corrected column, CTC-PID no longer shows the phenomenon (median +0.024, delta +0.209, both positive). ANN-CTC does: median +0.060 against delta -0.134. The paragraph's explanation applies verbatim, and gains force, because ANN-CTC is also the pair with the smallest `kappa*`. |
| 11 | Section 8.5 | SMC median post-fault error 0.15 mm on P3 | 0.145 mm | The value is 0.1453 mm over the 800 episodes, which sits on the two-decimal rounding boundary. Three significant figures removes it. |

## Figure to regenerate

`tables/tables.tex` must be regenerated as well, since it carries the corrected
effect sizes for all 28 pairs and the paper points readers to it.

`fig_margin_bounds.pdf` must be rebuilt (`python scripts/make_analysis.py`)
before submission: the exact-margin computation and the per-plant bounds change
what it shows. The other six figures are unaffected.

## Audit trail

`scripts/verify_paper_claims.py` recomputes all 70 numerical assertions of the
manuscript from the raw episode files and reports one PASS/FAIL line each. After
the eleven corrections above it reports zero disagreements. The scratch script
used to diagnose the stale effect-size column has been removed: it hard-coded
the superseded values and would only confuse a reader of the repository.

## Numbers checked and left unchanged

Medians and IQRs of Table 4; outright-failure rates; kappa = kappa* = 0.049 for
ANN–CTC on P3; maximum Lipschitz ratio 0.665; sigma_D = 0.288 -> N = 343;
N = 738 from Hoeffding; empirical Bernstein half-width 0.053 at N = 800;
Spearman range 0.969–1.000 and pooled 0.988 / 0.983; SMC failing on 89% of F7
against 1% of F6; CTC's m3 = 0.152 on P3; the 5th->8th and 7th->4th moves of
CTC; F4 at 0.78 and F2 at 0.67; 3,840 calibration and 32,000 evaluation
episodes; 26 minutes of wall-clock time. The bibliography has 107 entries, all
cited, none orphaned, ordered by first appearance as MDPI requires.
