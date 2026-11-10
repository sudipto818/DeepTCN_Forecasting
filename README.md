# DeepTCN Forecasting: Replication and Extensions

Replication of **"Probabilistic Forecasting with Temporal Convolutional Neural Network"**
(Chen, Kang, Chen & Wang, *Neurocomputing* 399, 2020) on the traffic and electricity
benchmarks, plus four extensions: calibrated intervals, count distributions, a learned
memory length and cross-series attention.

DA3041 (Applied Time Series Analysis) project, Group 2, Mehta Family School of Data Science
and AI, IIT Guwahati: Sudipto Ghosh, Kushagra Singhal, Sujal Patnaik, Pratham Saluja.

---

## Contents

1. [The paper](#1-the-paper)
2. [Repository layout](#2-repository-layout)
3. [Setup and data](#3-setup-and-data)
4. [Experimental protocol](#4-experimental-protocol)
5. [Replication results](#5-replication-results)
6. [Extensions and results](#6-extensions-and-results)
7. [Why our numbers differ from the paper](#7-why-our-numbers-differ-from-the-paper)
8. [Bugs found, fixes and known caveats](#8-bugs-found-fixes-and-known-caveats)
9. [How to run](#9-how-to-run)
10. [Conclusions and limitations](#10-conclusions-and-limitations)
11. [References](#11-references)

---

## 1. The paper

**Problem.** Forecast many related series at once (product demand, electricity customers,
traffic lanes) and give a full predictive distribution, not just a point. One shared
network learns patterns that hold across series, so it can also forecast series with
little or no history (cold start), which ARIMA-style per-series models cannot do.

**Objective.** Model the whole horizon at once:
`P(y_{t+1:t+Ω} | y_{1:t}, X)`. DeepTCN is **non-autoregressive**: all horizon steps are
predicted in parallel, so errors do not accumulate step to step and training is faster.

**Architecture** (Fig. 1 of the paper), a "neural dynamic regression" (like ARIMAX):

| Part | What it does |
|---|---|
| Encoder | Stacked residual blocks of **dilated causal convolutions**, `s(t) = Σ_k w(k) x(t − d·k)`. Doubling the dilation doubles how far back it sees; step `t` never sees the future. Output: a summary `h_t` of the history. |
| Decoder (resnet-v) | Residual block on the known future covariates: `δ_{t+j} = R(X_{t+j}) + h_t`, with `R = Dense → BN → ReLU → Dense → BN`, then ReLU. The same `h_t` serves every horizon `j`. |
| Covariates | Embeddings of the series id, hour of day, day of week and month. |
| Output head | **Non-parametric:** quantiles trained with pinball loss `L_q = q(y − ŷ)⁺ + (1 − q)(ŷ − y)⁺`. **Parametric:** e.g. Gaussian (μ, σ) trained by likelihood. |

**Paper results we compare against** (rolling 24 h ahead forecasts over the last 7 days):

| Table 4: QL50 / QL90 | electricity | traffic |
|---|---|---|
| ETS | 0.100 / 0.050 | 0.360 / 0.325 |
| auto.arima | 0.142 / 0.054 | 0.246 / 0.140 |
| DeepAR-t | 0.068 / 0.033 | 0.117 / 0.090 |
| SQF-RNN | 0.066 / 0.035 | 0.119 / 0.090 |
| DeepState | 0.043 / 0.025 | 0.084 / 0.058 |
| **DeepTCN-Quantile** | **0.057 / 0.029** | **0.058 / 0.040** |
| DeepTCN-Gaussian | 0.062 / 0.039 | 0.071 / 0.049 |

| Table 5: NRMSE / SMAPE / MASE | electricity | traffic |
|---|---|---|
| ETS | 0.838 / 0.156 / 1.247 | 0.872 / 0.594 / 1.881 |
| DeepAR-t | 0.550 / 0.110 / 0.931 | 0.396 / 0.104 / 0.442 |
| SQF-RNN | 0.518 / 0.113 / 0.937 | 0.381 / 0.117 / 0.449 |
| **DeepTCN-Quantile** | **0.523 / 0.118 / 0.926** | **0.364 / 0.110 / 0.438** |

**Sensitivity (paper Fig. 4).** 5 encoder layers see 63 h, 6 see 127 h, 7 see the full
168 h week. 6 and 7 layers train to a lower loss than 5 and differ little from each other:
once the receptive field covers enough history, depth matters less.

---

## 2. Repository layout

```
config.py                 paths, window sizes, splits, per-dataset settings
configs/                  YAML for the adaptive (synthetic) and traffic ablation runs
src/
  data_traffic.py         PEMS-SF parser, randperm direction check, 10-min -> hourly
  data_electricity.py     UCI LD2011_2014, 15-min -> hourly, keep 2012-2014
  data_synthetic.py       synthetic panel with rising noise and a shifted last week
  data_panel.py           synthetic panel with known memory length (Ext. 3)
  windows.py              origins, train/val/calib/test splits, calendar, batching
  model.py                DeepTCN (encoder, resnet-v decoder, embeddings)
  losses.py               pinball loss
  distributions.py        Gaussian / negative binomial / ZIP heads (Ext. 2)
  train.py, predict.py    training with early stopping and resume; prediction arrays
  baseline_naive.py       seasonal naive with empirical quantiles
  metrics.py              QL, SMAPE, NRMSE, MASE, coverage, width, Winkler, crossing
  cqr.py, aci.py          conformal calibration (Ext. 1)
  adaptive.py             learned memory length gate (Ext. 3)
  xseries.py              XTCN: cross-series attention + per-lane memory (Ext. 3, 4)
scripts/
  01_prepare_data.py      build data/processed/<name>.npy, check splits
  02_train.py             train a model           [--dist negbin|zip|gaussian] [--resume]
  03_predict.py           write prediction arrays (daily or hourly protocol)
  04_diagnostic.py        raw interval coverage by horizon and series size
  05_calibrate.py         CQR variants + ACI, headline/horizon/day/alpha CSVs
  06_paper_metrics.py     paper-style Table 4/5 numbers next to the paper's
  07_adaptive.py          Ext. 3 on the synthetic memory panel
  08_traffic_ext.py       Ext. 3/4 on full traffic (YAML configs)
  09_traffic_subset.py    Ext. 2 and Ext. 3/4 on the 320-lane traffic subset
R/plots.R, R/tables.R     figures and LaTeX tables from the result CSVs
tests/                    unit tests (python tests/run_all.py)
results/                  all result CSVs + per-dataset run logs (*_RESULTS.md)
data/raw, data/processed, arrays/   inputs, processed arrays, checkpoints (not in repo)
```

---

## 3. Setup and data

```bash
pip install -r requirements.txt     # numpy pandas scipy torch pyyaml (+ matplotlib for figures)
# R figures/tables need R with ggplot2
```

| Dataset | Source | Put in | After processing |
|---|---|---|---|
| traffic | UCI PEMS-SF (`PEMS_train`, `PEMS_test`, `randperm`) | `data/raw/PEMS-SF/` | 963 lanes × 10,560 h (440 days); hourly value = sum of six 10-min occupancies (0–6) |
| electricity | UCI `LD2011_2014.txt` | `data/raw/` | 370 customers × 26,304 h (2012–2014), 15-min kW summed to hours |
| traffic_sub | made by `09_traffic_subset.py` | – | 320 lanes (40 from each of 8 occupancy bands), all 440 days |
| synthetic | generated | – | 40 series × 2,400 h, for testing the pipeline |

The `randperm` direction in PEMS-SF is ambiguous; `data_traffic.py` picks the direction
that gives higher day-to-day continuity (file order 0.867, forward 0.917, backward 0.878,
so **forward**).

---

## 4. Experimental protocol

- **Windows:** 168 h of history in, 24 h out; quantiles 0.1, 0.5, 0.9 (80% interval).
- **Splits, counted back from the end:** test = last 7 days, calibration = the 7 days
  before that, validation = the 14 days before that, training = everything earlier.
  A check makes sure no target hour falls in two splits.
- **Daily protocol (the paper's):** one forecast per day, 7 test origins.
  **Hourly protocol (ours):** one forecast per hour, 145 test origins. It gives far more
  samples for calibration.
- One model, trained once before the first test window, never retrained.
- **Epoch selection** on the validation block, never on the test week.

**Metrics.**
`QL_q = Σ L_q(y, ŷ_q) / Σ|y|` (as in the paper, no factor 2);
`SMAPE = mean 2|y − ŷ| / (|y| + |ŷ|)`; `NRMSE = RMSE / mean|y|`;
`MASE = MAE / MAE of seasonal naive (m = 24)` on the in-sample part.
Intervals: **coverage** (target 0.80), **width** (mean width / mean|y|) and
**Winkler** score (width plus `2/α` times the miss, lower is better).

**Model settings (`config.py`):**

| | traffic | electricity |
|---|---|---|
| dilations | 1, 2, 4, 8, 16, 32 | 1, 2, 4, 8, 16, 32 |
| channels / batch / lr | 16 / 32 / 5e-3 | 32 / 64 / 5e-3 |
| target transform | none | log1p |
| early stopping | patience 5 | patience 5, run capped at 25 epochs |

---

## 5. Replication results

All results are from a single seed (seed 1). Daily protocol, raw model output.

| | QL50 | QL90 | NRMSE | SMAPE | MASE |
|---|---|---|---|---|---|
| traffic, ours | 0.068 | 0.050 | 0.408 | 0.135 | 0.522 |
| traffic, paper | 0.058 | 0.040 | 0.364 | 0.110 | 0.438 |
| electricity, ours | 0.108 | 0.064 | 1.293 | 0.239 | 2.478 |
| electricity, paper | 0.057 | 0.029 | 0.523 | 0.118 | 0.926 |

(`results/paper_metrics_*.csv`; the hourly protocol gives traffic 0.066 / 0.047 / 0.376 /
0.129 / 0.496.)

- **Traffic:** 12–25% behind the paper, but still ahead of the paper's DeepAR-t, SQF-RNN
  and DeepState on QL50/QL90. Best validation loss at epoch 31, early stop at 41.
- **Electricity:** about 2× worse; MASE above 1 means worse than repeating yesterday.
  This run was hit by the scale-floor bug (Section 8): one target became 4.3 × 10⁶, the
  validation loss was flat at about 312.6, so early stopping could not pick a good epoch.
- No quantile crossings in either model (crossing rate 0.0).

Full run logs: `results/TRAFFIC_RESULTS.md`, `results/ELECTRICITY_RESULTS.md` (the
electricity log predates the diagnosis of the scale bug and attributes the gap to the
epoch cap).

---

## 6. Extensions and results

| # | Extension | Question | Evaluated on |
|---|---|---|---|
| 1 | Calibrated intervals | Do the 80% intervals really cover 80%? | traffic, electricity (full data), synthetic |
| 2 | Count heads (NB, ZIP) | Do count distributions beat the paper's choices? | 320-lane traffic subset |
| 3 | Learned memory length | Does every series need a 168 h window? | 320-lane traffic subset (+ synthetic panel code) |
| 4 | Cross-series attention | Can lanes borrow from correlated lanes? | 320-lane traffic subset |

### Extension 1: Calibrated intervals (`cqr.py`, `aci.py`, `05_calibrate.py`)

**Motivation.** The raw 80% intervals cover 0.89 on traffic (too wide) and 0.73–0.78 on
electricity (too narrow).

**Methods** (run on saved predictions, no retraining):
- **CQR** (Romano et al., 2019): score the unseen calibration week with
  `E = max(q̂0.1 − y, y − q̂0.9)` and widen or shrink both ends by its conformal 80% quantile.
  Three variants: *pooled* (one offset), *horizon* (one offset per horizon step, on scores
  divided by the series scale), *two-tail* (a separate offset for each tail at α/2).
- **ACI** (Gibbs & Candès, 2021): adapt the level online, `α_{t+1} = α_t + γ(α − err_t)`,
  one level per horizon. Only errors that are already observable are used (delay `w` for
  horizon `w` in the hourly protocol). γ is chosen on the validation split by Winkler score.

**Raw coverage diagnostic (daily, `04_diagnostic.py`):**

| | coverage | width | by horizon |
|---|---|---|---|
| traffic | 0.888 | 0.455 | 0.94 at h1 down to 0.84 at h24, always above 0.80 |
| electricity | 0.731 | 0.746 | 0.84–0.87 at h1–4 down to 0.66 at h20 |

Coverage is about the same in every series-size quartile. On electricity the undercoverage
is a model problem, not a scaling problem.

**Hourly protocol (145 origins), headline:**

| | traffic cov / width / Winkler | electricity cov / width / Winkler |
|---|---|---|
| raw | 0.894 / 0.458 / 0.730 | 0.783 / 0.916 / 1.203 |
| CQR pooled | 0.801 / 0.395 / 0.713 | 0.736 / 0.912 / 1.204 |
| CQR horizon | 0.801 / 0.390 / 0.712 | 0.737 / 0.880 / 1.200 |
| CQR two-tail | 0.801 / 0.390 / 0.711 | 0.731 / 0.870 / 1.194 |
| ACI | 0.804 / 0.392 / 0.712 | 0.790 / 0.887 / 1.172* |

**Daily protocol (7 origins):** traffic CQR two-tail 0.799 / 0.382 / 0.746 (raw 0.888 /
0.455 / 0.770); electricity CQR makes coverage worse (0.688–0.711 vs raw 0.731), and ACI
gives 0.743.

**ACI guarantee.** The coverage bound is 1.286 for the daily protocol (above 1, so no
guarantee with 7 origins), and 0.441 (traffic) and 0.276 (electricity) for the hourly
protocol (a real guarantee).

**Takeaways.** On stable traffic, CQR fixes coverage with about 15% narrower intervals.
On electricity, whose test week (the last week of December 2014, Christmas) does not look
like the calibration week, CQR undercovers and ACI gets closest to 80%.

\* **Caveat:** 8.4% of ACI's electricity hourly intervals are infinite (`inf_share` =
0.084): ACI's level went past what the calibration scores support and it returned the whole
line. Width and Winkler drop those points while coverage counts them as covered. On finite
intervals only, ACI's coverage is roughly 0.77. `metrics.coverage_finite` now reports this
directly (Section 8). On traffic, ACI produced no infinite intervals.

### Extension 2: Count distributions (`distributions.py`)

**Motivation.** The series are non-negative and right-skewed; a Gaussian puts mass below 0.
The paper suggests the negative binomial for long-tailed data but never tries it.

**Method.** New output heads, `--dist negbin | zip`:
- **Negative binomial:** mean μ, `Var = μ + αμ²`.
- **Zero-inflated Poisson:** Poisson(λ) with an extra point mass π at zero.
- Trained by likelihood (in float64), with DeepAR-style scaling by `ν = 1 + mean|history|`,
  then turned back into the same 3 quantiles, so calibration and metrics treat every head
  alike. A Gaussian head is also implemented but was not evaluated.

**Catch.** The data are not counts: we fit on `round(y × count_unit)` (100 for traffic,
1 for electricity). For ZIP, Poisson has `Var = mean`, so the unit itself fixes the spread.

**Results** (main DeepTCN, 320 lanes, 30 epochs × 8,000 windows, early stopping, daily
protocol, seed 1, `results/subset_ext2.csv`):

| Output head | QL50 | QL90 | NRMSE | SMAPE | MASE | time |
|---|---|---|---|---|---|---|
| **Quantile (baseline)** | **0.076** | **0.058** | 0.455 | **0.150** | **0.572** | 4.1 min |
| Negative binomial | 0.088 | 0.067 | 0.479 | 0.164 | 0.671 | 3.0 min |
| Zero-inflated Poisson | 0.081 | 0.063 | **0.446** | 0.163 | 0.614 | 3.9 min |

The quantile head wins 4 of 5 metrics, as in the paper. ZIP is second; its 2% NRMSE edge is
noise, and traffic has almost no zeros. NB stopped early and probably needs a lower learning
rate.

### Extension 3: Learned memory length / adaptive receptive field (`adaptive.py`, `xseries.py`)

**Idea.** Every series gets the same 168 h window, but a new product with 3 days of data
does not need a week. A soft mask `m_i(l) = σ((τ_i − l)/s)` (l = 0 is the newest step)
hides old steps; the loss adds `λ · mean(τ_i / L)`. The encoder is unchanged.

Three details that make it work:
1. The **mask is fed as its own input channel**, so a hidden value is not mistaken for a
   real zero.
2. **It starts fully open:** `τ_max = L + 4s₀` and the mask-channel weights start at 0, so
   training starts from plain DeepTCN.
3. **Warm-up and annealing** of the temperature s from L/10 down to 1, otherwise τ shuts
   before the encoder has learned to use old history.

Gate modes in `adaptive.py`: `none`, `fixed`, `global` (one τ) and `series` (τ from the
number of observed steps and the zero ratio). A synthetic panel with known memory lengths
(A = 7, B = 24, C = 2, D = 3 with padded history) and its check (C < A < B, D smallest)
is in `07_adaptive.py`; its results are not stored in `results/`. On traffic (`xseries.py`)
τ_i comes from each lane's embedding.

### Extension 4: Cross-series attention (`xseries.py`)

**Motivation.** Neighbouring detectors see the same congestion, yet DeepTCN encodes each
lane alone.

**Method.** Attention over lane embeddings with the diagonal masked out:
`A = softmax(E Eᵀ / √d)`, then `h ← h + α · (A h)`, with one scalar α starting at 0 (so the
model starts as the baseline). One training step is one origin with all lanes, so the
attention sees the whole panel. A coherence test asks whether attention points at lanes that
really move together, comparing against embeddings shuffled across lanes (200 shuffles).

### Extensions 3 and 4: Results (`results/subset_ext34.csv`)

Compact DeepTCN (XTCN), 320 lanes, 900 steps (one origin × all lanes per step), daily
protocol, seed 1.

| Row | QL50 | QL90 | NRMSE | SMAPE | MASE | time |
|---|---|---|---|---|---|---|
| baseline | 0.129 | 0.085 | 0.564 | 0.292 | 1.023 | 2.8 min |
| + attention | 0.130 | 0.084 | 0.564 | 0.302 | 1.026 | 2.6 min |
| + memory | 0.130 | 0.084 | 0.554 | 0.284 | 1.036 | 2.7 min |
| + both | 0.130 | 0.083 | 0.558 | 0.282 | 1.036 | 2.7 min |

- No row beats the baseline beyond noise (about 3%).
- **Memory:** τ stayed fully open (median 230 h, range 228–232 h), the expected result for
  lanes with full history.
- **Attention:** α = −0.29; the lanes it attends to co-move more than chance
  (coherence 0.190 vs 0.185 ± 0.001 for shuffled embeddings, p < 0.005), but there is no
  accuracy gain.
- MASE ≈ 1 means these runs are undertrained, so compare the rows only with each other, not
  with Section 5.

Full-traffic, 3-seed configs for these ablations are in `configs/traffic_*.yaml`
(`08_traffic_ext.py`); those runs are not part of the reported results.

---

## 7. Why our numbers differ from the paper

| | Paper (App. C, Table 7) | Ours |
|---|---|---|
| Epoch chosen on | the test set | a separate validation block |
| Encoder dilations | 1, 2, 4, 8, 16, 20, 32 | 1, 2, 4, 8, 16, 32 |
| Batch / learning rate | 128 / 1e-2 (traffic), 512 / 5e-2 (electricity) | 32 and 64 / 5e-3 |
| Electricity training data | Decembers 2011–2013 only | all of 2012–2014 |
| Training length | not stated (Fig. 4 runs 200 epochs) | early stopping (traffic 41, electricity capped at 25) |

**Main reason:** picking the epoch on the test set makes the paper's numbers optimistic; we
never look at the test week before the end. Other differences: one seed, no hyperparameter
search, and the electricity scale bug below.

---

## 8. Bugs found, fixes and known caveats

**Found during the project (affect the reported numbers):**
- **MASE denominator** was wrong at first; fixing it moved traffic MASE from 0.13 to 0.52.
  The reported results use the fixed version.
- **Electricity scale floor:** each window is divided by its mean, floored at 1e-6. A
  customer with an all-zero history (late start) then gets a scale of 1e-6, which turned
  one target into 4.3 × 10⁶ on the log1p scale. That dominated the validation loss (flat at
  about 312.6) and blinded early stopping. **The reported electricity results come from this
  run.**

**Code changes made after the reported runs (results were not regenerated):**
- `src/train.py`, `src/predict.py`: `window_scale()` now gives a scale of 1 to all-zero
  history windows instead of 1e-6. Windows with any real signal are scaled exactly as
  before, so existing traffic checkpoints and results are unaffected. The parametric heads
  already used `1 + mean`.
- `src/metrics.py`: new `coverage_finite` in the headline output: coverage over finite
  intervals only, consistent with width and Winkler (see the ACI caveat in Section 6).
- `tests/run_all.py` now runs all 9 test files; `tests/test_xseries.py` gained a
  `__main__` block.

**Known minor caveats:**
- `windows.calendar` assumes the days are consecutive and starts on a Monday. PEMS-SF leaves
  out public holidays, so the day-of-week covariate slips after each gap; "month" is
  approximated as 30-day blocks.
- The main `DeepTCN` merges the encoder summary with the decoder output by concatenation
  followed by a dense layer; `XTCN` uses the paper's `R(X) + h`. That is one reason the
  Ext. 3/4 baseline differs from the main model.

---

## 9. How to run

Run from the repo root (on Windows use `python`).

```bash
# 1. data
python3 scripts/01_prepare_data.py traffic          # or electricity / synthetic

# 2. train, 3. predict (daily = paper protocol, hourly = ours)
python3 scripts/02_train.py traffic 1                # [--dist negbin|zip|gaussian] [--resume]
python3 scripts/03_predict.py traffic deeptcn daily 1
python3 scripts/03_predict.py traffic deeptcn hourly 1
python3 scripts/03_predict.py synthetic naive daily  # seasonal-naive baseline

# 4. raw interval diagnostic, 5. calibration (CQR + ACI)
python3 scripts/04_diagnostic.py traffic seed1 daily
python3 scripts/05_calibrate.py traffic seed1 hourly

# 6. paper-style table
python3 scripts/06_paper_metrics.py traffic daily

# extensions
python3 scripts/09_traffic_subset.py ext2            # quantile / NB / ZIP heads (~11 min, 8-core CPU)
python3 scripts/09_traffic_subset.py ext34           # baseline / attn / tau / both (~11 min)
python3 scripts/09_traffic_subset.py table
python3 scripts/07_adaptive.py configs/adaptive_synthetic.yaml
python3 scripts/08_traffic_ext.py configs/traffic_baseline.yaml configs/traffic_attn.yaml

# figures and LaTeX tables
Rscript R/plots.R traffic seed1 hourly
Rscript R/tables.R traffic seed1 hourly

# tests
python3 tests/run_all.py
```

The tests cover the metrics against hand-computed values, split overlap and leakage,
causality of every encoder variant (checked with autograd), the CQR order statistic and its
coverage on exchangeable data, ACI look-ahead and its bound on adversarial sequences, the
NLL and quantiles of each head against scipy, recovery of the parameters by maximum
likelihood, and that every extension starts out identical to the baseline.

---

## 10. Conclusions and limitations

**Findings**
- DeepTCN reproduces on traffic within 12–25% of the paper, with honest epoch selection.
- Checking the evaluation mattered: the MASE and scale-floor bugs changed the picture more
  than any model change.
- Good accuracy is not the same as good intervals: CQR fixes coverage on stable data, and
  ACI handles drift (with the infinite-interval caveat).
- Count heads, learned memory and attention gave no gain on traffic lanes with full history.
  The memory gate behaves as designed (it stays open when the history is real), and the
  attention points at lanes that really co-move.
- **Significance:** DeepTCN's intervals can be made trustworthy after training, for any
  output head, without retraining.

**Limitations:** one seed and no hyperparameter search; electricity was trained before the
scale fix; the extension runs use a 320-lane subset with short training.

**Next steps:** retrain electricity with the fixed scale; longer, multi-seed runs for NB,
ZIP, memory and attention; test the memory gate on cold-start splits.

---

## 11. References

- Chen, Y., Kang, Y., Chen, Y., Wang, Z. (2020). Probabilistic forecasting with temporal
  convolutional neural network. *Neurocomputing*, 399, 491–501.
- Romano, Y., Patterson, E., Candès, E. (2019). Conformalized quantile regression. *NeurIPS*.
- Gibbs, I., Candès, E. (2021). Adaptive conformal inference under distribution shift. *NeurIPS*.
- Salinas, D., Flunkert, V., Gasthaus, J., Januschowski, T. (2020). DeepAR: probabilistic
  forecasting with autoregressive recurrent networks. *International Journal of Forecasting*.
- Data: UCI ElectricityLoadDiagrams20112014; UCI PEMS-SF.
