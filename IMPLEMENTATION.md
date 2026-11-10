# Implementation Guide: DeepTCN and Our Four Extensions

This document explains **how each method is implemented** in this repository, both the
original DeepTCN of Chen et al. (2020) and our four extensions, and **how to run every
experiment** end to end. It is written so that someone with the raw data and this
repository can rebuild every number in `results/` and in the report.

---

## Contents

1. [Big picture](#1-big-picture)
2. [Environment setup](#2-environment-setup)
3. [Data preparation](#3-data-preparation)
4. [Forecasting windows and splits](#4-forecasting-windows-and-splits)
5. [The paper's method: DeepTCN](#5-the-papers-method-deeptcn)
6. [Training and prediction](#6-training-and-prediction)
7. [Evaluation metrics](#7-evaluation-metrics)
8. [Extension 1: conformal calibration (CQR and ACI)](#8-extension-1-conformal-calibration-cqr-and-aci)
9. [Extension 2: count likelihood heads (NB and ZIP)](#9-extension-2-count-likelihood-heads-nb-and-zip)
10. [Extension 3: learned memory length](#10-extension-3-learned-memory-length)
11. [Extension 4: cross-series attention](#11-extension-4-cross-series-attention)
12. [Running everything: step-by-step commands](#12-running-everything-step-by-step-commands)
13. [File formats and naming conventions](#13-file-formats-and-naming-conventions)
14. [Tests](#14-tests)
15. [Configuration reference](#15-configuration-reference)
16. [Troubleshooting](#16-troubleshooting)

---

## 1. Big picture

```
 raw data ──01──► data/processed/<name>.npy          (n_series × n_hours)
                        │
                        ├──02──► arrays/<name>_<tag>.pt            trained model
                        │
                        ├──03──► arrays/preds_*.npy, actuals_*.npy  forecasts (val / calib / test)
                        │                │
                        │                ├──04──► results/diagnostic_*.csv      raw coverage check
                        │                ├──05──► results/headline_*.csv ...    CQR + ACI  (Ext. 1)
                        │                └──06──► results/paper_metrics_*.csv   paper Tables 4/5
                        │
                        ├──07──► results/adaptive_synthetic_*      Ext. 3 on a synthetic panel
                        ├──08──► results/xseries_*.csv             Ext. 3/4 on full traffic
                        └──09──► results/subset_ext2.csv, subset_ext34.csv   Ext. 2-4 on 320 lanes
```

Two design principles run through the code:

- **Training is separated from evaluation.** Models write prediction arrays once.
  Diagnostics, calibration and paper metrics read only those arrays, so calibration
  (Ext. 1) never touches or retrains a network.
- **Every head produces the same three quantiles** (0.1, 0.5, 0.9). The parametric heads
  (Gaussian, NB, ZIP) convert their fitted distribution into these quantiles, so every
  downstream script treats all heads identically.

| Directory | Contents |
|---|---|
| `config.py` | paths, window sizes, split sizes, quantiles, per-dataset hyperparameters |
| `src/` | library code (data, model, training, metrics, calibration, extensions) |
| `scripts/` | numbered entry points `01`–`09` |
| `configs/` | YAML configs for Ext. 3/4 runs |
| `tests/` | unit tests |
| `R/` | figures and LaTeX tables from the result CSVs |
| `results/` | all result CSVs and run logs |

---

## 2. Environment setup

Requirements: Python 3.9+ with `numpy`, `pandas`, `scipy`, `torch`, `pyyaml`
(`requirements.txt`), plus `matplotlib` for the Ext. 3/4 figures. The R figures need R
with `ggplot2`.

```bash
git clone https://github.com/sudipto818/DeepTCN_Forecasting
cd DeepTCN_Forecasting
pip install -r requirements.txt
pip install matplotlib            # optional, for figures in scripts 07/08
```

- A GPU is used automatically when `torch.cuda.is_available()`; everything also runs on CPU.
- Run every command **from the repository root**. Each script adds `.` and `src/` to
  `sys.path` itself.
- On Windows use `python` instead of `python3`. If `Rscript` is not on the PATH, call it by
  its full path, e.g. `"C:\Program Files\R\R-4.5.1\bin\Rscript.exe" R/plots.R ...`.

---

## 3. Data preparation

### 3.1 Download

| Dataset | Source | Place the files in |
|---|---|---|
| Traffic (PEMS-SF) | UCI ML Repository, *PEMS-SF* (https://archive.ics.uci.edu/dataset/204/pems+sf) | `data/raw/PEMS-SF/` → `PEMS_train`, `PEMS_test`, `randperm` |
| Electricity | UCI ML Repository, *ElectricityLoadDiagrams20112014* (https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) | `data/raw/LD2011_2014.txt` |

### 3.2 Traffic (`src/data_traffic.py`)

1. **Parse.** Each line of `PEMS_train` / `PEMS_test` is one day written as
   `[v v ...; v v ...; ...]`: 963 sensor rows × 144 ten-minute readings.
   `parse_line()` strips the brackets, splits rows on `;` and returns a `(963, 144)` array.
   The days of both files are concatenated (440 days in total).
2. **Restore calendar order.** The days are shuffled in the release, and `randperm` gives
   their order, but its *direction* is ambiguous ("day i goes to position perm[i]" or
   "position i holds day perm[i]"). Getting it wrong silently scrambles every
   hour-of-day pattern, so `day_continuity()` decides from the data: it computes the
   correlation between the mean daily profiles of consecutive days for the file order, the
   forward permutation and the backward permutation, and keeps the higher one.
   Our run: file order 0.867, forward **0.917**, backward 0.878, so forward is used.
3. **Hourly aggregation.** Each day `(963, 144)` is reshaped to `(963, 24, 6)` and summed over
   the six readings, so values are sums of occupancies (range 0–6).
4. **Output:** `data/processed/traffic.npy`, shape `(963, 10560)`.

### 3.3 Electricity (`src/data_electricity.py`)

1. Read with `pd.read_csv(sep=";", decimal=",", index_col=0)`: 140,256 rows × 370 customers
   of 15-minute kW readings; NaNs become 0.
2. Sum every 4 rows to hours: 35,064 hours.
3. Keep the last 26,304 hours (2012-01-01 to 2014-12-31), as in the paper.
4. Transpose to `(370, 26304)` and save `data/processed/electricity.npy`.
   Many customers start late, so early histories contain long runs of zeros.

### 3.4 Synthetic data

- `src/data_synthetic.py`: 40 hourly series × 2,400 h with daily and weekly shape, noise that
  rises slowly over time (so a fitted model becomes over-confident, which CQR should fix)
  and an extra noise jump in the last 7 days (a shift the calibration week never saw, which
  ACI should handle). Used to test the pipeline and the calibration code.
- `src/data_panel.py`: synthetic windows with a **known memory length** for Ext. 3. There
  are four types: A (period-7 seasonal), B (period-24 seasonal), C (AR(1), φ = 0.3) and
  D (type A with only the last 3 history values real and the rest zero padding).

### 3.5 Run

```bash
python3 scripts/01_prepare_data.py traffic       # also: electricity, synthetic
```
It writes the `.npy` file, builds the splits, checks that they do not overlap and prints
the number of origins and the target-hour range of each split.

---

## 4. Forecasting windows and splits

Implemented in `src/windows.py`.

**A window** is a pair (series `i`, origin `t0`), where `t0` is the last hour the model may see:

```
input  = y[i, t0-167 : t0+1]      (IN_LEN  = 168 hours)
target = y[i, t0+1   : t0+25]     (OUT_LEN = 24 hours)
```

**Splits** (`split_origins(n_steps, protocol)`) are counted back in whole days from the end
of the series:

| Split | Length | Daily protocol origins | Hourly protocol origins |
|---|---|---|---|
| test | last 7 days | 7 (one per day) | 145 (one per hour) |
| calib | 7 days before test | 7 | 145 |
| val | 14 days before calib | 14 | 313 |
| train | every origin whose whole target ends before val | 9,697 (traffic), 25,441 (electricity) | same |

For a block starting at hour `s` with length `n`, the origins are `t − 1` for
`t = s, s+step, …` while `t + 24 ≤ s + n` (step 24 for daily, 1 for hourly), so a target
never crosses into the next block. `check_splits()` asserts that no target hour belongs to
two splits and that train < val < calib < test in time.

**Calendar covariates** (`calendar(n_steps)`): hour of day `t mod 24`, day of week
`(t // 24) mod 7`, month index `(t // 24 // 30) mod 12`.

**Batching** (`build_batch`): vectorised fancy indexing builds `(B, 168)` histories,
`(B, 24)` targets and their calendar slices with no Python loop over windows.

**Series scale** (`series_scale`): mean |y| of each series over the training part, floored
at 0.1 × the median scale. It is used only to normalise conformity scores in Ext. 1.

---

## 5. The paper's method: DeepTCN

Implemented in `src/model.py` (class `DeepTCN`) and `src/losses.py`.

### 5.1 Covariate embeddings

| Covariate | Embedding size |
|---|---|
| series id | 8 |
| hour of day (24) | 4 |
| day of week (7) | 3 |
| month (12) | 2 |

The concatenated covariate vector therefore has 17 dimensions.

### 5.2 Encoder: dilated causal convolutions

```
x_t = [ y_t , (extra channels) , covariates_t ]          shape (B, 1+n_extra+17, 168)
x   = Conv1d(1x1) → C channels
for d in dilations:                                       # e.g. 1,2,4,8,16,32
    out = pad_left(x, d) → Conv1d(k=2, dilation=d) → BN → ReLU
    out = pad_left(out, d) → Conv1d(k=2, dilation=d) → BN
    x   = ReLU(out + x)                                   # residual block (EncBlock)
h_t = x[:, :, -1]                                         # summary of the history
```

- **Causality** comes from left-only padding of `(k−1)·d` steps, so output `t` never sees
  inputs after `t`. The tests check this with autograd.
- **Receptive field** `r = 1 + 2·(k−1)·Σd = 1 + 2·63 = 127` hours for `d = 1…32`.

### 5.3 Decoder: resnet-v on future covariates

For every horizon `j = 1…24`:

```
R(X_{t+j}) = BN(Dense64( ReLU(BN(Dense64(cov_{t+j}))) ))
m_j        = Dropout0.2( ReLU( BN( Dense64( [h_t , R(X_{t+j})] ) ) ) )
output_j   = Dense(n_out)(m_j)        (+ final ReLU for the quantile head)
```

The same `h_t` is used for all 24 horizons, so the whole horizon is decoded **in
parallel** (non-autoregressive). The paper writes the merge as `R(X) + h_t`; the main model
concatenates and applies a dense layer, and the compact model of Ext. 3/4 (`XTCN`) uses the
residual sum exactly.

### 5.4 Output heads

- **Quantile head (DeepTCN-Quantile):** `n_out = 3` (q = 0.1, 0.5, 0.9) with a final ReLU,
  trained with the pinball loss (`losses.pinball_loss`):

  $$L_q(y,\hat y) = \max\big(q\,(y-\hat y),\;(q-1)(y-\hat y)\big),$$

  summed over the three quantiles and averaged over batch and horizon.
- **Parametric heads:** `n_out = 2` raw outputs and no final ReLU (`positive=False`); see
  Section 9.

### 5.5 Scaling (`train.window_scale`, `train.scale_window`)

Each window is divided by the mean |y| of its 168-hour history, and predictions are
multiplied back by the same number. A window whose history is entirely zero (a customer
that has not started yet) gets a scale of 1 instead of a tiny number, so its targets are not
blown up. For electricity, `log1p` is applied first (`log_target = True`), and `expm1`
undoes it after unscaling.

---

## 6. Training and prediction

### 6.1 Training loop (`src/train.py`, `train()`)

```
load y = data/processed/<name>.npy; build calendar and daily splits
model = DeepTCN(...); optimiser = Adam(lr)
for epoch in range(epochs):
    for b in range(n_per_epoch // batch_size):           # 20,000 windows per epoch by default
        sample batch_size random (series, train origin) pairs
        build batch, scale (quantile) or fit-scale (parametric)
        loss = pinball (quantile) or NLL (parametric); Adam step
        stop with an error if the loss is not finite
    val = mean loss over the val origins, all series at once
    if val improves: save arrays/<name>_<tag>.pt (best checkpoint), reset patience
    else: patience += 1, stop when it reaches `patience`
    save arrays/<name>_<tag>_resume.pt (model, optimiser, RNG states, counters)
         (written after every completed epoch that does not trigger early stopping)
```

- `--resume` restores the full state (including the NumPy and torch RNG states) from
  `<name>_<tag>_resume.pt`, so an interrupted run continues from the next epoch. It also
  allows raising `epochs` in `config.py` and continuing a finished run.
- `tag` is `seed1` for the quantile head and `negbin_seed1`, `zip_seed1` or
  `gaussian_seed1` for the parametric heads (`run_tag()`).
- The epoch is selected **only on the validation block**, never on the test week.

### 6.2 Prediction (`src/predict.py`, `run()`)

For each split (`val`, `calib`, `test`) and protocol (`daily` / `hourly`), all series are
forecast at each origin:

1. Build the batch exactly as in training.
2. Quantile head: multiply by the window scale, then `expm1` if `log_target`.
   Parametric head: convert the parameters to quantiles (Section 9.3), divide by
   `count_unit` for count heads, or `expm1` for log-scale Gaussian.
3. Save `preds_<name>_<split>_<protocol>_<tag>.npy` with shape
   `(n_origins, n_series, 24, 3)`, plus `actuals_...npy` with shape `(n_origins, n_series, 24)`.
4. Save `arrays/<name>_scale.npy` (series scale for conformal scores).

### 6.3 Seasonal-naive baseline (`src/baseline_naive.py`)

Point forecast `ŷ_{t0+w} = y_{t0+w−24}`. The quantiles add empirical quantiles of the
scale-normalised training residuals, per horizon, then are clipped at 0. It needs no
training and was used to develop and test the calibration code (`tag = naive`).

---

## 7. Evaluation metrics

Implemented in `src/metrics.py`. Arrays: `y (origins, series, 24)`,
`preds (origins, series, 24, 3)`, interval `[lo, hi]`.

| Metric | Formula | Notes |
|---|---|---|
| QL_q | Σ pinball_q(y, ŷ_q) / Σ\|y\| | the paper's normalised quantile loss (no factor 2) |
| SMAPE | mean 2\|y−ŷ\| / (\|y\|+\|ŷ\|) | on the median; 0/0 counted as 0 |
| NRMSE | RMSE / mean\|y\| | on the median |
| MASE | mean \|y−ŷ\| / s_i | `s_i` = mean \|y_t − y_{t−24}\| over the in-sample part of series i (`mase_scale`, `windows.insample`) |
| coverage | mean 1{lo ≤ y ≤ hi} | target 0.80 |
| coverage_finite | coverage over finite intervals only | consistent with width and Winkler |
| width | mean(hi−lo) / mean\|y\| | finite intervals only |
| Winkler | mean[(hi−lo) + (2/α)(lo−y)·1{y<lo} + (2/α)(y−hi)·1{y>hi}] / mean\|y\| | lower is better |
| crossing rate | share of forecasts with q0.1 > q0.5 or q0.5 > q0.9 | |
| inf_share | share of infinite interval ends | produced by ACI at its edges |

`by_horizon()` and `by_day()` give the same interval metrics per horizon step and per test
day (blocks of 24 origins in the hourly protocol).

---

## 8. Extension 1: conformal calibration (CQR and ACI)

**Goal:** make the nominal 80% interval `[q̂0.1, q̂0.9]` actually cover 80% of test
observations, after training and without touching the model.

### 8.1 CQR (`src/cqr.py`)

1. **Conformity score** on the calibration week (`scores_raw`):
   `E = max(lo − y, y − hi)`. It is negative inside the interval and positive outside.
2. **Conformal offset** (`conformal_offset`): the `⌈(1−α)(n+1)⌉`-th smallest of the `n`
   finite scores. If that index exceeds `n` the offset is `+∞` (whole line); if it is below
   1, `−∞` (empty set). ACI needs both edge cases.
3. **Three variants:**

| Variant | Fit | Apply to test |
|---|---|---|
| `fit_pooled` (B1) | one `Q` over all raw scores | `[lo − Q, hi + Q]` |
| `fit_horizon` (B2) | scores divided by the series scale `s_i`, one `Q_j` per horizon | `[lo − s_i Q_j, hi + s_i Q_j]` |
| `fit_two_tail` (B3) | lower score `(lo−y)/s_i` and upper score `(y−hi)/s_i` separately, each at level α/2, per horizon | `[lo − s_i Q^lo_j, hi + s_i Q^hi_j]` |

A negative offset means the raw interval was too wide, so CQR narrows it.

### 8.2 ACI (`src/aci.py`)

The test origins are walked **in time order**, with one adaptive level per horizon `j`:

```
for t in test origins (in order):
    for j in 1..24:
        r = t − delay[j]                          # newest origin whose horizon-j error is known
        a = α + γ · Σ_{s ≤ r} (α − ē[s, j])       # a = α while r < 0
        Q = conformal_offset(normalised calib scores at horizon j, a)
        interval = [lo − s_i Q, hi + s_i Q]  (or ±∞ at the edges)
        ē[t, j] = share of series outside their interval
```

- **Feedback delay** (`make_delay`): a horizon-`j` forecast is only verifiable `j` hours
  later, so `delay = [1..24]` in the hourly protocol and 1 (one day) in the daily protocol.
  Origin `t` uses only errors that are already observed, and a test checks this.
- **Choosing γ** (`pick_gamma`): the validation origins are split in half. The first half
  provides conformity scores, ACI runs on the second half for each γ in
  `{0.001, 0.005, 0.01, 0.02, 0.05, 0.1}`, and the γ with the lowest Winkler score wins.
- **Guarantee check** (`coverage_bound`):
  `B = (max(α, 1−α) + d·γ) / (T·γ)` with the maximum delay `d` and `T` test origins. `B > 1`
  means the long-run guarantee is vacuous: 1.29 for 7 daily origins, versus 0.44 (traffic)
  and 0.28 (electricity) for 145 hourly origins.

### 8.3 Driver (`scripts/05_calibrate.py`)

It loads the `val`, `calib` and `test` arrays, fits B1/B2/B3 on calib, picks γ on val, runs
ACI on test, and writes for each model (`base`, `cqr_pooled`, `cqr_horizon`, `cqr_twotail`,
`aci`):

| File | Content |
|---|---|
| `headline_<name>_<tag>_<protocol>.csv` | all accuracy and interval metrics |
| `coverage_by_horizon_...csv` | coverage / width / Winkler per horizon |
| `coverage_by_day_...csv` | the same per test day |
| `alpha_trace_...csv` | ACI level α_t and miss rate per origin and horizon |
| `example_series_...csv` | the largest series with raw and CQR intervals |

`scripts/04_diagnostic.py` is the step before: it measures raw coverage by horizon and by
series-size quartile and writes `diagnostic_<name>_<tag>.csv`.

---

## 9. Extension 2: count likelihood heads (NB and ZIP)

**Goal:** test the negative binomial (suggested but not tried by the paper) and a
zero-inflated Poisson against the quantile head. Implemented in `src/distributions.py` plus
small hooks in `train.py` and `predict.py`.

### 9.1 What the network sees (`train.to_fit_scale`)

- Count heads: `c = round(y · count_unit)` (`count_unit` = 100 for traffic, 1 for
  electricity), because neither dataset is a count.
- Gaussian: `log1p(y)` if `log_target`, else `y`.
- DeepAR-style scaling: the input history is divided by `ν = 1 + mean|history|`, while the
  target stays unscaled; location parameters are multiplied back by `ν`.

### 9.2 Parameters and likelihood (`params`, `nll`)

The network outputs two raw numbers per horizon:

| Head | Parameters | Constraint |
|---|---|---|
| Gaussian | μ = ν·raw₀, σ = ν·softplus(raw₁) + ε | σ > 0 |
| Negative binomial | μ = ν·softplus(raw₀) + ε, α = softplus(raw₁)/√ν + ε | Var = μ + αμ² |
| ZIP | λ = ν·softplus(raw₀) + ε, π = sigmoid(raw₁) | P(0) = π + (1−π)e^{−λ} |

The negative log-likelihood is computed in **float64**, because `lgamma` of counts up to
about 2·10⁵ is inaccurate in float32. NB uses
`lgamma(y+r) − lgamma(y+1) − lgamma(r) − r·log(1+αμ) + y·(log(αμ) − log(1+αμ))`
with `r = 1/α`. ZIP uses `logaddexp(log π, log(1−π) − λ)` at zero and
`log(1−π) + Poisson log-pmf` elsewhere.

### 9.3 Back to quantiles (`quantiles`)

- Gaussian: μ + σ·Φ⁻¹(q).
- NB: `scipy.stats.nbinom.ppf(q, n=1/α, p=1/(1+αμ))`.
- ZIP: 0 if q ≤ π, else `poisson.ppf((q−π)/(1−π), λ)`.

The quantiles are then divided by `count_unit` (count heads) or passed through `expm1`
(log Gaussian), so the prediction arrays have exactly the same format as the quantile
head's.

**Note.** For the Poisson part, Var(y) = E[y] / count_unit, so the chosen resolution fixes
ZIP's spread; NB's dispersion α absorbs this.

### 9.4 Run

```bash
python3 scripts/02_train.py traffic 1 --dist negbin        # or zip, gaussian
python3 scripts/03_predict.py traffic deeptcn daily 1 --dist negbin
python3 scripts/06_paper_metrics.py traffic daily seed1 negbin_seed1
# reported subset experiment (quantile, NB, ZIP back to back):
python3 scripts/09_traffic_subset.py ext2
```

---

## 10. Extension 3: learned memory length

**Goal:** let each series decide how much of the 168-hour window it needs, which is useful
for short or padded histories (cold start).

### 10.1 The gate (`src/adaptive.py`, `MemoryGate`)

```
age l = 0 (newest) … L−1 (oldest)
τ_i   = τ_max · sigmoid(f(context_i))         memory length of series i
m_i(l)= sigmoid((τ_i − l) / s)                soft mask
encoder input = [ y ⊙ m , m , covariates ]    (the mask is its own channel)
loss  = pinball + λ · mean(τ_i / L)           (penalty prefers short memory)
```

Modes: `none` (plain DeepTCN), `fixed` (hard cutoff at `fixed_len`), `global` (one learned
τ) and `series` (τ from a 2→16→1 MLP of the context features `log1p(n_observed)/log1p(L)`
and the zero ratio; `context()` counts steps since the first non-zero value).

### 10.2 Details that make it train

1. **Mask as a separate channel.** Otherwise `y·m = 0` would look like an observed zero.
2. **Start fully open.** `τ_max = L + 4·s₀`, the MLP bias starts at 4 (sigmoid(4) ≈ 0.98)
   and the mask channel's input weights start at **zero** (`DeepTCN(n_extra=1)` zeroes
   them), so at initialisation the model equals plain DeepTCN.
3. **Anneal the temperature.** `s` goes geometrically from `s₀ = L/10` to `s_end = 1`
   (`anneal()`), because a sharp mask has almost no gradient.
4. **Gate warm-up and smaller learning rate.** In `scripts/07_adaptive.py` the gate is
   frozen for `gate_warmup` epochs and trained at `lr × gate_lr_mult`, so an untrained
   encoder cannot shut it before it learns to use old history.

### 10.3 Synthetic check (`scripts/07_adaptive.py`, `configs/adaptive_synthetic.yaml`)

It trains rows `baseline`, `fixed_24`, `global_tau` and `per_series` on the data-panel
windows (L = 64, 12 ahead, 500 series per type), and reports QL50/QL90 overall and per type,
the median learned τ per type, and a **gate check** that passes if
`τ_C < τ_A < τ_B` and `τ_D` is the smallest. Outputs go to
`results/adaptive_synthetic_{rows,summary,tau,gate}.csv` and `_tau.png`.

### 10.4 On traffic (`src/xseries.py`, option `tau=True`)

τ_i comes from each lane's embedding via `tau_mlp` (32→16→1, bias 4), with
`τ_max = 168 + 4·16 = 232` h and `s` annealed from 16 to 1 over the epochs.

---

## 11. Extension 4: cross-series attention

**Goal:** let a lane's history summary borrow from correlated lanes. Implemented in
`src/xseries.py` (class `XTCN`, a compact DeepTCN).

### 11.1 Compact model XTCN

- Lane embedding `E` (dimension 32) plus hour (4) and weekday (3) embeddings.
- Encoder: dilations `1,2,4,8,16,20,32,32,32`, 16 channels (receptive field 295 h).
- Decoder exactly as in the paper: `head( R(X_fut) + h )`.

### 11.2 Attention on the encoder summary

```
A = softmax( E Eᵀ / √d  with the diagonal set to −∞ )     (N × N, rows sum to 1)
h ← h + α_att · (A h)                                     α_att: one scalar, starts at 0
```

- The diagonal is masked because otherwise `‖e_i‖²` dominates each row and `A ≈ I`.
- `α_att = 0` at the start means the model begins as the baseline, with no information
  flowing between lanes; a test checks this.
- **Every training step processes one origin with all lanes** (`08_traffic_ext.batch`),
  so `A` covers the whole panel.

### 11.3 Is attention meaningful? (`coherence_test` in `scripts/08_traffic_ext.py`)

1. De-seasonalise each training series (subtract its mean weekly profile) and compute the
   lane-by-lane correlation matrix `C` (diagonal 0).
2. Coherence = Σ_ik A_ik C_ik / N.
3. Null distribution: 200 random relabellings of `A` (`A[p][:, p]`). The p-value is the
   share of the null that is ≥ the observed value.

Our subset run gives coherence 0.190 against a null of 0.185 ± 0.001, with p < 0.005.

### 11.4 Runs

- `scripts/08_traffic_ext.py configs/traffic_{baseline,attn,tau,both}.yaml`: full 963
  lanes, 8 epochs × 100 steps, seeds 1–3, writes `results/xseries_<row>.csv` plus figures
  (attention heat map, τ histogram); `--table` prints the summary.
- `scripts/09_traffic_subset.py ext34`: the reported runs on 320 lanes, 9 epochs × 100
  steps, seed 1, with the coherence test and τ statistics, written to
  `results/subset_ext34.csv`.

**The 320-lane subset** (`make_subset`): lanes are sorted by mean training occupancy, split
into 8 bands and 40 are drawn from each (seed 0), keeping all 440 days so the splits, test
week and MASE denominators match the full dataset.

---

## 12. Running everything: step-by-step commands

### 12.1 Sanity check (no data needed)

```bash
python3 tests/run_all.py
python3 scripts/01_prepare_data.py synthetic
python3 scripts/03_predict.py synthetic naive daily
python3 scripts/03_predict.py synthetic naive hourly
python3 scripts/04_diagnostic.py synthetic naive daily
python3 scripts/05_calibrate.py synthetic naive daily
python3 scripts/05_calibrate.py synthetic naive hourly
```

### 12.2 Replication: traffic (repeat with `electricity`)

```bash
python3 scripts/01_prepare_data.py traffic
python3 scripts/02_train.py traffic 1                     # add --resume to continue a run
python3 scripts/03_predict.py traffic deeptcn daily 1
python3 scripts/03_predict.py traffic deeptcn hourly 1
python3 scripts/06_paper_metrics.py traffic daily         # paper Tables 4/5 next to ours
python3 scripts/06_paper_metrics.py traffic hourly
```

### 12.3 Extension 1: calibration

```bash
python3 scripts/04_diagnostic.py traffic seed1 daily
python3 scripts/05_calibrate.py traffic seed1 daily
python3 scripts/05_calibrate.py traffic seed1 hourly
Rscript R/plots.R  traffic seed1 hourly                   # 4 figures into results/
Rscript R/tables.R traffic seed1 hourly                   # LaTeX tables into results/
```

### 12.4 Extensions 2–4

```bash
python3 scripts/09_traffic_subset.py subset               # build the 320-lane subset
python3 scripts/09_traffic_subset.py ext2                 # quantile / NB / ZIP heads
python3 scripts/09_traffic_subset.py ext34                # baseline / attn / tau / both
python3 scripts/09_traffic_subset.py table                # print both tables
python3 scripts/07_adaptive.py configs/adaptive_synthetic.yaml          # Ext. 3, synthetic
python3 scripts/08_traffic_ext.py configs/traffic_baseline.yaml configs/traffic_attn.yaml \
        configs/traffic_tau.yaml configs/traffic_both.yaml              # Ext. 3/4, full traffic
```

### 12.5 Which command produces which result

| Result | Produced by | File |
|---|---|---|
| Replication accuracy (report Table 2) | `06_paper_metrics.py` | `results/paper_metrics_<name>_<protocol>.csv` |
| Raw coverage diagnostic | `04_diagnostic.py` | `results/diagnostic_<name>_seed1.csv` |
| Calibration table (report Table 3) | `05_calibrate.py` | `results/headline_<name>_seed1_<protocol>.csv` |
| Coverage by horizon / day, ACI trace | `05_calibrate.py` | `coverage_by_horizon_*`, `coverage_by_day_*`, `alpha_trace_*` |
| Example lane plot | `05_calibrate.py` | `example_series_*` |
| Count heads (report Table 4) | `09_traffic_subset.py ext2` | `results/subset_ext2.csv` |
| Memory / attention (report Table 5) | `09_traffic_subset.py ext34` | `results/subset_ext34.csv` |
| Training logs | `02_train.py` output | `results/TRAFFIC_RESULTS.md`, `ELECTRICITY_RESULTS.md` |

Measured run times on an 8-core CPU: the ext2 subset runs took 4.1 (quantile), 3.0 (NB) and
3.9 (ZIP) minutes; the ext34 rows took 2.6–2.8 minutes each. Full-data training is much
longer and benefits from a GPU.

---

## 13. File formats and naming conventions

| Pattern | Shape / content |
|---|---|
| `data/processed/<name>.npy` | `(n_series, n_hours)` float32 |
| `arrays/<name>_<tag>.pt` | best model state dict |
| `arrays/<name>_<tag>_resume.pt` | full resume state |
| `arrays/<name>_scale.npy` | `(n_series,)` series scale for conformal scores |
| `arrays/preds_<name>_<split>_<protocol>_<tag>.npy` | `(n_origins, n_series, 24, 3)`, quantiles 0.1/0.5/0.9, original units |
| `arrays/actuals_<name>_<split>_<protocol>_<tag>.npy` | `(n_origins, n_series, 24)` |

- `<name>` ∈ {traffic, electricity, synthetic, traffic_sub}
- `<split>` ∈ {val, calib, test}
- `<protocol>` ∈ {daily, hourly}
- `<tag>` = `seed1` (quantile head), `negbin_seed1`, `zip_seed1`, `gaussian_seed1` or `naive`

---

## 14. Tests

```bash
python3 tests/run_all.py          # runs all 9 test files
```

| File | What it checks |
|---|---|
| `test_metrics.py` | pinball, QL, SMAPE, NRMSE, MASE (and that its scale equals seasonal naive), coverage, width, Winkler, crossing, against hand-computed values |
| `test_windows.py` | split sizes, test block ends at the series end, no overlap, no target leaking into the input, batch indexing, calendar period, in-sample cut |
| `test_cqr.py` | order statistic and its edges, score sign, shrinking a too-wide interval, ≈ 1−α coverage on exchangeable data, offsets growing with horizon |
| `test_aci.py` | no look-ahead, base α before feedback, infinite sets at the edges, the bound on adversarial sequences, reaction to a harder test period, vacuous daily bound |
| `test_data_traffic.py` | PEMS line parsing, randperm direction detection on a planted permutation, hourly aggregation |
| `test_model.py` | output shape, non-negativity, causality, future covariates matter, loss decreases, rough quantile order |
| `test_distributions.py` | NLL vs scipy, quantiles invert the CDF, maximum likelihood recovers parameters, each head trains, old checkpoints still load |
| `test_adaptive.py` | causality with the gate, starts as baseline, τ and mask ranges, fixed mask, gradients reach τ, annealing, context features, panel padding |
| `test_xseries.py` | causality of every variant, baseline equivalence at initialisation, no cross-series leak when α_att = 0, attention rows, gradients reach α_att and τ |

---

## 15. Configuration reference

`config.py` (shared): `IN_LEN = 168`, `OUT_LEN = 24`, `QUANTILES = [0.1, 0.5, 0.9]`,
`TEST_DAYS = 7`, `CALIB_DAYS = 7`, `VAL_DAYS = 14`, `ALPHA = 0.2`, `MASE_PERIOD = 24`.

| Setting | traffic | electricity | traffic_sub | synthetic |
|---|---|---|---|---|
| n_series × n_steps | 963 × 10560 | 370 × 26304 | 320 × 10560 | 40 × 2400 |
| dilations | 1,2,4,8,16,32 | 1,2,4,8,16,32 | 1,2,4,8,16,32 | 1,2,4,8 |
| channels | 16 | 32 | 16 | 16 |
| kernel | 2 | 2 | 2 | 2 |
| batch size | 32 | 64 | 32 | 64 |
| learning rate | 5e-3 | 5e-3 | 5e-3 | 5e-3 |
| max epochs | 100 | 60 | 30 | 3 |
| patience | 5 | 5 | 5 | 10 |
| log target | no | yes | no | no |
| count unit (NB/ZIP) | 100 | 1 | 100 | 10 |

XTCN runs (`configs/traffic_*.yaml` and the subset): dilations `1,2,4,8,16,20,32,32,32`,
16 channels, embedding 32, lr 1e-2, `s0 = 16 → s_end = 1`, 100 steps per epoch (8 epochs for
full traffic with seeds 1–3; 9 epochs for the subset with seed 1).

Adaptive synthetic (`configs/adaptive_synthetic.yaml`): L = 64, 12 ahead, dilations
1–16, 30 epochs, batch 128, lr 5e-3, λ = 1e-3, gate warm-up 5 epochs, gate lr × 0.2.

---

## 16. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `missing file: data/raw/...` | Download the raw data into the paths in Section 3.1. |
| `expected 370 customers` | The electricity file must be read with `;` separator and `,` decimals and the timestamp as index (the loader does this; check that the file is unmodified). |
| Traffic hour count differs from 10560 | `01_prepare_data.py` prints the value to put in `config.py` (`n_steps`). |
| "neither direction beats the raw file order" | The `randperm` file does not match the day files; re-download PEMS-SF. |
| `loss went to nan` | Lower the learning rate in `config.py`; for NB, a lower lr also stabilises the dispersion parameter. |
| Training interrupted | Re-run the same command with `--resume`. |
| `no test arrays ... in arrays/` | Run `03_predict.py` for that dataset, protocol and tag first. |
| R script stops with `missing ... run scripts/05_calibrate.py first` | Run calibration for the same `<name> <tag> <protocol>`. |
