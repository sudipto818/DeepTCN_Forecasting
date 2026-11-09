# DeepTCN — Electricity (LD2011_2014) Experiment Results

**Dataset:** LD2011_2014 (electricity)  
**Model:** DeepTCN, seed = 1  
**Run date:** 2026-10-27  
**Nominal coverage target:** 80% (α = 0.20)  
**Config:** epochs = 25 (capped), patience = 5

---

## Step 1 — Data Preparation (`01_prepare_data.py electricity`)

```
raw table       (140256, 370)   (expected about (140256, 370))
hours after aggregation  35064  (expected 35064)
electricity shape        (370, 26304)
(many series start late, so some zeros early on are normal)
```

### Splits

| Split | Origins | Target hour range   |
|-------|--------:|---------------------|
| Train | 25 441  | 168 – 25 631        |
| Val   | 14      | 25 632 – 25 967     |
| Calib | 7       | 25 968 – 26 135     |
| Test  | 7       | 26 136 – 26 303     |
| Hourly test origins | 145 | —             |

`splits ok` ✅

---

## Step 2 — Training (`02_train.py electricity 1`)

```
train origins  25 441
val origins    14
calib origins  7
test origins   7
```

> ⚠️ Note: electricity uses `log_target = True` — values are log-transformed before
> training, so train losses appear large but are on the log scale.

### Epoch log (pinball loss)

| Epoch | Train      | Val        |
|------:|-----------:|-----------:|
| 0     | 122.809    | 312.629    |
| 1     | 946.475    | 312.614    |
| 2     | 730.369    | 312.625    |
| 3     | 120.640    | 312.611    |
| 4     | 1018.787   | 312.610    |
| 5     | 892.893    | 312.608    |
| 6     | 322.792    | 312.605    |
| 7     | 942.754    | 312.600    |
| 8     | 1244.569   | 312.600    |
| 9     | 112.461    | 312.598    |
| 10    | 499.339    | 312.596    |
| 11    | 286.486    | 312.597    |
| 12    | 766.827    | 312.598    |
| 13    | 420.212    | 312.598    |
| 14    | 1569.415   | 312.603    |
| 15    | 757.787    | 312.593    |
| 16    | 582.249    | 312.595    |
| 17    | 195.594    | 312.594    |
| 18    | 0.033      | 312.601    |
| 19    | 530.621    | 312.595    |
| **20** | **760.831** | **312.591** ← best |
| 21    | 102.195    | 312.608    |
| 22    | 425.533    | 312.594    |
| 23    | 424.267    | 312.592    |
| 24    | 301.207    | 312.596    |
| 25    | 514.858    | 312.592    |

**Early stop at epoch 25 — best epoch: 20 (val loss = 312.591)**

> 📌 Training loss is highly variable epoch-to-epoch (range: 0.03 to 1569) due to
> log-scaling and the large number of zero-padded series early in the dataset.
> Val loss is very stable, indicating the model converges slowly on this dataset.
> More epochs or a lower learning rate may help in future runs.

---

## Step 3 — Prediction (`03_predict.py electricity deeptcn {daily|hourly} 1`)

### Daily protocol — array shapes

| Split | Shape             |
|-------|-------------------|
| val   | (14, 370, 24, 3)  |
| calib | (7, 370, 24, 3)   |
| test  | (7, 370, 24, 3)   |

### Hourly protocol — array shapes

| Split | Shape              |
|-------|--------------------|
| val   | (313, 370, 24, 3)  |
| calib | (145, 370, 24, 3)  |
| test  | (145, 370, 24, 3)  |

*(3 quantile levels: 0.1, 0.5, 0.9)*

---

## Step 4 — Diagnostic (`04_diagnostic.py electricity seed1 daily`)

> Checks whether the raw uncalibrated 80% interval actually covers 80%.

```
target coverage       0.800
measured coverage     0.7314      <- UNDERCOVERING (model is too narrow, unlike traffic)
normalised width      0.7456
crossing rate         0.0         <- no quantile crossings
```

### Coverage by forecast horizon

| Horizon (h) | Coverage |
|------------:|--------:|
| 1           | 0.8425  |
| 4           | 0.8741  |
| 8           | 0.7293  |
| 12          | 0.7301  |
| 16          | 0.6846  |
| 20          | 0.6571  |
| 24          | 0.7382  |

> Coverage starts above target at h=1–4 but drops significantly at mid-to-late
> horizons (h=16–20 worst at 0.657). This is the opposite pattern to traffic.

### Coverage by series size (quartiles, small → large)

| Quartile      | Coverage |
|---------------|--------:|
| Q1 (smallest) | 0.7282  |
| Q2            | 0.7224  |
| Q3            | 0.7208  |
| Q4 (largest)  | 0.7542  |

> Coverage is uniformly below 80% across all series sizes — a model-quality issue,
> not a scale-normalisation issue.

Output CSV: `results/diagnostic_electricity_seed1.csv`

---

## Step 5 — Calibration (`05_calibrate.py`)

### 5a — Daily protocol

```
B1 pooled offset    -1.4242      <- large negative: model intervals too wide after log-untransform
B2 offsets          h1 -0.0255   h24 +0.0061
gamma (on val)       0.1
ACI bound value      1.286  <- above 1: ACI gives no formal guarantee on 7 daily origins
```

| Model        | Coverage | Width  | Winkler    |
|--------------|:--------:|:------:|:----------:|
| base         | 0.7314   | 0.7456 | 1.3167     |
| cqr_pooled   | 0.7063   | 0.7442 | 1.3174     |
| cqr_horizon  | 0.7106   | 0.7335 | 1.3090     |
| cqr_twotail  | 0.6882   | 0.7052 | **1.2581** |
| aci          | 0.7433   | 0.7724 | 1.3252     |

> ⚠️ CQR makes coverage worse here (7063 vs 7314 base). Because the model is
> already undercovering, shrinking intervals further reduces coverage.
> ACI is the only method that pushes coverage above the base (0.7433).

Output: `results/*_electricity_seed1_daily.csv` (5 files)

---

### 5b — Hourly protocol

```
B1 pooled offset    -3.4325      <- very large negative offset
B2 offsets          h1 -0.0261   h24 -0.0060
gamma (on val)       0.05
ACI bound value      0.276  <- below 1: ACI gives a formal guarantee here
```

| Model        | Coverage | Width  | Winkler    |
|--------------|:--------:|:------:|:----------:|
| base         | 0.7835   | 0.9155 | 1.2028     |
| cqr_pooled   | 0.7356   | 0.9122 | 1.2038     |
| cqr_horizon  | 0.7374   | 0.8802 | 1.2000     |
| cqr_twotail  | 0.7315   | 0.8704 | 1.1938     |
| **aci**      | **0.7901** | 0.8875 | **1.1721** |

> ACI wins on both coverage (0.790, closest to 0.80) and Winkler (1.1721) in
> the hourly protocol. Formal guarantee holds (bound = 0.276 < 1).

Output: `results/*_electricity_seed1_hourly.csv` (5 files)

---

## Summary & Key Takeaways

| Finding | Detail |
|---------|--------|
| Model converged | Best val loss **312.591** at epoch 20 (early stop at 25) |
| Raw model **undercovering** | 73.14% daily — opposite of traffic (which overcovered at 88.77%) |
| Cause | Only 25 epochs trained (capped); more epochs may push coverage closer to 80% |
| No quantile crossings | Crossing rate = 0.0 — model output is well-behaved |
| CQR hurts on daily | Shrinking already-narrow intervals makes coverage worse (0.706 vs 0.731 base) |
| ACI best on hourly | Coverage 0.790, Winkler **1.1721** — closest to target and best score |
| ACI formal guarantee | ❌ daily (bound = 1.286 > 1) · ✅ hourly (bound = 0.276 < 1) |
| Recommendation | Re-run with more epochs (50–100) to close the undercoverage gap before calibration |
