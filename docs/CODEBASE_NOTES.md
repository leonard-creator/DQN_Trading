# Codebase notes (Step 0 orientation)

**Date:** 2026-10-05 · **Branch:** `multi_Input_Claude` · **HEAD at time of writing:** `b7086a7` plus uncommitted edits in 5 files (see §1). The owner committed those edits afterwards as `076989b`.
**Purpose:** map what the code does today, check the "known facts" in `docs/02_claude_code_prompt.md` against the code, and list the gaps relative to the spec (`docs/03_extracted_plan.md`). No code was changed in Step 0.

Line references point to the **current working tree**, which includes the uncommitted edits.

---

## 1. Repository map

| File | Lines | Role |
|---|---|---|
| [agent/agent.py](../agent/agent.py) | 206 | `ReplayBuffer` (pre-allocated ring buffer), `build_model` (two-input Q-network), `Agent` (ε-greedy, Double-DQN update in a `tf.function`) |
| [env.py](../env.py) | 144 | `TradingEnv`: single asset, actions hold/buy-1/sell-1, long-only, average-cost inventory, differential Sharpe reward |
| [functions.py](../functions.py) | 149 | CSV loader, feature list + per-feature transforms, `FeatureScaler` (fit on train, saved as JSON), `get_window` |
| [train.py](../train.py) | 178 | CLI training loop: one env, greedy validation every episode, saves `_best` (by validation net €) and `_last` |
| [evaluate.py](../evaluate.py) | 127 | `evaluate()` greedy rollout + plot; CLI for standalone test |
| [scrape_data.py](../scrape_data.py) | 144 | yfinance download → indicators (ROC12, MFI14, FVolatility, SMA20_Rel, SMA50_Rel) → 70/15/15 chronological split into `train_data/` and `test_data/` |
| `train_data/`, `test_data/` | — | 21 CSVs, see §4 |
| `models/`, `graphs/`, `wandb/` | — | 45 model files, plots, 29 wandb runs (ignored here except as evidence) |

There are **no tests, no config files, no docs** besides the README (which still describes the original q-trader CLI and is out of date).

**Uncommitted changes in the working tree** (summarised in the earlier commit draft):
`tf.function` training step, Flatten instead of GAP, Huber in `compile`, SMA20/50 features, XLA disabled in three files.

---

## 2. Checking the "known facts" from the spec

| # | Claim in `02_claude_code_prompt.md` | Verdict | Evidence |
|---|---|---|---|
| 1 | Two-input Q-network: `market_in` (window × features) + `pos_in`, late fusion via `Concatenate` (lines 24/25) | **True**, different lines | [agent/agent.py:70-89](../agent/agent.py#L70-L89); `Concatenate` at line 84 |
| 2 | Conv branch uses `GlobalAveragePooling1D()` (line 15) | **Outdated.** True at HEAD; the uncommitted edit already replaced it with `Flatten()` (GAP is commented out) | [agent/agent.py:75-76](../agent/agent.py#L75-L76) |
| 3 | `model.compile(loss="mse")` (line 34) | **Misleading.** `compile` now says Huber, but training no longer goes through `compile`: the custom `_train_step` computes `reduce_mean(square(...))`, i.e. **the effective loss is still MSE** | [agent/agent.py:90](../agent/agent.py#L90), [agent/agent.py:157](../agent/agent.py#L157) |
| 4 | Replay batch 2048, buffer 100–200k | **Partly.** CLI defaults are batch 64 / buffer 100k. Logged runs used batch 64, 256, 2048 and 4096 with buffers 75k–200k. The latest run used 256 / 75k | `wandb/run-*/files/config.yaml` |
| 5 | Training plateaus around episode 800, then collapses | **Not verifiable.** All 29 logged runs ran exactly 200 episodes. What the logs *do* show is the plan's alternative explanation: **train P&L keeps rising while validation peaks early (episode ~10–75) and then decays**, i.e. overfitting to one fixed training path | §5 |
| 6 | ~8 years of daily bars for one asset; features include absolute prices | **Partly.** `DAX_new_train.csv` is 2015-03 → 2023-01 (~8 y) ✓. Absolute prices are **no longer** network inputs: `Close` is turned into a z-scored log return ([functions.py:25](../functions.py#L25)). But the **reward and costs are still in absolute currency units** (€ per index point/share), see §3.2 | [env.py:125-127](../env.py#L125-L127) |

---

## 3. How the pieces work today, and what is wrong

### 3.1 Model and learning ([agent/agent.py](../agent/agent.py))

- **Architecture.** Conv1D(32,3,causal) → Conv1D(64,3,causal) → Flatten, concatenated with a 3-dim position vector → Dense 64 → Dense 32 → 3 Q-values. Alternatives `lstm` and `mlp`. ✓ matches the spec's "keep late fusion".
- **Double DQN.** Implemented correctly in `_train_step`: online net picks `argmax`, target net evaluates it, `(1-done)` masks the bootstrap ([agent/agent.py:133-145](../agent/agent.py#L133-L145)). ✓
- **Loss.** MSE in practice (see fact #3). ✗ The spec wants Huber on *normalised* rewards.
- **No action masking.** Invalid actions (buy at max position, sell when flat) are silently treated as *hold* in the env ([env.py:117](../env.py#L117)). The network still learns a Q-value for them, and the target `argmax` can pick them. ✗ (spec Phase 2.4)
- **Target network.** Hard copy every 500 steps. No soft update (τ). ✗
- **No LR schedule.** Fixed Adam LR. ✗
- **No dueling / n-step / PER / QR head.** ✗ (optional in spec)
- **Speed.** `act()` calls `model.predict_on_batch` once per env step ([agent/agent.py:188](../agent/agent.py#L188)). That Python→TF round-trip dominates runtime: ~7 ms/step on CPU (measured: one DAX episode of 1,994 steps + 427 validation steps = 25 s wall time incl. ~8 s startup).
- **XLA disabled** at import time in three modules. In [evaluate.py:21-22](../evaluate.py#L21-L22) the env var is set *after* `import tensorflow`, so that line has no effect there (it works only because `agent.py` was imported first).

### 3.2 Environment and reward ([env.py](../env.py))

- **Action space.** hold / buy 1 unit / sell 1 unit, long-only, 0 … `max_position` units. A "unit" is one share or one index point, **not a fraction of capital**. So P&L, costs and exposure are not comparable across assets with different price levels.
- **Reward.** Differential Sharpe ratio (Moody & Saffell 1998) of the per-step mark-to-market P&L. The closed form matches the paper (uses A<sub>t-1</sub>, B<sub>t-1</sub>) ✓. It is clipped to ±5 during warm-up. Naming: code and logs call it "DSR"/`differential_sharpe`; the spec requires `diff_sharpe` to avoid confusion with the *Deflated* Sharpe Ratio (W2). ✗
- **Costs.** A flat amount per trade (default 1, runs used 2). On the DAX (~15,000 points) that is ~0.01 bp; on AAPL (~$150) it is ~130 bp. ✗ The spec wants proportional bps + half-spread, swept over 0/5/10/25 bp.
- **Only one reward.** No `mean_variance`, no `vol_scaled_pnl`, and the legacy "profit" reward was removed in `af5db08`. ✗ The spec requires `config/legacy.yaml` to reproduce MSE + GAP + profit reward.
- **State.** `[position/max_position, unrealised P&L fraction, running Sharpe]`. Spec wants one-hot position, normalised unrealised P&L and **holding time**. Holding time is missing. ✗
- **Episodes.** Every episode starts at t = 0 and walks the *same* full training series. The agent sees the identical path 200 times, which invites memorisation (see §5). ✗ (spec Phase 5: random ticker + random start, fixed horizon)
- **Inventory penalty** is quadratic in position size, not a holding-time penalty ρ(h) as in the spec. Default 0.

### 3.3 Data and features ([functions.py](../functions.py), [scrape_data.py](../scrape_data.py))

- **Loader** is robust to the different CSV layouts ✓, but there is no Parquet support and paths are hard-coded to `train_data/` / `test_data/`.
- **Features today:** z-scored log return, log-volume z-score, ROC12, MFI14/100, 14-bar realised vol, Close/SMA20, Close/SMA50.
  Compared with spec Phase 4: log return ✓, P/SMA ✓ (as a ratio, equivalent after z-score), rolling σ ✓ (14 bars instead of 20), **missing**: Bollinger %B, RSI, MACD histogram/P, V/SMA20(V) (log-volume z-score is used instead and is *not* stationary across years), ATR/P, lagged VIX.
- **Normalisation.** One global mean/std per feature, fit on the training file. No leakage across splits ✓, but it is not a rolling past-only z-score, so the scale drifts over a decade of data. ✗
- **`DEFAULT_FEATURES` now includes `SMA20_Rel`/`SMA50_Rel`, but only the `DAX_new_*` files have those columns.** Training on any other dataset with the defaults raises `ValueError`. ✗
- **Splits.** `scrape_data.py` splits each ticker 70/15/15 by its *own* row count, so cut dates differ per ticker (spec: same cut dates for all). No purge, no embargo (W4). ✗
- **Validation/test windows start cold.** Each split file is a separate series, so the first `window` observations of every validation/test env are zero-padded ([functions.py:147](../functions.py#L147)) instead of using the real preceding history. ✗
- Indicators are computed on the full series before splitting. They are all backward-looking (rolling), so this is **not** look-ahead ✓, but there are no tests that prove it.

### 3.4 Training and evaluation ([train.py](../train.py), [evaluate.py](../evaluate.py))

- **One seed, one run, one validation block.** No walk-forward, no multi-seed, no trial log, no baselines, no Deflated Sharpe, no PBO, no locked test period. This is W1, W11, W12: the biggest gap.
- **Metrics are not financial metrics.**
  - `val Sharpe` in the logs is the Sharpe of the *reward stream* (the differential Sharpe increments), not of returns, and it is not annualised ([evaluate.py:57](../evaluate.py#L57)).
  - `return_pct` = net € / sum of all buy prices ([evaluate.py:55](../evaluate.py#L55)), which is not a portfolio return.
  - `net_pnl` counts **realised** P&L only. An open position at the end of the period is ignored, so a buy-and-hold agent would score ~0.
- **Checkpoint selection.** `_best` = the episode with the highest validation net € out of 200. That uses the validation block 200 times. It's legitimate model selection, but it must be counted as trials and it makes the validation number optimistic.
- **Evaluation/training parameter mismatch.** `evaluate.py` defaults to `max_position=10`, `transaction_cost=1` and does not read what the model was trained with. Several runs used `max_position` 15–40. Because the position feature is `position/max_position`, evaluating such a model with the default silently changes its inputs.
- Seeds: `random`, NumPy and TF are seeded ✓, but op determinism is off and library versions are not logged.

---

## 4. Data actually available locally

| File(s) | Rows | Dates | Notes |
|---|---|---|---|
| `DAX_new_{train,val,test}` | 1994 / 427 / 428 | 2015-03-12 → 2026-06-05 | `^GDAXI` via yfinance; only set with SMA columns |
| `AAPL_{train,val,test}` | 1926 / 412 / 414 | 2015-01-23 → 2025-12-31 | single stock picked with hindsight (W5) |
| `DAXInd_{train,val,test}` | 1059 / 227 / 228 | 2020-01-22 → 2025-12-30 | |
| `DAXInd_2_{train,val,test}` | 523 / 112 / 113 | 2023-01-20 → 2025-12-30 | |
| `MSCI_World_{train,val,test}` | 341 / 73 / 74 | 2024-01-23 → 2025-12-31 | ~2 years only |
| `SundPGI_{train,test}` | 1048 / 211 | 2019-10-31 → 2024-10-31 | |
| `SAP.csv`, `SAP_test.csv` | 1258 / 211 | 2019-11-01 → 2024-10-31 | different column order |
| `iSharesMSCI_World_*` | 2517 / 199 | 2014-10-14 → 2024-10-14 | no indicator columns |

Takeaways: no ticker has a common cut date with another; total history is ≤ 11 years; several test periods (2024–2026) have **already been looked at** with plots in `graphs/`.

**Network access works**: yfinance returned daily data from 2000-01-03 to 2026-09-30 for all 33 candidate tickers checked on 2026-10-05 (used for the `PROTOCOL.md` proposal). Intraday history is not realistic from free sources (yfinance 1h ≈ last 730 days, W7), so **daily bars** are the practical frequency.

---

## 5. What the past runs say (29 wandb runs, 2026-07-06 → 07-09)

Median validation net € per 50-episode block (from `wandb/run-*/files/output.log`):

| Run | Val ep 1-50 | 51-100 | 101-150 | 151-200 | Train ep 1-50 → 151-200 | Best val episode |
|---|---|---|---|---|---|---|
| DAX_newFeat6_big3_1 | 37,257 | 26,407 | 22,595 | 10,325 | 106k → 118k | 18 |
| DAX_newFeat6_big2 | 36,969 | 32,400 | 28,209 | 33,241 | 103k → 163k | 44 |
| Dax_base | 3,308 | 7,861 | 4,270 | 2,673 | 45k → 141k | 112 |
| apple | −24 | 124 | 0 | −28 | 3.5k → 2.6k | 76 |

**Buy-and-hold check.** Holding `max_position` units through the same validation file:

| Validation file | B&H at max position | Logged agent validation net |
|---|---|---|
| DAX_new_val (×15 units) | **€56,158** | €10k–37k |
| DAX_new_val (×10 units) | **€37,438** | €15k–23k |
| AAPL_val (×20 units) | **$653** | ~$250 |

The comparison is rough: the agent is not always fully invested, and its "net" ignores the open position at the end. But it means the logged numbers cannot be read as skill. A **one-episode** smoke run on 2026-10-05 already scored €22.7k on DAX_new_val, the same range as 200-episode runs. Most of the validation P&L is long exposure to a rising index. This is exactly what the spec's baselines and Sharpe-based metrics are meant to expose.

---

## 6. Environment (`dqn_ml` conda env)

> **Update 2026-10-05 (after owner approval):** installed `tensorflow[and-cuda]==2.15.1`, `pytest 9.1.1`, `scipy 1.17.1` and `protobuf 4.25.9` into `dqn_ml`. TF now sees all 4 V100s, the protobuf errors are gone, and wandb 0.28 still works. `requirements.txt` now matches the env. The table below is the state **before** that fix.

| Item | Found | Problem |
|---|---|---|
| Python | 3.11.15 | ✓ |
| TensorFlow / Keras | 2.15.0 / 2.15.0 | `requirements.txt` says TF ≥ 2.16 and NumPy ≥ 2, so requirements and env disagree |
| NumPy / pandas | 1.26.4 / 3.0.3 | pandas 3 is very new (copy-on-write default), but the loader works |
| **GPU** | 4 × Tesla V100 32 GB, driver 565 (CUDA 12.7) | **TF sees no GPU.** The env has CUDA **11.8** runtime libs; TF 2.15 needs CUDA **12.2** + cuDNN 8.9 ("Cannot dlopen some GPU libraries"). Whether the July runs used the GPU can't be told from the logs; with the env as it is today, they would have run on CPU |
| protobuf | 7.35.1 | TF 2.15 expects protobuf < 5. Causes `MessageFactory ... GetPrototype` errors at startup (noise, not fatal so far) |
| pytest, scipy | **not installed** | spec requires pytest |
| pyyaml, yfinance, wandb, matplotlib | 6.0.3, 1.5.1, 0.28.0, 3.11.0 | ✓ |
| Machine | 40 cores, 247 GB RAM | Shared server: keep any job ≤ 90 % of CPU/RAM/GPU |

---

## 7. Gap analysis against the spec

✓ = done · ◐ = partial · ✗ = missing

| Spec item | Status | Where / note |
|---|---|---|
| **Phase 1: harness** — walk-forward + purge/embargo, ≥10 seeds, metrics, Deflated Sharpe, PBO, trial log, baselines, permutation + Holm, cost sweep, locked test | ✗ | nothing exists yet |
| Phase 2.1 Flatten + causal conv, late fusion | ✓ | uncommitted edit |
| Phase 2.2 Huber on normalised rewards | ✗ | Huber compiled but bypassed; no reward normaliser |
| Phase 2.3 Double DQN | ✓ | `_train_step` |
| Phase 2.4 action masking (ε-greedy and target) | ✗ | invalid actions = silent hold |
| Phase 2.5 LR schedule, soft τ / hard K | ◐ | hard K = 500 only |
| Phase 2.6 dueling / n-step / PER | ✗ | |
| Phase 2.7 QR-DQN + CVaR | ✗ | optional, after M3 |
| Phase 3 `reward_type` ∈ {mean_variance, diff_sharpe, vol_scaled_pnl} | ◐ | only differential Sharpe, named "DSR" |
| Phase 3 holding penalty ρ(h), reward normaliser, per-component logging | ✗ | |
| Phase 3 state: one-hot position, unrealised P&L, holding time | ◐ | holding time missing; position is a scalar fraction |
| Phase 4 pluggable loader, no paid APIs | ◐ | CSV only; yfinance (free) |
| Phase 4 feature set (10 features) | ◐ | 3 of 9 daily-relevant features present |
| Phase 4 rolling past-only z-score | ✗ | global train z-score |
| Phase 4 tests (look-ahead, NaN, VIX lag, purge) | ✗ | |
| Phase 5 multi-asset env, survivorship-free universe, leave-assets-out | ✗ | |
| Phase 6 stationary bootstrap, jitter | ✗ | optional |
| Engineering: YAML configs, `legacy.yaml`, pytest, version logging, CSV/TensorBoard logs | ✗ | argparse + wandb only |

### W1–W15 status

| W | Topic | Status today |
|---|---|---|
| W1 | evaluation protocol | ✗ |
| W2 | "DSR" naming | ✗ (reward is called DSR) |
| W3 | shuffled master tensor | n/a: never built; the env-based design is already right |
| W4 | overlapping windows / purge | ✗ |
| W5 | survivorship / hindsight assets | ✗ (AAPL, DAX picked by hand) |
| W6 | H4 bar arithmetic | n/a: daily data only |
| W7 | data-source limits | ◐ daily yfinance is fine; intraday not feasible |
| W8 | VIX look-ahead | n/a: VIX not used yet |
| W9 | reward scale vs Huber δ | ✗ |
| W10 | exponential holding penalty | n/a: not implemented (good default) |
| W11 | baselines | ✗ |
| W12 | trials as multiple testing | ✗ |
| W13 | buffer size as config | ◐ CLI flag, default 100k |
| W14 | augmentation leakage | n/a: none yet |
| W15 | tail risk | ✗ (optional) |

---

## 8. Decisions needed before Phase 1

These are listed in the PROTOCOL draft and in the change report:

1. **GPU/test tooling.** Fixing the GPU and installing `pytest`/`scipy` means changing the `dqn_ml` env (outside this folder).
2. **Universe, frequency, date range, frozen test period** (PROTOCOL.md §2–§4).
3. **Position sizing.** Units of one share vs fraction of capital, long-only vs long/short (affects every metric).
4. **Baseline commit.** Commit the pending working-tree edits before the refactor, so legacy behaviour has a clean reference point.

---

## v2 — Code map for the v2 programme (2026-10-06; PROTOCOL Part II §V12, item 1)

What each v2 component touches in the **current** code (state after M4, commit with `rl/`), and where it will hook in. All new behaviour goes behind config flags; the v1 configs keep their meaning and hashes.

| Concern | Where it lives now | How it works now | v2 change and hook |
|---|---|---|---|
| **Execution timing** | `harness/backtest.py::backtest` | Exposure decided at the close of bar t earns close t → t+1 (**lag 0**); every block starts flat; cost on \|Δexposure\| | **Done (0a):** `backtest(..., lag=L)` and `run_experiment(..., execution_lag=L)`; lag 1 = the exposure decided at t is first held from t+1 → t+2, block starts flat (§V12.1 item 1). Applies to agents and baselines alike |
| **Costs** | `harness/backtest.py::CostModel`, `scenario_cost`; `config/cost_scenarios.yaml` | Proportional rate + fixed fee per transaction + holding cost; the same object feeds the env reward | V1's cost-structured head uses κ = rate / σ̂ (plus φ = fee / σ̂ for neo-broker variants, §V6.1) |
| **Actions and positions** | `rl/env.py::VecTradingEnv.step` | Exposure k/K (K = 4, long-only); actions hold / +¼ / −¼; invalid actions masked (`valid_mask`) | **Done (V1):** `agent.replay.mode: exogenous` → target exposures e′ ∈ {0, ¼, ½, ¾, 1} chosen directly (one transaction) by `rl/exogenous.band_exposures` |
| **Reward** | `rl/env.py::step` (`reward_type`) | diff_sharpe, pnl, profit, mean_variance, vol_scaled_pnl, active_return; shadow cost; holding penalty; `rc_*` components | V1 needs a Markov reward: `vol_scaled_pnl` (already used since M4) + lazy anchor η = 0.01 for e′ ≠ 1 (training only) |
| **Replay** | `rl/replay.py` (`ReplayBuffer`, `PrioritizedReplay`, `NStepCollector`) | Stores bar indices g, position, mask, action, reward, next g; filled by on-policy ε-greedy collection | **Done (V1):** `rl/exogenous.train_exogenous`. A static dataset of every (ticker, training day); all 5 U-targets computed inside the loss; no environment loop (§V9, item 2). Same return values as `rl/trainer.train_run`, selected in `rl/policy.run_job` |
| **Agent / update** | `rl/agent.py::DQNAgent` (`targets`, `_train_step`, `act`) | Double DQN, masked argmax, Huber/MSE, soft/hard target, cosine LR | **Done (V1):** `rl/exogenous.UAgent` (market input only, 5 outputs) with Q = U − κ\|e′ − p\| − φ·1[e′ ≠ p]; Double-DQN targets over e″ with the cost matrix; anchor η and buy-and-hold bias prior. Open for V2: HL-Gauss loss, LayerNorm, K = 10 bootstrapped heads + gating rule |
| **Encoder** | `rl/networks.py` (`conv`, `conv_transformer`, `lstm`, `mlp`; dueling) | Late fusion with the position vector | **Done (V1):** `build_q_network(pos_dim=0)` drops the position branch; the encoder is R0′'s conv-transformer |
| **Features** | `rl/features.py` (legacy, scaler fitted on inner-train); `rl/features_m4.py` (causal M4 set, trailing z-score) | M4 features computed once per experiment in `rl/policy.py::DQNPolicy.prepare` | **Done (0b), `agent.m4.pipeline: v2`:** Q1 lag of ^GDAXI's `resid`/`resid_cum30` (`early_close_tickers`); Q4 mask of `vol_rel20` (`mask_volume`); availability flag `resid_avail` with the 450-bar warm-up (`warmup_bars`) and z-score minimum 126 bars. v1 mode unchanged (100 % reproduction in 0a). Open for V3: D1 cross-asset context, D2 multi-horizon trend + BOCPD (+ their flags), a feature cache on disk (§V9, item 1) |
| **Purge** | `harness/splits.py::fold_ranges`; P from `harness/splits.py::purge_bars` and `rl/policy.py::run_job` (`P = max(protocol, W + n_step)`) | P = 25 for all v1 agents | v2: P = W + H_max = 40 for every v2 config, incl. R0′: `splits.purge_horizon: 20` in `config/v2/R0prime.yaml` (later v2 configs inherit from it) |
| **Training loop** | `rl/trainer.py::train_run` | Vectorised env collection, update ratio, inner-validation scoring (`greedy_exposures`, `score_ranges`), fixed budget | V1: a second loop for exogenous replay (batched over (ticker, day)). **Fixed training length: no early stopping** (owner decision) |
| **Parallel jobs** | `rl/policy.py` (spawn pool, one GPU per worker, ≤ 90 % CPU) | One job per (seed, training cut) | `rl/policy.train_jobs(jobs, runtime)`: the pool, shared by `DQNPolicy.prepare` and the synthetic worlds. Long runs go through the detached queue `scripts/run_queue.sh experiments/v2_queue.txt` (survives session disconnects; lines can be appended while it runs) |
| **Re-scoring** | `rl/policy.py::StoredExposurePolicy`; `scripts/rescore_costs.py`; `exposures.npz` per job (selected and last checkpoint) | Exposures of every agent trial since M2 are stored | **Done (0a):** `scripts/rescore_v2.py` → `experiments/reports/V2_0a_rescore.md`. Exposure-based diagnostics in `harness/diagnostics.py` (`sleeve_timing`, `diagnose`, `learning_curve_stats`); Q-value diagnostics in `rl/diagnostics.py` (re-runs `best.weights.h5` on CPU through `rl/policy.build_job_data`, checks reproduction). For new runs these are computed by re-running the same scripts |
| **Baselines** | `harness/baselines.py` | buy-and-hold, momentum-60, MACD, random | **Done (0a):** `vol_target` added (variance-managed, §V12.1 item 8; `config/experiments/baseline_vol_target.yaml`); TSMOM-252 = the existing `momentum_252` |
| **Synthetic worlds** | – | – | **Done (0c):** `harness/synthetic.py` (W-null, W-vol, W-regime, oracles, calibration, verdicts, stopping rule 1). Entry point: `scripts/run_agent.py --synthetic / --calibrate`; injected prices via `run_experiment(prices=…)` (never logged, never unlocks the test) |
| **External data** | `scripts/download_external.py` (owner-run only) | – | Q2 audit inputs → `data/audit/` (downloaded 2026-10-06; stooq failed, owner: not needed); French portfolios → `data/longhistory/french/`; FRED → `data/external/fred/`. Data audit: `scripts/audit_data.py` → `DATA_AUDIT.md` |

**Shared report code (2026-10-07).** `harness/report.py` holds the report helpers used by every report script (latest trial, median (IQR) tables, fold and cost tables, paired test, deploy baselines, stability rows). `harness/diagnostics.py` holds the §V7 scoring of one strategy (`strategy_scores`) and of one agent trial (`agent_scores`). `scripts/analyze_v2.py` is the generic report for every v2 step (`--step`, `--trials`, `--reference`, §V8 adoption rule). Regenerating the M2/M3/M4/cost and 0a reports with the shared code gave identical tables; the only differences were the deflated Sharpe values, because N_trials has grown since those reports were written.

**Step 4, long-history pretraining (2026-10-07).**
- `harness/longhistory.py` parses the French CSVs (`french_table`) and builds the 6-series price dict inside the window (`load`). Rows after 2007-06-29 are never returned.
- `harness/longhistory.run` pretrains a configuration once per seed into `agent.pretrain.dir`, reusing `rl/policy.make_jobs` + `train_jobs` and the synthetic helpers `ohlcv` / `repoint`.
- `rl/exogenous.train_exogenous` loads `<dir>/agent/PRE_s<seed>/best.weights.h5` when `agent.pretrain` is set.
- `rl/features_m4.py` has a second availability flag, `vix_avail`.
- Entry point: `scripts/run_agent.py --pretrain`.


**Hyperparameter search (2026-10-07, PROTOCOL Part II §V12.1 item 23).**
- `harness/hpo.py`: the search on the synthetic worlds (never a trial).
  - `sample` draws a balanced random design over `SPACE`; `overrides` scales the budget (about 5 checkpoints per run); `score` gives the mean oracle capture in W-swing and W-regime minus the W-null penalty.
  - `run` does the successive halving with one CSV per round in `experiments/reports/v2_hpo/` (a finished round is re-used on restart); `write_report` writes `experiments/reports/V2_hpo.md`.
  - `write_configs` writes `config/v2/H1.yaml`, `H2.yaml` and their neo-broker twins `H1nb.yaml`, `H2nb.yaml` from the last round's best two.
  - Entry point: `scripts/run_agent.py --config config/v2/V1b.yaml --hpo`.
- `harness/synthetic.py`:
  - world W-swing (`SWING`, `simulate`, the per-ticker Kalman oracle `kalman_target`); oracles are now (bars × tickers);
  - `run_many()` trains several configurations in one `train_jobs` pool, with paths and features built once per (world, seed); `run()` wraps it;
  - `calibrate(worlds=…)` also returns the oracle's turnover.
- `rl/exogenous.py`: `hl_probs` + `UAgent(support=…)` (HL-Gauss, support stored in `info.json` as `hl_support`), `UAgent._network` (frozen random prior added to the output), AdamW (`algo.weight_decay`), `agent.prior: none`.
- `rl/policy.cost_function(cfg)`: the costs an agent trains and decides under, including `agent.env.cost_positions`.
