# Project Prompt for Claude Code (v2): Risk-Aware Cross-Asset DQN Trading Agent

> Copy everything below the line into Claude Code, either as your first message or as `CLAUDE.md` in the repository root. Put these files in `docs/`:
> - `03_extracted_plan.md` (plan v2, the authoritative spec)
> - `evidence-dqn-trading.md` (literature)
> - `01_stitched_text.md` (the original German design discussion)

---

## Mission

This repository contains a Keras/TensorFlow **DQN agent for algorithmic trading**. Your job is to refactor it into a **risk-aware, cross-asset DQN** and, just as important, to build an **evaluation harness that is honest about overfitting**.

The research question is whether the agent beats simple baselines **net of costs, over many random seeds, on data it never saw during development**. A well-documented negative result counts as success. A positive result produced by tuning on the test period counts as failure.

The authoritative spec is `docs/03_extracted_plan.md`. Sections 2 (weaknesses W1–W15), 3 (evaluation protocol) and 4 (phases A–F) define the work. If this prompt and the spec disagree, the spec wins. Ask me if anything is unclear.

## Known facts about the current code (from a screenshot review; verify them)

- `build_model(arch, window, n_…)` builds a two-input Q-network: `market_in` (window × features) and `pos_in` (position vector), combined by late fusion via `Concatenate` (around lines 24/25). **Keep this design.**
- The `conv` branch uses `GlobalAveragePooling1D()` (around line 15). Replace it with `Flatten()`.
- `model.compile(loss="mse")` is around line 34. Switch to Huber.
- Replay batch 2048, buffer 100–200k. Training plateaus around episode 800 and then collapses.
- Data: about 8 years of daily bars for one asset; features include absolute prices.

## Step 0: Orient (no code changes yet)

1. Map the repository: model, environment and reward, replay buffer, training loop, data loading, configuration.
2. Write `docs/CODEBASE_NOTES.md` with what you found, including where my facts above are wrong.
3. Draft `PROTOCOL.md` (template below) and propose a concrete asset universe and date splits **given the data actually available locally**.
4. **Stop and wait for my approval** before continuing.

## Phase 1: Evaluation harness first (spec §3)

Build the harness before you change the agent, so that every later change is measured the same way.

- **Splits:** a frozen final test period (default: the last 15% of the timeline). Walk-forward train/validation folds before it. At every boundary, purge `W + reward_horizon` bars and add an embargo (configurable). Use the same cut dates for all tickers.
- **Seeds:** a `run_experiment(config, seeds=range(10))` entry point. Store per-seed equity curves and metrics.
- **Metrics:**
  - net return, annualised Sharpe and Sortino, max drawdown, Calmar, turnover, average holding period, hit rate, exposure
  - the **Deflated Sharpe Ratio** (Bailey & López de Prado 2014), using `n_trials` taken from the trial log
  - **PBO via CSCV** (Bailey et al. 2015) across all configurations in the log
- **Trial log:** append-only `experiments/trials.csv` holding config hash, seeds, fold metrics and timestamp. Every run counts as a trial.
- **Baselines, net of the same cost model:** buy-and-hold, random agent, sign-of-past-k-return momentum, MACD crossover.
- **Statistics:** paired permutation test over (seed × fold) against each baseline, with Holm correction.
- **Cost model:** proportional bps plus a half-spread. Sweep 0, 5, 10 and 25 bp.
- **Guard:** the test period can only be evaluated by `scripts/final_test.py --i-am-sure`, which writes a lock file. A second run refuses to start.

## Phase 2: Model and algorithm (spec §4A)

1. Use `Flatten()` in the conv branch with causal padding. Keep late fusion.
2. Huber loss (configurable δ, default 1.0) **applied to normalised rewards**.
3. Double DQN targets: the online network chooses argmax, the target network evaluates it.
4. **Action masking on Q-values:** no `buy` at maximum long, no `sell` when flat (or no `short` if shorting is disabled). Apply the mask at ε-greedy selection **and** inside the target max/argmax. Unit-test both.
5. LR schedule (ExponentialDecay or CosineDecay). Soft target update τ (default 0.005), with hard updates every K steps as an alternative.
6. Behind flags, each ablated on its own: dueling head, n-step returns, PER (proportional sum-tree, α≈0.6, β annealed 0.4→1.0, importance-sampling weights multiplied into the per-sample Huber loss).
7. Optional, behind a flag, only after M3: a QR-DQN head with CVaR_α action selection (see Lim & Malik 2022 in the evidence log for the convergence caveat).

## Phase 3: Reward (spec §4B)

- `reward_type` ∈ {`mean_variance`, `diff_sharpe`, `vol_scaled_pnl`}:
  - `mean_variance`: ΔPnL − (λ/2)·ΔPnL² − ρ(h) − c
  - `diff_sharpe`: the online Differential Sharpe Ratio (Moody & Saffell) with EMA parameter η. **Name it `diff_sharpe` and never just "DSR"**, because "DSR" is also used for the *Deflated* Sharpe metric.
  - `vol_scaled_pnl`: position·return scaled to a volatility target, minus costs (Zhang, Zohren & Roberts 2019)
- Holding penalty ρ: default `none`. Options are `linear(k)` and `exp(k, α)`, kept for ablation.
- Use a running reward normaliser, configurable.
- The state contains the one-hot position, unrealised PnL (normalised) and holding time (normalised).
- Log each reward component per step (aggregated per episode).

## Phase 4: Data and features (spec §4C)

- A pluggable loader for local CSV/Parquet OHLCV files. **No hard-coded paid APIs.** Document the data source in `PROTOCOL.md`.
  - Known limits: yfinance 1h ≈ last 730 days only; Alpha Vantage free tier = 25 requests/day.
- **Bar frequency** is configurable. If intraday, align bars to the session. Do not assume 4 bars per day for US equities (the session is 6.5 h). Derive "bars per week" from the data.
- Per-ticker features, all dimensionless and stationary:
  1. log return
  2. P/SMA(n) − 1
  3. Bollinger %B
  4. RSI(14) in [0,1]
  5. (MACD − signal)/P
  6. V/SMA20(V)
  7. ATR/P
  8. rolling σ(returns, 20)
  9. VIX **as-of joined with a one-bar lag**
  10. time of day as sin/cos, only if intraday
  - Optional: day of week as sin/cos, and a 50-bar volatility "beta proxy"
- Rolling z-score that uses past data only.
- **Tests:**
  - no look-ahead: perturbing future values must not change the feature at time t
  - no NaN after warm-up
  - VIX lag correctness
  - purge/embargo correctness at split boundaries

## Phase 5: Cross-asset environment (spec §4D)

- **Do not** concatenate and shuffle windows into one supervised tensor. That breaks the s→s′ transitions DQN needs.
- `MultiAssetTradingEnv`: on `reset()`, sample a ticker (uniformly or weighted by length) and a start index inside the current training fold. Step forward in time for a fixed horizon. Precomputed per-ticker feature arrays serve as fast lookups.
- **Universe:** chosen to avoid survivorship bias (e.g. index membership at the start date, or a pre-declared list of liquid ETFs/futures). Write the rationale in `PROTOCOL.md`.
- A **leave-assets-out** evaluation: about 20% of tickers never appear in training.
- Optional asset-ID embedding, ablated against "no ID".

## Phase 6: Augmentation (optional, training fold only; spec §4E)

- Stationary bootstrap (random block lengths, mean block length configurable) on training-fold returns. Rebuild prices and features from the resampled returns.
- Gaussian jitter on normalised features (σ configurable, default 0.01).
- Assert that augmentation never touches validation or test data.

## Milestones (one PR/commit group each; report results after each)

| M | Content | Go criterion |
|---|---|---|
| M1 | Harness + baselines (Phase 1) | Baseline metrics are reproducible across runs |
| M2 | Model fixes (Phase 2) on the *original* single asset | Stable across ≥10 seeds, no late collapse |
| M3 | Reward variants (Phase 3) | Report the median and IQR of validation Sharpe vs baselines |
| M4 | New features + multi-asset environment (Phases 4–5) | Cross-asset ≥ single-asset on leave-assets-out |
| M5 | Only after I approve: a single run on the frozen test period | Report H1 from `PROTOCOL.md`, pass or fail |

## `PROTOCOL.md` template (fill this in at Step 0)

```
Hypothesis H1: <agent> beats buy-and-hold and momentum on net Sharpe over seeds × folds.
Success: Deflated Sharpe > 0.95, PBO < 0.5, Holm-adjusted p < 0.05 vs each baseline.
Universe + selection rule: ...
Data source(s), frequency, date range: ...
Splits: folds ..., purge ..., embargo ..., frozen test ...
Trial budget: max N configurations × 10 seeds.
Cost model: ...
```

## Engineering rules

- Python ≥3.10, TensorFlow 2.x. Dependencies: numpy, pandas, tensorflow, pytest, pyyaml, and optionally pandas-ta. Ask before adding more.
- One YAML config per experiment. No magic numbers. Seed numpy, TF and `random`, and log library versions.
- Keep the old behaviour reproducible with `config/legacy.yaml`, which uses MSE, GAP and the profit reward.
- Small commits with clear messages. Do not delete my code paths; put new behaviour behind flags.
- Logging to CSV/TensorBoard. Select checkpoints by **validation** metrics only.
- Research code only. No live trading or order execution.

## Definition of done

- `pytest` is green, including the look-ahead, masking, Double-DQN target, PER and purge tests.
- `RESULTS.md` covers:
  - legacy vs new, over ≥10 seeds and all walk-forward folds
  - baselines
  - cost sweep
  - the Deflated Sharpe and PBO numbers
  - the leave-assets-out results
  - and, only after my approval, the single frozen-test result
- The README explains how to add assets, switch the reward, run the harness, and why the test period is locked.
