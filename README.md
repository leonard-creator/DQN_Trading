# Deep Q-Trader

This project began from the q-trader by [edwardhdlu](https://github.com/edwardhdlu/q-trader/tree/master). It is being turned into a **risk-aware, cross-asset DQN** together with an **evaluation harness that is honest about overfitting**.

The research question: *does the agent beat simple baselines net of costs, over many random seeds, on data it never saw during development?* A well-documented negative answer counts as a valid result.

> Research code only. No live trading, not investment advice.

## Project status (2026-10-06)

**M1–M4 are done; the v2 programme is approved and frozen** ([`PROTOCOL.md`](PROTOCOL.md) Part II). 16 of 50 trials are used; the test period is untouched.
- **v1 result:** no agent beats buy-and-hold net of costs. Best on the 26-ETF H1 set: the conv-transformer at a median Sharpe of 0.44 vs 0.67. There is no timing skill even before costs.
- **v2 idea:** learn small, confident deviations from buy-and-hold, with costs built into the decision (a no-trade band), using every market day for every possible exposure. At most 10 new trials, with a stopping rule that protects the test period.
- **v2 Steps 0a + 0b done (2026-10-06):**
  - Re-scoring confirms that no trial has timing skill; all agents survive a one-day execution delay.
  - The value estimates are noise-dominated: action gaps are smaller than the TD noise.
  - The data is clean apart from the ^GDAXI close-time leak, which is now fixed (pipeline v2) and which the v1 agents never exploited.
- **v2 Step 0d done (2026-10-07):** the clean reference R0′ scores 0.43, against 0.44 for R0, so the Q1 fix changes nothing.
- **v2 Step 0c done (2026-10-07):**
  - R0′ fails all three synthetic worlds. V1 stops the overtrading but does not time, so **stopping rule 1 paused real-data trials**.
  - The synthetic iteration found **V1b** = V1 + MSE + 20-step targets + 10 gated heads. It passes W-null and W-regime on the pre-registered seeds (+0.26 Sharpe, 72 % of the oracle). On 5 new seeds it gains only +0.05; pooled over 10 seeds it gains +0.06 (16 %), so it is safe but weak.
- **Next (owner decision):** run V1b on real data as Step 1 (`config/v2/V1b.yaml`, +1 trial), or first strengthen it on synthetic data (more regime events: Step 4 long history; larger ensembles). The queue is idle; all results are in `experiments/reports/V2_0c.md`.
- Details: [`RESULTS.md`](RESULTS.md), plan summary in [`docs/03_extracted_plan.md`](docs/03_extracted_plan.md) §8.

| Document | What it is |
|---|---|
| [`PROTOCOL.md`](PROTOCOL.md) | Pre-registration (approved): hypothesis H1, universe, splits, trial budget, costs, statistics |
| [`RESULTS.md`](RESULTS.md) | Results per milestone (M1–M4) and per v2 step |
| [`DATA_AUDIT.md`](DATA_AUDIT.md) | v2 Step 0b data audit: effective sample size, drawdown events, luck level, data-quality checks Q1–Q6 |
| [`docs/03_extracted_plan.md`](docs/03_extracted_plan.md) | The authoritative plan: weaknesses W1–W15, evaluation protocol, phases A–F |
| [`docs/02_claude_code_prompt.md`](docs/02_claude_code_prompt.md) | Engineering brief derived from the plan (milestones M1–M5, definition of done) |
| [`docs/evidence-dqn-trading.md`](docs/evidence-dqn-trading.md) | Literature log behind each design choice |
| [`docs/CODEBASE_NOTES.md`](docs/CODEBASE_NOTES.md) | Step 0 audit: what the original code does, which spec facts hold, gap analysis |

## Evaluation harness (`harness/`)

Every strategy, baseline or agent, is measured by the same code on the same dates with the same cost model.

| Module | Role |
|---|---|
| `harness/config.py` | Merge `config/protocol.yaml` with an experiment YAML; config and code hashes |
| `harness/data.py` | Per-ticker raw data in `data/raw/`, test-period guard |
| `harness/splits.py` | 5 anchored walk-forward folds, purge P = W + H, embargo, inner validation |
| `harness/backtest.py` | Exposure → net daily returns; proportional cost + half-spread; equal-weight portfolio |
| `harness/baselines.py` | Buy-and-hold, random agent (same action space + masking), momentum, MACD |
| `harness/metrics.py` | Sharpe, Sortino, max drawdown, Calmar, turnover, holding period, hit rate, exposure |
| `harness/stats.py` | Deflated Sharpe ratio, PBO via CSCV, paired permutation test, Holm |
| `harness/trials.py` | Append-only `experiments/trials.csv` |
| `harness/experiment.py` | `run_experiment(config, seeds)`: the single entry point |

```bash
conda activate dqn_ml
python scripts/download_universe.py        # once: 33 ETFs/indices + ^VIX into data/raw (skips existing files)
python -m pytest                           # 63 tests: purge, look-ahead, costs, statistics, guard, masking, DDQN, PER
python scripts/run_baselines.py            # M1: baselines over 5 folds x 10 seeds x cost sweep
```

From Python:

```python
from harness.experiment import run_experiment
res = run_experiment("config/experiments/baseline_momentum.yaml", seeds=range(10))
res.summary["train@10bp"]["sharpe_median"]
```

**Adding an asset.** Add the ticker to a bucket in `config/protocol.yaml` **and** log the change in `PROTOCOL.md` §10, because it changes the universe and therefore every config hash. Then run `python scripts/download_universe.py`; only the new ticker is downloaded.

**Why the test period is locked.** Every look at 2023-10 → 2026-09 that influences a design decision turns it into development data, and the final result would then be optimistic in a way no statistic can correct (PROTOCOL §9; Arnott, Harvey & Markowitz 2018). So `harness.data.load_prices()` cuts every series before 2023-10-02. The only way past that is `scripts/final_test.py --i-am-sure`, which first writes `experiments/FINAL_TEST.lock` and refuses to run a second time. The test period is evaluated as three 12-month sub-blocks that share one training cut (PROTOCOL §10a.8).

## Agent (`rl/`)

The DQN agent is a harness policy: `run_experiment` trains it once per (seed, fold) on the purged training range, selects the checkpoint on the inner-validation slice, and measures its greedy exposures exactly like a baseline.

| Module | Role |
|---|---|
| `rl/features.py` | Per-ticker feature matrix (reuses `scrape_data.compute_indicators` + `functions.FeatureScaler`), scaler fitted on training bars only |
| `rl/env.py` | Vectorised env: exposure k/K of capital (K = 4, long-only), action masking, rewards `diff_sharpe` / `pnl` / `profit` (legacy), random-start episodes |
| `rl/replay.py` | Uniform and prioritised replay (sum tree) storing bar indices, n-step collector |
| `rl/networks.py` | Two-input Q-network, late fusion, `flatten`/`gap` pooling, optional dueling head |
| `rl/agent.py` | Double DQN with masked targets, Huber/MSE, LR schedule, soft/hard target updates |
| `rl/trainer.py` | One training run: collect, learn, inner-validation checkpoint selection, training curve |
| `rl/policy.py` | `DQNPolicy`: parallel (seed, fold) jobs, one GPU per worker, ≤ 90 % of CPU cores |

```bash
python scripts/run_agent.py --config config/experiments/m2_dqn_base.yaml --smoke   # 1 job, timing only, not logged
python scripts/run_agent.py --config config/experiments/m2_dqn_base.yaml           # 10 seeds x 5 folds, logged trial
python scripts/run_agent.py --config config/legacy.yaml                            # old algorithmic choices
python scripts/analyze_m2.py                                                       # M2 report
```

**Switching the reward.** Set `agent.env.reward` in a config (`diff_sharpe`, `pnl`, or `profit` for the legacy reward). M3 adds `mean_variance` and `vol_scaled_pnl`. Make a new YAML that `inherit:`s the base and changes only that key, so the trial log shows exactly one difference.

**Every agent run is a trial.** A new configuration (or a code change) adds one to `N_trials` in the deflated Sharpe ratio, and the budget is 50 (PROTOCOL §5). Use `--smoke` for mechanics checks: it logs nothing and deliberately prints no outer-validation numbers.

## Original code (single asset, kept unchanged)

| File | Role |
|---|---|
| `scrape_data.py` | Download daily OHLCV from Yahoo Finance, add indicators, write a 70/15/15 chronological split |
| `functions.py` | CSV loader, feature transforms, `FeatureScaler` (fit on training data only) |
| `env.py` | Single-asset trading environment: hold / buy 1 / sell 1, long-only, differential Sharpe reward |
| `agent/agent.py` | Two-input Q-network (market window + position vector), replay buffer, Double-DQN update |
| `train.py` | Training loop with greedy validation each episode; saves `<name>_best.keras` and `<name>_last.keras` |
| `evaluate.py` | Greedy evaluation and plot of a saved model |

### Usage (current CLI)

```bash
conda activate dqn_ml

# 1. data: writes train_data/AAPL_{train,val}.csv and test_data/AAPL_test.csv
python scrape_data.py AAPL --start 2015-01-01 --name AAPL

# 2. train (wandb optional; --no-wandb to skip it)
python train.py AAPL_train.csv aapl --val-stock AAPL_val.csv --episodes 50 --no-wandb

# 3. evaluate a checkpoint on the test file; the plot goes to graphs/
python evaluate.py AAPL_test.csv aapl_best --test --max-position 10
```

Known limitations of the current code are listed in [`docs/CODEBASE_NOTES.md`](docs/CODEBASE_NOTES.md). The most important ones:

- The logged "val Sharpe" is the Sharpe of the reward stream, not of returns.
- "Net" P&L ignores open positions at the end of a period.
- `evaluate.py` does not know the `--max-position` used in training, so pass the same value by hand.
- The default feature list requires the `SMA20_Rel`/`SMA50_Rel` columns, which only `DAX_new_*` has. For other files, pass `--features Close Volume ROC12 MFI14 FVolatility`.
- TensorFlow allocates nearly all memory on **every** visible GPU by default. On this shared server, pin one GPU per job: `CUDA_VISIBLE_DEVICES=0 python train.py ...`.
- wandb logging is still active here (disable with `--no-wandb`). The new harness does not use wandb; its record is `experiments/trials.csv`.

## Legacy results (original q-trader, single stock, pre-2026)

These plots come from the original agent. They were made with one seed, no baselines and no cost-aware metrics, so they are **not comparable** with anything produced under `PROTOCOL.md`.

- Google, trained on 5 years, one episode: ![Agent1](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/Google0_googletest.png)
- Google, two episodes: ![Agent1.2](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/Google1_googletest.png)
- Google model on 2024 S&P data: ![Agent1_generalization](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/Google0_SundP_test.png)
- Google model on 2024 MSCI World: ![Agent1_generalization](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/Google0_iSharesMSCIWorld_test.png)
- S&P 2019–2022 model on 2024 S&P: ![Agent3](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/smallSP2_SundPtest.png)
- S&P model on 2024 Google: ![Agent3](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/smallSP2_Google_test.png)
- S&P model on 2024 MSCI World: ![Agent3](https://github.com/leonard-creator/DQN_Trading/blob/main/graphs/smallSP2_iSharesMSCIWorld-test.png)

---

## Change report

Newest entry first. Each entry says what changed, why, and what was verified.

### 2026-10-07 — v2 Step 0d (R0′), Step 0c and V1 implemented, detached queue, shared report code

**Owner instructions (2026-10-07):**
- no optional connectors;
- always look for ways to improve the agent (profit in the later live test is the goal);
- reuse the harness and scripts, with fewer single-use files and short, safe, commented code;
- GPU training runs detached from the session.

**Results:**
- Step 0d: R0′ = 0.43 vs R0 0.44 on the 26 ETFs (paired Δ −0.00, p = 0.33). It is trial 17 (`experiments/reports/V2_0d.md`).
- Calibration of the synthetic worlds: the W-vol oracle gained only +0.08 Sharpe with GARCH β = 0.90. The rule needs ≥ 0.15, so β was raised to 0.91 (+0.19). W-regime passes with the specified parameters (+0.36).

**Added**
- `scripts/run_queue.sh` + `experiments/v2_queue.txt`: a detached job queue.
  - It runs each line one after another, independent of Claude Code or VS Code.
  - Lines can be appended while it runs, and it resumes after the last finished line.
  - A lock prevents two runners on the same queue.
  - Its log is `experiments/v2_queue.log`.
- `harness/synthetic.py`: Step 0c worlds and their oracles (W-null, W-vol, W-regime), the calibration rule, verdicts and stopping rule 1, with the report `experiments/reports/V2_0c.md` and the log `experiments/synthetic.csv` (no trials). Entry point: `scripts/run_agent.py --synthetic` / `--calibrate`.
- Synthetic iteration options in `rl/exogenous.py` (all off by default, so V1 itself is unchanged):
  - `algo.n_step` (hold targets);
  - `algo.heads` + `agent.gate_z` (bootstrapped heads, uncertainty gate);
  - `network.layer_norm`;
  - `env.reward: mean_variance` with `mv_lambda: auto`.

  `scripts/run_agent.py` gains `--set key=value` (YAML overrides, part of the config hash), `--worlds` and `--seeds`. The proposed candidate is `config/v2/V1b.yaml`, not run.
- `rl/exogenous.py` + `config/v2/V1.yaml`: **V1**.
  - Target-exposure actions with the exact cost head Q = U − κ|e′ − p|.
  - Exogenous replay: all 5 exposures learnt from every (ticker, day), with no environment loop.
  - The lazy buy-and-hold anchor (η = 0.01), with buy-and-hold value priors.
  - It is selected by `agent.replay.mode: exogenous`.
- `harness/report.py`: the report helpers that were copied across four scripts, now in one place.
- `scripts/analyze_v2.py`: one generic report for every v2 step, including the §V8 adoption rule.
- Tests: `tests/test_synthetic.py` (5) and `tests/test_exogenous.py` (7). The full suite has **116 passed**.

**Changed**
- `rl/policy.py`: the worker pool is now `train_jobs()`, shared by real and synthetic runs. V1 is selected in `run_job`.
- `rl/networks.py`: `pos_dim = 0` gives a market-only network.
- `rl/diagnostics.py`: the LC5/LC7 rollout for V1.
- `harness/experiment.py`: `run_experiment(prices=…)` accepts injected data (synthetic worlds only). It refuses to log such a run as a trial or to unlock the test period.
- `harness/diagnostics.py`: `strategy_scores` / `agent_scores`, moved there from `scripts/rescore_v2.py`, which is now 155 lines.
- `scripts/analyze_m2.py`, `analyze_m4.py`, `rescore_costs.py`: use `harness/report.py`.
- `PROTOCOL.md` **v2.0.2**: §V12.1 items 17–20 (synthetic implementation, calibration, V1 details, automatic stopping rule). They were written before the runs.

**Verified**
- Regenerated M2/M3/M4/cost-scenario and 0a reports have identical tables. The only differences are deflated-Sharpe values, because N_trials has grown; the committed historical versions were restored.
- R0′'s decisions are reproduced 100 % through the refactored code.
- The V1 smoke run on real data (F1, 5k updates, CPU, 83 s) trades 1.4–1.8 times per year on inner validation. This is a mechanics check only; no outer numbers were looked at.

### 2026-10-06 — v2 Steps 0a (re-scoring) and 0b (data audit, pipeline v2)

**Results** (details: `RESULTS.md` v2 sections, `experiments/reports/V2_0a_rescore.md`, `DATA_AUDIT.md`):
- **0a:**
  - Timing IC lies between −0.001 and +0.016 in all 16 trials, so there is no timing skill. The gap to buy-and-hold is mostly cost (R0: −0.23 = −0.09 timing − 0.14 cost).
  - All agents pass gate G-lag (Δ lag-1 between −0.05 and +0.06).
  - Action gaps are 0.15–0.52 of the TD-noise spread, so decisions are noise-driven, which explains why 13 of 16 agents switch every second day.
  - Q is optimistic by 1.1–3.5 SDs of the realised return.
  - The median best checkpoint comes at 26–47 % of training.
  - New baseline vol_target: 0.14 (^GDAXI) / 0.53 (26 ETFs), below buy-and-hold.
- **0b:**
  - ρ̄ = 0.48, so N_eff ≈ 2. There are 6 drawdowns ≥ 15 %.
  - Luck level for the best of 16 trials over the 9.7 validation years: Sharpe ≈ 0.58.
  - Q1 leak quantified: the residual feature's correlation with the next-day DAX return is −0.15. The affected agents' ^GDAXI timing IC is ≈ 0, so they did not use it.
  - Q2: frozen = fresh data within 0.03 bp.
  - Q3: all 12 moves beyond 8σ are genuine.
  - Q4: ^GDAXI has 9 zero-volume bars.
  - Q5: clean.

**Added**
- `scripts/rescore_v2.py` (Step 0a driver) → `experiments/reports/V2_0a_rescore.md`, CSVs in `experiments/reports/v2_0a/`.
- `harness/diagnostics.py`:
  - timing IC and timing Sharpe (LC3), switches and time at anchor (LC6);
  - exposure- and vol-matched drawdowns (H3 preview);
  - learning-curve statistics (LC1).
- `rl/diagnostics.py`: re-runs the saved checkpoints (CPU) for the action gap (LC5) and predicted vs realised value (LC7), with a reproduction check against the stored exposures.
- `harness/baselines.py::vol_target` + `config/experiments/baseline_vol_target.yaml`: a descriptive, variance-managed baseline, logged as a baseline row (not a trial).
- `scripts/audit_data.py` → `DATA_AUDIT.md`, CSVs in `experiments/reports/v2_0b/`. Offline, development data only; it runs the point-in-time tests itself.
- `config/v2/R0prime.yaml`: R0′ for Step 0d = R0 + pipeline v2 + P = 40. **Prepared, not launched.**
- Tests: `tests/test_v2_diagnostics.py` (6) and 7 new tests in `tests/test_features_m4.py`:
  - the Q1 lag;
  - the Q4 mask;
  - the availability flag and the 450-bar warm-up;
  - expanding z-score;
  - v2 defaults;
  - point-in-time behaviour of the v2 features.

**Changed**
- `harness/backtest.py`, `harness/experiment.py`: execution-lag option (default 0 = v1 behaviour).
- `rl/policy.py`: the data pipeline is factored into `build_job_data`, `eval_market`, `eval_ranges`, `experiment_features` and `make_jobs`, shared by training and re-scoring. Behaviour is identical.
- `rl/features_m4.py`: `pipeline: v2`:
  - Q1 lag of ^GDAXI's residual features;
  - Q4 volume mask;
  - `resid_avail` flag with the 450-bar warm-up;
  - z-score minimum 126 bars.

  The default `v1` is unchanged.
- `harness/config.py`: the `inherit:` depth limit is raised from 5 to 10, because the v2 configs sit 7 levels below `m2_dqn_base`.
- `PROTOCOL.md` Part II **v2.0.1**: implementation clarifications §V12.1 (definitions, G-lag wording, pipeline details, z-score window kept at 252). No rule changed.
- `RESULTS.md` (v2 sections, M4 provenance note quantified), `docs/CODEBASE_NOTES.md` §v2.
- `.gitignore`: only the CSVs under `data/audit/` are ignored, so the Q2 manifests are committed; console logs of report scripts are ignored.

**Verified**
- 100 % of the stored exposures reproduced for all 16 trials (800 runs), so the refactored v1 path is identical.
- Full test suite: **104 passed**.
- R0′ config loads with P = 40. Its pipeline-v2 features are finite on real data. The availability flag starts at bar 450 (for DBC, USO and SLV between 2007-11 and 2008-02).
- No trial added (N_trials = 16). The test period is untouched. No network access: Q2 used the owner's download of 2026-10-06, whose stooq part failed and was declared not needed by the owner.

### 2026-10-06 — v2 programme approved and merged; download script; owner ideas analysed

**Owner decisions:**
- the v2 programme (owner draft `NEW_PROTOCOL.md` v2.2) is approved;
- early stopping removed;
- M6 (live forward test) deferred to future work;
- Track B (H2) deferred to its own protocol, not dropped;
- the Q2 data cross-check is kept;
- **no automatic network access or git pushes** (owner-run downloads only), with **wandb online** as the one approved exception;
- sources added by the owner are owner-verified.

**Changed**
- `PROTOCOL.md`: **Part II — v2 research programme** (v2.0, frozen before any v2 run). It is the owner's v2.2 text with headings/references prefixed V, the decisions above written in, and the owner's two ideas recorded in §V6.1. Part I is unchanged except for its status line and change log.
- `docs/03_extracted_plan.md` §8: summary of v2.
- `docs/evidence-dqn-trading.md`: the owner's 2026-10-06 literature entry plus the further references of Part II §V13, copied unchanged and marked owner-verified.
- `RESULTS.md`: the v2 step table; provenance note in M4 on the ^GDAXI close-time leak (Q1).
- `docs/CODEBASE_NOTES.md` §v2: map of where each v2 component hooks into the current code.
- `.gitignore`: downloaded data under `data/audit/`, `data/longhistory/`, `data/external/` stays out of git; their `MANIFEST.json` files are committed.

**Added**
- `scripts/download_external.py`: owner-run downloads, never called by the code.
  - `q2`: fresh Yahoo bars for all 34 series + unadjusted bars and dividends/splits for 5 spot-check tickers + stooq as a second source (~25 MB);
  - `french`: daily industry portfolios + Fama/French factors (~1–3 MB; the 49-industry set adds ~6 MB);
  - `fred`: BAA10Y (~0.3 MB).
  - `--dry-run` prints the plan and sizes without any network access. It never overwrites, and it writes a sha256 manifest per folder.

**Owner ideas analysed** (measured on stored exposures, no new runs):
1. *Buy more than one unit per decision / based on the remaining capital.* Since M2 the agent already trades **capital fractions** (0–100 % of each ETF's sleeve), not shares. It is limited to one ¼-step per day. In R0 that limit rarely binds:
   - 89 % of its moves are single steps;
   - merging multi-step moves would remove only ≈ 12 % of its transactions.
   - The real problem is jitter: R0 sits at 75 % or 100 % for 88 % of the time and toggles between them ≈ 64 times per ETF per year.
   - Choosing the target exposure directly is part of V1 anyway, together with the cost band that targets exactly this jitter.
2. *Costs per transaction, not per unit.* True for a broker's fixed fee (already modelled in the neo-broker scenario), not for the bid-ask spread, which is paid on every euro traded. V1's cost head gets an exact fixed-fee term for neo-broker variants.
3. *Price-level bias (expensive vs cheap stocks).* Already excluded since M2: exposure is a fraction of capital, P&L is computed from returns, costs are proportional, and every input is scale-free. Fractional shares are assumed.
4. **Deferred, needing their own protocol:** leverage above 100 % (it does not change the Sharpe ratio), a continuous fraction (needs policy-gradient methods), a shared capital pool across ETFs (portfolio allocation).

### 2026-10-05 — Phase 4–5 / M4: cross-asset agent, new features, conv-transformer

**Owner decisions:** go to M4 with `vol_scaled_pnl`; skip mean-variance tuning; build the conv-transformer while the first M4 runs train; 2-ETF deployment set (SPY + EFA) for the EUR 10k neo-broker case; no commit yet.

**Added**
- `rl/features_m4.py`: the spec Phase-4 features, fully causal, with no scaler fitting:
  - log return, P/SMA20 − 1, P/SMA50 − 1, Bollinger %B, RSI(14), MACD histogram/P, log relative volume, ATR/P, 20-bar volatility;
  - **VIX as-of with a one-bar lag** (`vix_lag1`, `vix_chg_lag1`);
  - **residual features** (plan §7 A1): out-of-sample residual return vs the first 3 principal components of the 26 training ETFs (PCA on the 252 days before t, betas on the 60 days before t), plus its 30-day sum;
  - trailing 252-bar z-score, clip ±5.
  - Computed once per experiment and shared by all jobs.
- `rl/networks.py`: `arch: conv_transformer` (plan §7 A2):
  - 2 causal Conv1D layers (8 filters, kernel 2) with per-channel instance norm and a residual connection;
  - learned position embedding;
  - 4-head self-attention + feed-forward with residual LayerNorms;
  - signal = last time step.
  - Two corrections during the build: a per-time-step `LayerNormalization(axis=1)` was replaced by a true per-channel `InstanceNorm`, and a causal attention mask was dropped because it changes nothing for the last-step output. No look-ahead is guaranteed by the window itself, which ends at the decision bar.
- `harness`:
  - `extra_ticker_sets` in experiment configs (here `deploy2` = SPY + EFA), so `protocol.yaml` and its hashes stay unchanged;
  - agent runs also load the context series (^VIX) and the factor-set tickers.
- PROTOCOL §10a.10: the M4 evaluation sets; `deploy2` is secondary.
- `config/experiments/m4_{cross_base,cross_resid,cross_resid_transformer,single_resid}.yaml` (trials 13–16), run from the detached `experiments/runs/m4_launcher.sh`.
- `scripts/analyze_m4.py` → `experiments/reports/M4_cross_asset.md`:
  - all evaluation sets;
  - the go criterion as a paired permutation test;
  - baselines on `deploy2` under neo-broker costs;
  - permutation tests, deflated Sharpe, PBO;
  - training stability.
- Tests (91 pass):
  - `tests/test_features_m4.py`: no look-ahead with every feature on, at once (future prices, volume and VIX perturbed); VIX strictly-before join; residuals remove the common factor and use no information from day t for their model; no NaN after warm-up on all 33 real tickers; end-to-end M4 training.
  - `tests/test_rl.py`: the conv-transformer builds, trains, and its signal depends on the first and the last bar of the window.

**Results** (`RESULTS.md` M4). Median Sharpe on the 26-ETF set at 10 bp:
- conv 0.18 (no residuals), conv 0.16 (residuals), **conv-transformer 0.44**; buy-and-hold 0.67, MACD 0.44.
- **Go criterion met** by the transformer on the 7 leave-out ETFs: 0.32 vs 0.00 for the single-asset agent (p < 0.001). The conv agents are not significant (p ≈ 0.08–0.10).
- The transformer trades half as much (16 vs 31.5 turns/yr) and is invested 84 % of the time. At 0 bp it still trails buy-and-hold (0.58 vs 0.67), so the gain is less overtrading, not timing skill.
- It beats momentum (Holm p = 0.011). Deflated Sharpe 0.69 < 0.95.
- PBO over the 3 cross-asset configs: 0.00.
- Residual features: no measurable effect.
- Cross-asset training is more stable (late drift ≈ 0, best checkpoint at 37–47 % of training).

### 2026-10-05 — Phase 3 / M3: reward designs implemented and compared on ^GDAXI

**Owner decisions:** continue the plan on ^GDAXI (cheaper, cleaner comparison) with the reduced training budget (100k transitions), and implement the reward designs so they can be reused for later ideas.

**Added** (all switched on per config, defaults unchanged, so earlier trials keep their hashes):
- `rl/env.py` rewards, all on the same net return R the harness measures:
  - `mean_variance`: x − (λ/2)x², with x = R / ex-ante daily vol (`mv_lambda`);
  - `vol_scaled_pnl`: R / ex-ante daily vol;
  - `active_return`: (R − buy-and-hold return) / ex-ante daily vol.
- `rl/env.py` training-only shaping:
  - `cost_penalty_mult`: shadow cost; trades look m× as expensive in the reward only;
  - `holding_penalty`: `linear` k·h/252 or `exp` k·(e^{αh/252} − 1).
- **Reward components logged** per step and per checkpoint (`rc_pnl`, `rc_cost`, `rc_shadow`, `rc_risk`, `rc_hold`, `reward_raw`), as the spec's Phase 3 requires.
- `rl/features.py::ex_ante_vol`: EWMA (span 60) daily volatility known at each bar; `MarketData(sigmas=...)`.
- `config/experiments/m3_{mean_variance,vol_scaled_pnl,active_return,diff_sharpe_shadow3}.yaml`, fixed before any M3 result (trials 9–12). They ran from the detached `experiments/runs/m3_launcher.sh`.
- `scripts/analyze_m2.py`: reusable via `--milestone M3` / `--configs`, with a reward-components section. Fixed a Python 3.11 f-string error the first rerun had silently hidden; the regression check of the M2 report then showed only the expected differences (N_trials and the generic PBO label). The M2 report file was kept as its milestone snapshot (N_trials = 5).
- `tests/test_rewards.py` (13 tests):
  - each reward against a hand calculation;
  - shadow cost and holding penalty change the reward but never the measured return;
  - env = harness with shaping on;
  - holding penalty forms;
  - mean-variance risk term logged;
  - bad configs fail loudly;
  - no look-ahead in `ex_ante_vol`;
  - end-to-end training with vol-scaled rewards.
  - Total: **84 tests pass**.

**Results** (`RESULTS.md` M3, `experiments/reports/M3_rewards.md`). Median Sharpe on ^GDAXI at 10 bp:
- reference 0.17; `mean_variance` −0.01; `vol_scaled_pnl` **0.21**; `active_return` 0.13; shadow-cost ×3 0.17; buy-and-hold 0.39.
- **Kill criterion not met.**
- At 0 bp the reference and `vol_scaled_pnl` equal buy-and-hold (0.39), so the gap is cost drag.
- PBO over the five M3 configs is 0.59: the ranking between rewards is not reliable.
- λ = 0.5 in volatility units makes `mean_variance` stay mostly flat; λ ≈ 0.05 would balance the terms.
- The shadow cost cuts turnover by 25 % and doubles holding time, without changing the Sharpe ratio.

### 2026-10-05 — Neo-broker cost scenario, M3 reference run, session hygiene

**Requested by the owner:** a real-world scenario with EUR 1 for every transaction (each buy and each sell) plus ETF running costs (TER), including training under it. The M3 training budget is halved to 100k transitions.

**Added**
- `config/cost_scenarios.yaml`:
  - named scenario `neo_broker`: EUR 1 per transaction on EUR 10,000, 3 bp half-spread, TER 15 bp/yr on held ^GDAXI exposure;
  - sensitivity variants at EUR 1k / 50k capital and 1 / 5 bp half-spread.
  - The file documents all estimates. It explains why TER is charged only on the index series: ETF prices are already net of their TER, so charging it there would double-count.
- `harness/backtest.py`: `CostModel` (proportional + fixed fee per transaction + holding cost per bar), `load_scenarios()`, `scenario_cost()`.
  - The fee is a fraction of the capital behind each position (capital / number of tickers).
  - A plain float still means the PROTOCOL bp level, so M1/M2 numbers are unchanged.
- `harness/experiment.py`:
  - `run_experiment(..., cost_scenarios=...)` reports scenarios next to the bp sweep (`cost` column, `returns_<set>_scen-<name>.csv`);
  - the code hash is taken at run start.
- `rl/env.py`, `rl/trainer.py`, `rl/policy.py`:
  - per-ticker cost models in the training reward and the checkpoint score (`agent.env.cost_scenario`);
  - optional `agent.env.levels` (K);
  - `StoredExposurePolicy` re-scores a finished trial without retraining (not a new trial).
- `harness/metrics.py`: `trades_per_year`.
- `scripts/rescore_costs.py` → `experiments/reports/cost_scenarios.md`: all trials and baselines under all scenarios, plus permutation tests and deflated Sharpe under `neo_broker`.
- Configs `m3_dqn_base_100k` (10 bp reference, 100k transitions), `nb_dqn_base` (trained under neo-broker costs) and `nb_dqn_k2` (the same with K = 2). Trials 6–8 of 50.
- `tests/test_costs.py` (8 tests):
  - one fee for every buy and every sell, two per round trip, none on hold;
  - per-sleeve scaling;
  - TER only while invested;
  - env = harness with fee and TER;
  - scenario reporting and round-trip loading;
  - end-to-end scenario training.
  - Total: **71 tests pass**.
- `PROTOCOL.md` §10a.9: scenarios are secondary; H1 stays at 10 bp + 1 bp.

**Changed**
- `harness/config.py`: `code_hash` covers only code that can change results (`harness/`, `rl/`, `functions.py`, `scrape_data.py`), not analysis scripts.
- The two neo-broker trainings ran from a detached launcher (`experiments/runs/nb_launcher.sh`, `setsid nohup`), so they survive a VS Code disconnect.
- A duplicate Claude session (tilingl-14) that had been editing the repo in parallel was stopped at the owner's request. Its M2 write-up in `RESULTS.md` was correct and is kept.

**Results** (`RESULTS.md`, last two sections). Median Sharpe on ^GDAXI:
- Halving training: 0.09 → **0.17** at 10 bp. Shorter training helps, consistent with overfitting.
- Neo-broker, EUR 10k:
  - re-scored 10 bp agents gain ≈ +0.1 (reference 0.26);
  - **training under the scenario gives no improvement** (0.13; K = 2: 0.17), with trading activity unchanged at ≈ 110–120 transactions/yr.
  - Buy-and-hold stays at 0.38; no agent's median beats it.
- At EUR 1k the fee makes every active agent strongly negative. For a 26-ETF portfolio on EUR 10k it ruins every active baseline.
- Legacy beats buy-and-hold on the mean difference under this scenario (Holm p = 0.033), but its median is lower and its deflated Sharpe is 0.72. Not robust; documented, not used as evidence.

### 2026-10-05 — Knowledge base: "Deep Learning Statistical Arbitrage" + strategy addendum

**Requested by the owner.** Guijarro-Ordonez, Pelger & Zanotti, *Management Science* 72(9):7502–7549 (Sept 2026 issue), https://doi.org/10.1287/mnsc.2022.03132.
- `docs/evidence-dqn-trading.md`: new dated entry.
  - Metadata comes from Crossref. Methods, results and cost numbers come from the open arXiv v2 (2022), because the published full text was not accessible (INFORMS 403). Everything not verified is listed.
  - Main lesson: the paper's edge comes from trading **factor residuals** (relative value) with a **convolutional-transformer** time-series encoder. Trading raw return levels did much worse (Sharpe 1.64 vs ≈ 4). Signal extraction mattered far more than the allocation function.
- `docs/03_extracted_plan.md` §7 (addendum): **H1 and the PROTOCOL are unchanged.**
  - Track A, inside the protocol and decided before any results: residual features (A1) and a conv-transformer encoder ablation (A2) for M4, plus an exposure-timing diagnostic (A3).
  - Track B, a contingency: a market-neutral residual hypothesis **H2**. It must be pre-registered **before M5** and tested in the same single test-period run, with Holm correction across H1 and H2. **Needs the owner's decision.**
- The original `../project_mds/evidence-dqn-trading.md` was **not** edited (outside this folder); only the copy in `docs/` was updated.
- `scripts/analyze_m2.py`: added a noise-robust collapse measure ("late drift"). The best-checkpoint comparison overstates collapse, because the maximum of ~20 one-year Sharpe estimates is biased upward.

### 2026-10-05 — Phase 2 / M2: agent rebuilt on the harness (single asset ^GDAXI)

**Decisions by the project owner:** test period split into three 12-month sub-blocks (PROTOCOL v1.2); `.gitignore` for raw data, run outputs and caches; wandb logging stays online.

**Added: `rl/` package** (the original `env.py`, `agent/`, `train.py` are unchanged)
- `rl/env.py`: vectorised environment. B = 32 episodes step together, so one network call serves 32 decisions; the old code made one `predict_on_batch` call per step.
  - Positions are capital fractions k/K, K = 4, long-only (PROTOCOL decision 3).
  - The per-step net return uses **exactly the harness backtest formula** (tested).
  - Invalid actions are masked: no buy at max long, no sell when flat.
  - The state is one-hot position + volatility-normalised unrealised P&L + holding time (spec Phase 3).
  - Episodes are one year (252 bars) with a random start inside the training range. The old code walked the same path every episode, which invites memorisation.
- `rl/replay.py`: the replay buffer stores bar indices, not window copies (~280× smaller per transition, multi-asset ready). Also proportional PER on a sum tree and an n-step collector.
- `rl/agent.py`, `rl/networks.py`:
  - Double DQN with masking at ε-greedy **and** inside the target argmax/max.
  - Huber loss on running-std-normalised rewards. This also fixes the old issue where Huber was compiled but MSE was used.
  - Cosine LR decay, global-norm gradient clipping, soft target updates (τ = 0.005) or hard copies.
  - Flatten or GAP pooling, optional dueling head.
- `rl/trainer.py`: checkpoint selection on the **inner-validation slice only** (the last 252 bars of each fold's training range, purged), scored with the harness backtest at 10 bp. A training curve is logged per run.
- `rl/policy.py`: the harness policy. It trains all (seed, fold) jobs in parallel worker processes:
  - one GPU per worker, with memory growth on;
  - ≤ 90 % of CPU cores;
  - TF op determinism on;
  - one wandb run per job, grouped by configuration.
- `config/experiments/m2_dqn_base.yaml` + three one-change ablations (`_dueling`, `_nstep`, `_per`), and `config/legacy.yaml`: MSE, GAP, profit reward, no masking, full-series episodes, hard target, constant LR 1e-3. All five were fixed **before** any agent result was seen; they use 5 of the 50-trial budget.
- `scripts/run_agent.py` (with `--smoke`: timing only, nothing logged, no outer-validation output), `scripts/analyze_m2.py`.
- `harness`:
  - configs can `inherit:` another config;
  - execution-only keys (`runtime`, `logging`) are excluded from the config hash;
  - `run_experiment` calls an optional `policy.prepare()` and creates the run folder up front;
  - `load_result()` rebuilds a stored run;
  - `Fold.train_cut` and `final_test_blocks()` implement the test sub-blocks.
- `tests/test_rl.py` (16 tests): masking at acting and in the target, Double-DQN target vs a manual computation (with an invalid unmasked argmax), soft update, PER sampling proportions and weights, sum tree, n-step sums and flush discounts, env return = harness backtest, feature look-ahead, no NaN features on real data, end-to-end determinism. Total: **63 tests pass**. Tests run on CPU.

**Verified before the sweep:** a GPU smoke run took 31 s for 20k transitions, so ~5 min per full (seed, fold) job and ~25 min per configuration with 12 workers on 4 GPUs.

**Sweep (14:05–17:14):** 5 configurations × 50 runs. In practice ~37 min per configuration, because three jobs share each GPU. All runs finished, 0 failures, and the 90 % CPU cap was respected (load ≤ 13 of 40 cores).

**Results** (`RESULTS.md` M2, `experiments/reports/M2_agent.md`). Median Sharpe on ^GDAXI at 10 bp:

| Config | Median Sharpe |
|---|---|
| legacy | 0.33 |
| base | 0.09 |
| dueling | 0.05 |
| n-step | 0.05 |
| PER | 0.05 |
| *buy-and-hold* | *0.39* |

- **No collapse:** Q-values stay bounded. The inner-validation Sharpe drifts down mildly (−0.1 to −0.3) after ~30 % of training, both with the legacy hard target and with the soft target + LR decay. That points to overfitting, not instability.
- **Why the fixes underperform:** the new agents trade ~30× per year (legacy 7×) with no timing skill. Even at 0 bp they stay below buy-and-hold.
- **Statistics:** no configuration beats buy-and-hold (legacy +0.06, Holm p = 0.10). All deflated Sharpe ratios are below 0.95. PBO over the 5 configurations is 0.02.

**Fixed while analysing:** the first collapse measure (best checkpoint vs last) flagged 35/50 runs. Most of that is the upward bias of the maximum of ~20 noisy one-year Sharpe estimates, so a noise-robust "late drift" measure was added and is reported instead.

**Changes made in the owner's parallel session (tilingl-91), 16:49–16:57, documented here for completeness:**
- secondary cost scenarios: `config/cost_scenarios.yaml`, e.g. `neo_broker`: EUR 1 per transaction, 3 bp half-spread, TER on ^GDAXI;
- in `harness/backtest.py`: `CostModel` with fixed fee + holding cost; a plain float rate keeps the old behaviour;
- PROTOCOL §10a.9;
- M3 configs `m3_dqn_base_100k`, `nb_dqn_base`, `nb_dqn_k2` (training budget halved to 100k);
- `tests/test_costs.py`;
- the code hash is now taken at run start over `harness/`, `rl/`, `functions.py`, `scrape_data.py` only.

All 71 tests pass on the combined code.

### 2026-10-05 — Phase 1 / M1: evaluation harness and baselines

**Decisions by the project owner:** PROTOCOL v1.0 approved (33-ticker daily universe, test period 2023-10-02 → 2026-09-30, capital-fraction long-only sizing with K = 4). Packages may be installed into `dqn_ml`. The owner commits to git.

**Environment**
- Installed into `dqn_ml`: `tensorflow[and-cuda]==2.15.1` (CUDA 12.2 / cuDNN 8.9 as pip wheels), `pytest`, `scipy`, and `protobuf` 7.35 → 4.25. **TensorFlow now uses the 4 V100 GPUs** (verified with a GPU matmul); the protobuf startup errors are gone; wandb 0.28 still works.
- `requirements.txt` now matches the tested env (it previously asked for TF ≥ 2.16 / NumPy ≥ 2, which is not what runs).

**Added**
- `config/protocol.yaml`: every protocol number in one machine-readable file.
- `config/experiments/baseline_*.yaml`: one config per baseline.
- `harness/` (9 modules, see "Evaluation harness" above): walk-forward splits with purge and embargo, one shared cost model, baselines, metrics, deflated Sharpe, PBO via CSCV, permutation test + Holm, append-only trial log, test-period guard, `run_experiment()`.
- `scripts/download_universe.py`: downloads the universe once into `data/raw/` with a sha256 manifest and never overwrites without `--force`.
- `scripts/run_baselines.py`: M1 run, reproducibility check, report.
- `scripts/final_test.py`: the only way into the test period; refuses baseline-only configs and second runs.
- `tests/` (46 tests) + `pytest.ini`.
- `RESULTS.md`, `experiments/trials.csv` (7 baseline rows), `experiments/reports/M1_baselines.md`.
- `data/raw/`: 34 daily series 2000 → 2026-09-30 (20 MB). Quality check: no unadjusted splits; the largest moves are real events (2008-10, 2020-03); gaps are real closures.

**Changed**
- `PROTOCOL.md`: approved (v1.0), then v1.1 with implementation clarifications (§10a). One item stays open until M5: how to pair seeds within the single test block.
- `docs/CODEBASE_NOTES.md`: environment fix noted.

**Bugs found and fixed while testing**
- A constant return series has a floating-point std of ~1e-19, not 0, which gave absurd Sharpe/kurtosis values. Std below 1e-12 is now treated as 0 (`harness/metrics.py`, `harness/stats.py`).
- `harness.splits.test_fold` was being collected by pytest as a test; renamed to `final_test_fold`.

**Verified**
- `pytest`: 46 passed. Covered: purge for 4 (W, H) settings, no look-ahead in momentum/MACD under perturbed future prices, decision → return timing, cost charging, guard and lock, trial counting, PBO calibration on noise (mean 0.53), deflated Sharpe vs the closed-form normal case, Holm against a hand calculation, a policy never seeing data past its block, reproducibility.
- M1 go criterion: all 7 baseline configurations give bit-identical metrics on a re-run.

**Main result:** at 10 bp, **buy-and-hold has the highest median Sharpe (0.67)** on the training universe, ahead of MACD (0.44), momentum (0.28) and random (−0.25). Details and interpretation in `RESULTS.md`.

**Not changed:** `env.py`, `agent/agent.py`, `train.py`, `evaluate.py`, `functions.py`, `scrape_data.py`. The original single-asset pipeline still works as before. The harness only reads `data/raw/`.

### 2026-10-05 — Step 0: orientation (no code changes)

**Added**
- `docs/` with copies of the plan, the engineering brief and the evidence log (originals in `../project_mds/` are untouched).
- `docs/CODEBASE_NOTES.md`: repository map, spec facts checked against the code, gap analysis vs W1–W15 and phases 1–6, environment audit.
- `PROTOCOL.md`: pre-registration **draft** with a 33-ticker daily ETF universe, a fixed leave-assets-out set, 5 anchored walk-forward folds with purge/embargo, a frozen test period, baselines, cost model and statistics.

**Changed**
- `README.md`: status and document index, usage corrected to the current CLI (the old commands no longer worked), legacy results labelled as such, this change report.

**Main findings** (details and evidence in `docs/CODEBASE_NOTES.md`)
1. **No evaluation protocol** (single seed, single validation block, no baselines). This is the largest gap and the reason Phase 1 comes first.
2. **Effective loss is still MSE.** Huber is set in `compile()`, but the custom `tf.function` training step computes squared error itself.
3. **Logged validation numbers do not show skill.** Buy-and-hold over the same DAX validation file earned €37k–56k vs €10k–37k for the agent; a one-episode smoke run already reached €22.7k. Train P&L rises while validation decays after episode ~20–75, an overfitting signature. The "collapse at ~800 episodes" could not be checked: every logged run stopped at 200.
4. **Costs and P&L are in currency units per share/point**, so they are not comparable across assets (a flat €2 cost is ~0.01 bp on the DAX and ~130 bp on AAPL).
5. **Data splits differ per ticker** and have no purge; validation/test windows start zero-padded.
6. **TensorFlow cannot use the GPUs** (env has CUDA 11.8, TF 2.15 needs 12.2); `pytest` and `scipy` are not installed; protobuf 7 vs TF 2.15 causes startup errors.

**Verified**
- Current code trains end-to-end on CPU: 1 episode on `DAX_new` in 25 s (~7 ms/step). The smoke-run output files were deleted afterwards.
- Yahoo daily data is reachable for all 33 proposed tickers back to ≤ 2006-04.

**Open decisions:** see `PROTOCOL.md` ⚑ markers and `docs/CODEBASE_NOTES.md` §8.
