# Deep Q-Trader

This project began from the q-trader by [edwardhdlu](https://github.com/edwardhdlu/q-trader/tree/master). It is being turned into a **risk-aware, cross-asset DQN** together with an **evaluation harness that is honest about overfitting**.

The research question: *does the agent beat simple baselines net of costs, over many random seeds, on data it never saw during development?* A well-documented negative answer counts as a valid result.

> Research code only. No live trading, not investment advice.

## Project status (2026-10-05)

**M1 (harness + baselines) and M2 (model fixes on ^GDAXI) are done.** [`PROTOCOL.md`](PROTOCOL.md) v1.2 is approved and frozen. M2 result: training is stable, but no agent beats buy-and-hold net of costs, and the old (legacy) choices beat the Phase-2 fixes (see [`RESULTS.md`](RESULTS.md)). A secondary **neo-broker cost scenario** (EUR 1 per transaction) is implemented and evaluated: cheaper trades help a single ETF on EUR 10k slightly, but the agent gains no timing skill from them. Next: M3, reward variants, on the halved training budget (100k transitions).

| Document | What it is |
|---|---|
| [`PROTOCOL.md`](PROTOCOL.md) | Pre-registration (approved): hypothesis H1, universe, splits, trial budget, costs, statistics |
| [`RESULTS.md`](RESULTS.md) | Results per milestone; currently the M1 baselines |
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
