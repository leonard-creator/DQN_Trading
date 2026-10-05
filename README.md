# Deep Q-Trader

This project began from the q-trader by [edwardhdlu](https://github.com/edwardhdlu/q-trader/tree/master). It is being turned into a **risk-aware, cross-asset DQN** together with an **evaluation harness that is honest about overfitting**.

The research question: *does the agent beat simple baselines net of costs, over many random seeds, on data it never saw during development?* A well-documented negative answer counts as a valid result.

> Research code only. No live trading, not investment advice.

## Project status (2026-10-05)

**M1 (evaluation harness + baselines) is done.** [`PROTOCOL.md`](PROTOCOL.md) v1.1 is approved and frozen. Next: M2, model fixes on the single asset ^GDAXI.

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
python -m pytest                           # 46 tests: purge, look-ahead, cost timing, statistics, guard
python scripts/run_baselines.py            # M1: baselines over 5 folds x 10 seeds x cost sweep
```

From Python:

```python
from harness.experiment import run_experiment
res = run_experiment("config/experiments/baseline_momentum.yaml", seeds=range(10))
res.summary["train@10bp"]["sharpe_median"]
```

**Adding an asset.** Add the ticker to a bucket in `config/protocol.yaml` **and** log the change in `PROTOCOL.md` §10, because it changes the universe and therefore every config hash. Then run `python scripts/download_universe.py`; only the new ticker is downloaded.

**Why the test period is locked.** Every look at 2023-10 → 2026-09 that influences a design decision turns it into development data, and the final result would then be optimistic in a way no statistic can correct (PROTOCOL §9; Arnott, Harvey & Markowitz 2018). So `harness.data.load_prices()` cuts every series before 2023-10-02. The only way past that is `scripts/final_test.py --i-am-sure`, which first writes `experiments/FINAL_TEST.lock` and refuses to run a second time.

## Original code (single asset, kept unchanged for now)

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
