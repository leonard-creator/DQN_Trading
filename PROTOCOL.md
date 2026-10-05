# PROTOCOL — Risk-aware cross-asset DQN (pre-registration)

> **Status: APPROVED v1.0 (2026-10-05), current v1.2 (clarifications, §10). Frozen.** Approved before any training run under this protocol.
> Decisions taken: universe as listed (§2), frozen test period **2023-10-02 → 2026-09-30** (§4.1), position sizing **capital fraction, long-only, K = 4** (§5).
> Any later change gets a dated entry in §10 and counts against the trial budget where relevant.
>
> Background: `docs/03_extracted_plan.md` §3 (evaluation protocol), `docs/CODEBASE_NOTES.md` (current state).

---

## 1. Hypothesis and success criteria

**H1.** A cross-asset DQN with a risk-adjusted reward has a higher **annualised net-of-cost Sharpe ratio** than **buy-and-hold** and than **time-series momentum**, over 10 seeds × 5 walk-forward folds.

**Success** requires all three of:

1. Deflated Sharpe Ratio (Bailey & López de Prado 2014) of the selected configuration **> 0.95**, with `N_trials` taken from `experiments/trials.csv`.
2. PBO via CSCV (Bailey et al. 2015) across all logged configurations **< 0.5**.
3. Holm-adjusted p **< 0.05** in a paired permutation test against **each** of the four baselines in §6.

The confirmatory test (milestone M5) is run **once** on the frozen test period, at the primary cost level (§7).

A failed H1 is a valid outcome and will be reported as such. The protocol will not be loosened to get a positive result.

---

## 2. Universe and selection rule

**Rule.** A pre-declared list of broad, liquid exchange-traded instruments. Choose **one or more per asset-class bucket**, by asset class and inception date (≤ 2006-04, so every ticker has ≥ 252 bars of warm-up before the common start), **not by past performance**. Single stocks are excluded, so no ticker was picked because it later became a winner (W5; Brown et al. 1992). Broad index ETFs do occasionally close, but none of these has. That residual survivorship risk is accepted and stated here.

**Universe (33 tickers, all with Yahoo daily data 2000/2006 → 2026-09-30, checked 2026-10-05):**

| Bucket | Tickers |
|---|---|
| US equity | SPY, QQQ, IWM, DIA, MDY |
| US sectors | XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY |
| International equity | EFA, EEM, EWJ, EWG, EWU, EWZ, EWA, EWC, FXI, ^GDAXI |
| Bonds | TLT, IEF, LQD, TIP |
| Real estate | IYR |
| Commodities | GLD, SLV, USO, DBC |

**Context series (feature only, never traded):** ^VIX, joined as-of with a **one-bar lag** (W8).

**Leave-assets-out set (fixed now, before any training):** `QQQ, XLE, XLF, EEM, EWZ, IEF, SLV` (7 of 33 = 21 %).
Drawn by `numpy.random.default_rng(20261005)`, stratified per bucket (1/2/2/1/0/1). ^GDAXI is excluded from the draw because it is the single-asset reference. These tickers are **never** used for training or hyper-parameter selection. They are used only for the M4 go/no-go (cross-asset vs single-asset) and are reported at M5.

**Single-asset reference (M2):** ^GDAXI (the DAX, which most past runs used).

**Decision 1 (approved 2026-10-05):** this universe, unchanged.

---

## 3. Data source, frequency and date range

- **Source:** Yahoo Finance via `yfinance`, split/dividend-adjusted (`auto_adjust=True`), stored locally as one CSV/Parquet file per ticker with its **full** history. Splits are made by date at load time, never by writing separate split files. Free source, no API key.
- **Frequency:** **daily bars.** Free sources cannot supply long intraday history (yfinance 1h ≈ last 730 days, W7), so H4/1h is out of scope. Time-of-day features are therefore not used.
- **Download range:** first available date (≥ 2000-01-03) → 2026-09-30.
- **Usable common range:** **2007-07-02 → 2026-09-30** (≈ 19.25 years). Earlier bars are used only as warm-up for the causal features.
- **Calendar:** each ticker trades on its own exchange calendar. Bars are aligned by date and a ticker's missing dates are **not** forward-filled into tradable bars.

---

## 4. Splits

### 4.1 Frozen test period

**Decision 2 (approved 2026-10-05): option A.**

| | Period |
|---|---|
| **Frozen test** | **2023-10-02 → 2026-09-30** (36 months, ≈ 15.6 % of the timeline) |
| Development | 2007-07-02 → 2023-09-29 |

(Rejected alternative: an 18-month test from 2025-04-01, which would have given more development data but a weaker test.)

**Contamination statement.** Part of 2024–2026 has already been looked at: legacy single-asset models were plotted on DAX, AAPL, MSCI World, S&P and SAP test files from that period (`graphs/`). No configuration of the *new* system has been evaluated on it, and the lock in §9 prevents that before M5. The prior exposure was to a different model family. It is disclosed here and will be repeated in `RESULTS.md`.

### 4.2 Walk-forward folds (development period)

Expanding training window, two-year validation blocks, identical cut dates for every ticker:

| Fold | Train | Validation |
|---|---|---|
| F1 | 2007-07-02 → 2013-12-31 | 2014-01-02 → 2015-12-31 |
| F2 | 2007-07-02 → 2015-12-31 | 2016-01-04 → 2017-12-29 |
| F3 | 2007-07-02 → 2017-12-29 | 2018-01-02 → 2019-12-31 |
| F4 | 2007-07-02 → 2019-12-31 | 2020-01-02 → 2021-12-31 |
| F5 | 2007-07-02 → 2021-12-31 | 2022-01-03 → 2023-09-29 |

- **Purge:** at every train→validation and development→test boundary, the last **P = W + H** bars before the boundary are removed from training. W is the observation window and H the longest reward horizon (n-step). With W = 20 and H ≤ 5, P = 25 bars.
- **Embargo:** **E = 10 bars** after every evaluation block are excluded from any training set that follows it (configurable). In anchored walk-forward no training data follows a validation block, so E only binds in the CSCV/PBO variant and in any non-anchored ablation.
- **Inner validation for checkpoint selection:** the last 252 bars of each fold's training range (purged on both sides). Checkpoints and early stopping are chosen on this inner slice **only**. The outer validation block is used for reporting, never for picking a checkpoint.

---

## 5. Agent, seeds and trial budget

- **Seeds:** 0–9 for every configuration (10 seeds × 5 folds = 50 runs per configuration).
- **Trial budget:** **at most 50 configurations** in total, counted from the first run under this protocol. Every run is appended to `experiments/trials.csv` (config hash, seeds, fold metrics, timestamp), including failed and aborted ones.
- **Position sizing:**

**Decision 3 (approved 2026-10-05):** exposure is a **fraction of capital, long-only**: k/K with **K = 4**, so exposure ∈ {0, 25, 50, 75, 100 %}. Actions are hold / buy (+¼) / sell (−¼). Short selling (k ∈ {−K … K}) exists only behind a flag, for ablation.
(Rejected alternatives: a long/short target position {−1, 0, +1}; today's one-share-per-unit sizing, which is not comparable across assets.)

---

## 6. Baselines (same exposure grid, same cost model, same periods)

1. **Buy-and-hold:** 100 % long from the first bar of each block.
2. **Random agent:** uniformly random actions in the same action space, seeds 0–9.
3. **Momentum:** 100 % long if the past **60-bar** return > 0, else flat (short instead of flat if shorting is enabled). Other lookbacks (20, 120, 252) are reported descriptively only, **not** tested.
4. **MACD crossover (12, 26, 9):** long if MACD > signal line, else flat.

---

## 7. Cost model

- Cost per rebalance = (**c** + **s/2**) × |Δ exposure| × equity, with c in bp and half-spread s/2 = **1 bp** for all instruments.
- **Primary cost level for H1: c = 10 bp.**
- Reported sweep: c ∈ {0, 5, 10, 25} bp.

---

## 8. Metrics and statistics

- **Per run (daily net returns, annualised with 252):** net return, Sharpe, Sortino, max drawdown, Calmar, turnover, average holding period, hit rate, exposure.
- **Aggregation:** median and IQR over seeds × folds, plus the full distribution (Henderson et al. 2018; Grądzki 2026).
- **Deflated Sharpe Ratio:** on the selected configuration's out-of-sample daily net returns, with `N_trials` from the trial log and the sample skewness and kurtosis.
- **PBO:** CSCV with S = 16 blocks over the configurations × days matrix of concatenated validation returns.
- **Comparisons:** paired sign-flip permutation test (10,000 permutations) on the per-(seed, fold) difference in annualised net Sharpe, agent vs each baseline. Holm correction over the 4 baselines.
- **Names:** the reward is `diff_sharpe`; the metric is `deflated_sharpe`. "DSR" is never used alone (W2).

---

## 9. Test-period lock

The frozen test period can be evaluated only by `scripts/final_test.py --i-am-sure`. That script writes `experiments/FINAL_TEST.lock` (timestamp, config hash, git commit). A second run refuses to start. All other code paths raise an error if asked for data inside the test period.

---

## 10a. Implementation details (v1.1 — clarifications, no change of substance)

The approved text left the points below open. They were fixed while building the harness, **before any agent run**, and are implemented and tested in `harness/`.

1. **Block timing.** An evaluation block [start, end] is the set of daily *returns* dated inside it. The first decision is taken at the close of the last bar before `start`. Every block starts flat, so the entry trade is charged, and open positions are marked to market at the end, not liquidated. Adjacent folds therefore chain without losing a day.
2. **Portfolio.** Each (seed, fold) value is the metric of an equal-weight portfolio: one fixed 1/N sleeve per ticker of the evaluation set. On a date where a ticker has no bar (exchange holiday) its sleeve earns 0. Cross-sleeve rebalancing costs are ignored, identically for all strategies.
3. **Deflated Sharpe of a configuration.** For each seed, the five validation blocks' daily portfolio returns are concatenated and the deflated Sharpe is computed on that series. The configuration's value is the **median over seeds**. `N_trials` = number of distinct (config hash, code hash) agent trials in `experiments/trials.csv`. The variance is that of their per-period Sharpe ratios.
4. **What counts as a trial.** Every `run_experiment` call is logged. Agent runs count as trials, once per distinct (config, code) pair. Baseline runs are logged but **not** counted, because the baselines are fixed in §6 and never selected among.
5. **PBO.** CSCV matrix columns = agent configurations; values = seed-averaged daily portfolio returns over the concatenated validation blocks. A split counts as overfit when the in-sample winner's out-of-sample rank is at or below the median (logit ≤ 0). Calibration check: on pure noise the mean PBO is 0.53 (N = 20), as expected.
6. **Permutation test.** Pairs are (seed, fold). The statistic is the mean difference in annualised net Sharpe; the test is one-sided (agent > baseline). Deterministic baselines take the same value for every seed of a fold.
7. **H1 evaluation set.** Primary = the 26 training-universe tickers. The 7 leave-out tickers and ^GDAXI alone are reported as secondary results.

8. **Test-period pairing (v1.2, approved by the owner 2026-10-05).** For the H1 statistics at M5, the test period is split into three consecutive 12-month sub-blocks: **T1 2023-10-02 → 2024-09-30, T2 2024-10-01 → 2025-09-30, T3 2025-10-01 → 2026-09-30**. That gives 10 seeds × 3 sub-blocks = 30 pairs. All three share **one training cut** at the test start (purged by P bars): the agent is trained once on development data and is never retrained on T1 before T2, so no test data enters training. Each sub-block starts flat, like a validation fold. The full-period result (the three sub-blocks' daily returns concatenated) is reported next to it.

9. **Secondary cost scenarios (2026-10-05, owner request).** Named scenarios in `config/cost_scenarios.yaml` are reported **next to** the cost sweep of §7. The first is `neo_broker`: EUR 1 for every transaction (each buy and each sell) on EUR 10,000, 3 bp half-spread, and a TER on held exposure where the price series is a fee-free index. Agents may also be **trained** under a scenario; each such configuration is a trial. Scenarios never replace the primary 10 bp level, and **H1 is evaluated only at the primary level**.

## 10. Change log of this protocol

| Date | Version | Change |
|---|---|---|
| 2026-10-05 | v0.1 | First draft (Step 0). |
| 2026-10-05 | v1.0 | Approved by the project owner: universe as drafted, test period option A, position sizing capital fraction long-only K = 4. Frozen. |
| 2026-10-05 | v1.1 | §10a: implementation clarifications fixed while building the harness (before any agent run). One open item for M5 listed there. |
| 2026-10-05 | v1.2 | Owner approved the open item: test period split into three 12-month sub-blocks with one shared training cut (§10a.8). Still before any agent run. |
| 2026-10-05 | v1.2 (addendum) | §10a.9: secondary cost scenarios (neo-broker). H1, its cost level and every existing config hash unchanged; scenarios live in their own file. |
