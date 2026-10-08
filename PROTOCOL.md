# PROTOCOL — Risk-aware cross-asset DQN (pre-registration)

> **Status: APPROVED v1.0 (2026-10-05), current v1.2 (clarifications, §10). Frozen.** Approved before any training run under this protocol.
> **Part II (v2 research programme, approved 2026-10-06) follows at the end of this file.** It extends Part I and changes none of H1, the universe, the folds, the test lock or the cost model.
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
10. **M4 evaluation sets (2026-10-05, owner decision).** Cross-asset agents train on the 26 training-universe tickers. They are evaluated on (a) the same 26 tickers in the validation blocks, which is **H1's primary set** as in item 7, (b) the 7 leave-out tickers, used for the M4 go criterion "cross-asset ≥ single-asset" via a paired permutation test against a single-asset (^GDAXI) agent with identical features, reward and budget, (c) ^GDAXI alone, and (d) a **secondary** 2-ETF deployment set `deploy2` = SPY + EFA, chosen as a realistic EUR 10k neo-broker portfolio (together ≈ MSCI World). `deploy2` is defined in the experiment configs (`extra_ticker_sets`), not in `config/protocol.yaml`; it never enters H1.

## 10. Change log of this protocol

| Date | Version | Change |
|---|---|---|
| 2026-10-05 | v0.1 | First draft (Step 0). |
| 2026-10-05 | v1.0 | Approved by the project owner: universe as drafted, test period option A, position sizing capital fraction long-only K = 4. Frozen. |
| 2026-10-05 | v1.1 | §10a: implementation clarifications fixed while building the harness (before any agent run). One open item for M5 listed there. |
| 2026-10-05 | v1.2 | Owner approved the open item: test period split into three 12-month sub-blocks with one shared training cut (§10a.8). Still before any agent run. |
| 2026-10-05 | v1.2 (addendum) | §10a.9: secondary cost scenarios (neo-broker). H1, its cost level and every existing config hash unchanged; scenarios live in their own file. |
| 2026-10-05 | v1.2 (addendum) | §10a.10: M4 evaluation sets, incl. the secondary 2-ETF deployment set; fixed before any M4 result. H1 unchanged. |
| 2026-10-06 | v2.0 | **Part II added**: the v2 research programme, approved by the owner and frozen before any v2 run (see Part II §V14). |

---

# Part II — v2 research programme

> **Status: APPROVED v2.0 (2026-10-06). Frozen before any v2 run.** It extends Part I (v1.2) and does not replace it.
> Drafted by the owner from Part I and `RESULTS.md` (draft v2.2, `NEW_PROTOCOL.md`), then merged here with the owner's decisions of 2026-10-06:
> - **Early stopping is removed** from the draft: every configuration trains for its fixed budget (§V6.0, §V9).
> - **M6, the live forward test, is deferred**: §V10 is kept as a specification for later and is not built or run under v2.
> - **Track B (H2) is deferred to its own protocol, not dropped** (§V0 item 2).
> - **No automatic network access or git pushes.** Downloads (Q2 cross-check, French long history, optional FRED) are run by the owner via `scripts/download_external.py`. The one approved exception is **wandb online logging**.
> - **Q2 is not skipped.**
> Code mapping: `docs/CODEBASE_NOTES.md` §v2. Cross-references in this part use the prefix V (§V6.1). "Part I §3" means Part I's section 3.
>
> **What stays exactly as in Part I (v1.2):**
> - H1 and its three success criteria
> - the universe and the leave-out set
> - walk-forward folds and embargo. The purge **rule** P = W + H is unchanged, but H now covers every label horizon, so all v2 configurations use **P = 40** (§V0, item 4).
> - the frozen test period and its lock
> - the cost model, with 10 bp as the primary level
> - metrics, the Deflated Sharpe Ratio, PBO and Holm
> - the trial log
>
> **`N_trials` continues from the current log (16).**
>
> **v2.0.1 (2026-10-06) / v2.0.2 (2026-10-07):** implementation clarifications in §V12.1 (items 1–16 after Steps 0a/0b; items 17–20 for Step 0c and V1, before their runs). They change no rule.

## V0. Resolutions before freezing (v2.2)

| # | Point raised | Resolution in this version |
|---|---|---|
| 1 | **The Q1 close-time leak already exists** in the current pipeline: same-date US information reaches ^GDAXI decisions. Fixing it changes R0, so "V1 vs R0" would mix two changes. | Step 0b fixes the leak. New **Step 0d** re-runs the reference as **R0′** = R0 + the Q1 fix + the v2 purge (P = 40), with 10 seeds × 5 folds. That is **+1 trial** and +50 runs. **V1 is compared against R0′**, not R0. The v1 results stay as reported, with a note in RESULTS that M4's ^GDAXI sleeve carried the leak. Its numbers were weak anyway (0.03–0.18), so no conclusion changes. Baselines use only their own ticker's prices and are unaffected. Synthetic runs have one calendar, so they are unaffected too. |
| 2 | **H2 is defined twice.** Plan §7 Track B uses "H2" for the market-neutral residual hypothesis; v2.1 used "H2" for drawdown. | **Label "H2" stays reserved for Track B**, which came first. The drawdown hypothesis becomes **H3** (§V4). **Track B is not part of v2.** It needs short positions (only an ablation flag in v1.2), a different benchmark (cash, not buy-and-hold) and its own cost model for shorting. M4 also found residual features add nothing for broad ETFs, so the prior is weak. **Owner decision (2026-10-06): deferred to its own protocol, not dropped.** No Track-B run happens under v2. |
| 3 | **H2/H3 could be met without skill.** An agent that is 84 % invested has a smaller drawdown than buy-and-hold at 100 % simply by holding less. | H3 now compares against **exposure-matched buy-and-hold**: each sleeve's daily return is ē·r, where ē is the agent's mean exposure on that sleeve in the same (seed, fold) block. The benchmark is costless, which is conservative. A **volatility-matched** benchmark is reported as a sensitivity check (§V4). |
| 4 | **The auxiliary heads look 20 days ahead**, so the 25-bar purge would leak training labels into validation. | The purge rule is generalised: **P = W + H_max**, where H_max is the longest forward horizon of *any* training target (n-step return, auxiliary label, demonstration label). For v2, W = 20 and H_max = 20, so **P = 40**. It applies **uniformly to all v2 configurations, including R0′**, so later comparisons never mix a purge change with a method change. It covers every train→validation boundary, both sides of the inner-validation slice, the development→test boundary, and the end of the long-history pretraining window. |
| 5 | **Not fully specified:** the synthetic worlds and the changepoint method. | Fully specified in §V6.5 (three worlds: W-null, W-vol, W-regime, with parameters, oracle, seeds and pass criteria) and §V6.3 (changepoints = Bayesian online changepoint detection with fixed hyperparameters, two hazard rates, 4 features). Both are fixed before any v2 run. |
| 6 | *(found while specifying)* **Warm-up for long-lookback features.** D2's 252-day returns and the 200-bar rolling z-score need ≈ 450 bars of history. Tickers whose data start in 2006 have only ≈ 315 bars before 2007-07-02. | Before 450 bars of history exist, long-lookback features are set to 0 and carry an **availability flag = 0**. The z-score uses an expanding window with a minimum of 126 bars until 200 are available. Fixed now and unit-tested. |

### What v2.1 changed compared with v2.0

1. **New §V2: is the training data sufficient?** A quantitative answer, plus a data audit (Step 0b) and an optional long-history pretraining step (Step 4).
2. **Condensed programme (§V5).**
   - The 15–20 single-change configurations become **3 core bundles plus 1 conditional step**, tested "bundle first, ablate after".
   - Weekly decisions and new diagnostics are obtained by **re-scoring** existing runs, at no training cost.
   - The 50-seed final ensemble is replaced by an **ensemble inside each network** (bootstrapped heads).
   - Result: **about 70–85 % fewer training runs** and **≤ 10 new trials (with R0′) instead of ≤ 20**, so the bar for the Deflated Sharpe Ratio is also lower.
3. **New §V10: M6, a live forward test.** Today's close leads to tomorrow's position, with timestamped signals, a paper ledger and pre-registered pass criteria.

---

## V1. Diagnosis of the v1 results (summary)

| Observation (c = 10 bp, median over 10 seeds × 5 folds) | Reading |
|---|---|
| At 0 bp the agents roughly tie buy-and-hold on ^GDAXI (0.39 vs 0.39) and lose on the 26 ETFs (transformer 0.58 vs 0.67) | **No gross timing skill** |
| The net gap is turnover: 16–31 turns/yr vs 0.5. Rough check: 16 × 11 bp ≈ 1.8 %/yr ÷ ≈ 10 % vol ≈ 0.18 Sharpe, which matches the observed 0.58 → 0.44 | The trades are not worth their cost |
| Dueling, n-step and PER: ±0.05. Reward variants: PBO 0.59 | Neither the algorithm nor the reward shape is the lever |
| The encoder was the only lever (0.16 → 0.44), and it won by trading less and being long more (84 %) | Signal extraction matters; staying close to buy-and-hold helps |
| Best checkpoint at 30–47 % of training; seed IQR 0.3–0.4 | Overfitting and noisy decisions |

**Root causes:**
- **RC1.** The agent trades from flat, but the real task is deciding *when not to be invested*. Under a long-only cap, a constant exposure has the same Sharpe as buy-and-hold.
- **RC2.** The daily signal (≈ 0.04 vol-units) is far smaller than the noise (≈ 1), so the arg-max switches on noise.
- **RC3.** Each action learns only from the days it happened to be chosen, although prices don't depend on the agent's actions.
- **RC4.** The inputs are short-horizon and carry little slow, cross-asset information.

---

## V2. Is the training data sufficient?

**Short answer: the data is not "garbage", but it is thin in exactly the dimension that matters.** There are many rows and few independent events. A de-risking agent learns from *bear markets and regime changes*, and the development period contains only a handful of them.

### V2.1. Rows vs information

| Quantity | Value | Comment |
|---|---|---|
| Ticker-days in development (26 ETFs × 16.25 yr × 252) | ≈ 106,000 | Looks large |
| Effective independent assets, N_eff = N / (1 + (N−1)·ρ̄) | ρ̄ = 0.3 → **3.1**; ρ̄ = 0.5 → **1.9**; ρ̄ = 0.7 → **1.4** | 26 ETFs carry roughly the information of 1.5–3 independent assets. Correlations rise in crises, exactly when the agent should act. The audit measures ρ̄. |
| Independent equity drawdowns ≥ 15 % in development (2007-07 → 2023-09) | roughly **5–6** (2008–09, 2011, 2015–16 borderline, 2018 Q4, 2020, 2022) | My approximate count; the audit counts them exactly. These are the "labels" a de-risking agent learns from. |
| Of those, inside each fold's training window | F1 sees ~2, F5 sees ~5 | Early folds learn de-risking from 1–2 crises |

### V2.2. Statistical limits (what can be learned or proven at all)

- **Sharpe estimates are noisy.** Under the standard iid approximation, SE(SR) ≈ √((1 + SR²/2)/T). With SR = 0.67 and T = 16.25 years that is about **0.27**. Serial correlation can distort Sharpe ratios further (Lo 2002).
- **Years needed to detect an edge over buy-and-hold** at t = 2, by information ratio (IR) of the active return: T ≈ (2/IR)².

  | IR of active return | 0.25 | 0.5 | 1.0 | 2.0 |
  |---|---|---|---|---|
  | Years of data needed | 64 | 16 | 4 | 1 |

- **Minimum backtest length.** Bailey, Borwein, López de Prado & Zhu (2014) derive MinBTL < 2·ln(N)/E[max_N]². Rearranged for our 16.25 years, the best of N **zero-skill** configurations can be expected to show an annual Sharpe of up to:

  | N trials | 16 | 25 | 36 | 50 |
  |---|---|---|---|---|
  | Spurious "best" Sharpe (upper bound) | 0.58 | 0.63 | 0.66 | 0.69 |

  This bound assumes independent trials; our trials are correlated, so it is conservative. It is still the same size as the effects we hope for. **Every avoided trial matters, which is the second reason to condense the programme (§V5).**

**Conclusion:**
- 106k rows are enough to train a network that **does not overtrade**.
- They are not enough to *learn* subtle daily timing reliably from ~5 crises, or to *prove* a small edge.
- Three levers follow:
  1. use more history with more regime events (Step 4);
  2. use priors and structure instead of raw learning (the default anchor, the cost-structured head);
  3. try fewer configurations.

### V2.3. Data quality risks ("garbage in") to audit

| # | Risk | Check (Step 0b) |
|---|---|---|
| Q1 | **Close-time mismatch, confirmed as an existing leak.** ^GDAXI closes at 17:30 CET (11:30 ET), before the US close. Same-date US-based features in the current pipeline leak US afternoon information into DAX decisions. ^GDAXI is the only non-US-close ticker among the 33. | Fix: for every ticker whose exchange closes before 16:00 ET, all cross-asset and context features use the t−1 value. Add a unit test. Then re-run the reference as R0′ (Step 0d). |
| Q2 | Adjusted prices are re-computed by Yahoo whenever a new dividend occurs, so levels differ by download date (returns should not). | Compare returns from the frozen `data/raw/` with a fresh download over the overlap. Spot-check 5 tickers × 50 dates against a second free source. Flag \|Δ\| > 5 bp. |
| Q3 | Structural breaks inside a series (e.g. commodity ETFs changing their futures holdings or splitting) | List all daily moves beyond 8σ and verify each one. Document breaks and, if needed, mask the affected windows from training. |
| Q4 | Index volume (^GDAXI, ^VIX) may be zero or unreliable | Mask the relative-volume feature where volume is zero or constant |
| Q5 | Missing bars and holiday alignment | Confirm no forward-filled tradable bars (Part I §3) |
| Q6 | Look-ahead in new features | The point-in-time unit test (perturb the future, the feature is unchanged) for every new feature |

Output: `DATA_AUDIT.md` with N_eff, the drawdown event list, the MinBTL table for the actual N_trials, and Q1–Q6 results. Any data fix triggers a baseline re-run (cheap) and a note in RESULTS.

---

## V3. Core idea of v2 (unchanged)

**Learn small, confident deviations from a strong default, using every market day for every possible decision.**

| Principle | Fixes | How |
|---|---|---|
| P1 Default + deviation | RC1 | Anchor = buy-and-hold. The agent learns *when to deviate*, with an explicit hurdle. |
| P2 Costs inside the value function | RC2 | An exact cost-structured head gives a no-trade band. Ensemble uncertainty widens it. |
| P3 Every day for every decision | RC3 | Exogenous (counterfactual) replay |
| P4 Better, slower signals | RC4 | Cross-asset, multi-horizon trend, changepoints, auxiliary heads, longer history |

---

## V4. Hypotheses

- **H1 (confirmatory, unchanged):** tested once at M5 on the configuration selected in §V8.
- **H2:** reserved for plan §7 Track B (market-neutral residuals). **Not tested under v2** (§V0, item 2).
- **H3 (secondary, pre-registered now, never replaces H1): drawdown control beyond lower exposure.**
  - **Benchmark:** exposure-matched buy-and-hold. For each (seed, fold) block and sleeve i, the benchmark's daily return is ēᵢ·rᵢ,ₜ, where ēᵢ is the agent's mean exposure on sleeve i in that block, rebalanced daily without costs (conservative). Sleeves are combined into the equal-weight portfolio exactly as in the harness.
  - **Claim:** the agent's maximum drawdown is lower than the matched benchmark's, **and** its net Sharpe is non-inferior to buy-and-hold (median Δ ≥ −0.05; with rf = 0, scaling doesn't change Sharpe).
  - **Test:** one-sided paired sign-flip permutation test on the per-(seed, fold) differences MDD(agent) − MDD(matched), α = 0.05. Reported at M5 next to H1.
  - **Sensitivity (reported, not tested):** a volatility-matched benchmark, with buy-and-hold scaled per block to the agent's realised volatility.
- **Development hypotheses (validation only):**
  - **HV1:** the structure bundle is non-inferior to buy-and-hold at ≤ 3 turns/yr.
  - **HV2:** the robust-learning bundle raises net Sharpe or lowers seed IQR.
  - **HV3:** the signal pack produces positive **gross** timing skill (0 bp Sharpe > buy-and-hold).
  - **HV4:** long-history pretraining delays overfitting and raises timing IC.

---

## V5. The condensed programme

| Step | What | New trials | Training runs | Decides |
|---|---|---|---|---|
| **0a** | **Re-score** all 16 existing trials and the baselines: learning-curve diagnostics (§V7), execution lag 1, and new descriptive baselines (vol-target, TSMOM-252) | 0 | 0 | Timing IC of existing agents; whether results survive a realistic execution delay |
| **0b** | **Data audit** (§V2.3) | 0 | 0 | Data fixes before anything else |
| **0c** | **Synthetic positive controls** (§V6.5): 3 worlds × {R0′, V1} × 5 seeds, one training path each, evaluated on independent paths | 0 (logged in `synthetic.csv`) | 30 | "No signal" vs "cannot learn" |
| **0d** | **Clean reference R0′** = R0 + the Q1 fix + P = 40, with 10 seeds × 5 folds | 1 | 50 | The reference for V1 (§V0, item 1) |
| **1** | **V1 decision-structure bundle** = target-exposure U-head + exogenous replay + lazy buy-and-hold anchor | 1 | 50 | HV1 |
| **2** | **V2 robust-learning bundle** = V1 + HL-Gauss loss + LayerNorm + bootstrapped heads with uncertainty-gated switching | 1 | 50 | HV2. The internal ensemble replaces the 50-seed final ensemble. |
| **3** | **V3 signal pack** = best + cross-asset context + multi-horizon trend/changepoint inputs + auxiliary heads | 1 | 50 | HV3. Stopping rule 2 is applied here. |
| **3w** | **Weekly decisions** by re-scoring the Step-3 winner: act only every 5th bar | 1 | 0 | Whether fewer decisions beat daily ones |
| **4** | **(conditional) V4 long-history pretraining**, 1926–2007, then fine-tuning; and/or **E2 stationary-bootstrap augmentation** if best checkpoints are still < 50 % of training | ≤ 2 | 10 + 50 (+ 50) | HV4 |
| **Abl** | Ablations of an adopted bundle at screening scale (3 seeds × 5 folds), one component removed each time | ≤ 3 | ≤ 45 | Removes components that don't help, giving a simpler and cheaper final model |
| **5** | Final candidate (§V8) = the winner as trained; its ensemble is internal | 0 | 0 | M5 request |
| **Total** | | **≤ 10** (cumulative ≤ 26) | **≈ 230–385** (lower end: Step 4 and ablations skipped) | |

**Bundle-first logic.** Each bundle groups components that serve the same mechanism and are expected to help together.
- If a bundle **fails**, its whole mechanism is dropped with one trial instead of 3–5.
- If it **passes**, a few cheap ablations show which parts carry the effect.
- The price is less attribution when a bundle fails. That is acceptable, because a positive result is what needs attributing.

**Comparison references:**
- V1 is compared against **R0′** (`m4_cross_resid_transformer` with the Q1 fix and P = 40). Its v1 value of 0.44 is no longer the reference.
- Each later bundle is compared against the best adopted configuration so far.

---

## V6. Component specifications

### V6.0. Common settings for every v2 configuration (including R0′)

- **Data pipeline:** Q1 fix applied, plus the audit fixes from Step 0b.
- **Purge:** P = W + H_max = 20 + 20 = **40 bars** at:
  - every train→validation boundary;
  - both sides of the inner-validation slice;
  - the development→test boundary;
  - the end of the pretraining window.
  Embargo E = 10 as before.
- **Warm-up:** long-lookback features carry availability flags (§V0, item 6).
- **Fixed training length** per configuration: no early stopping (owner decision 2026-10-06). Checkpoint selection on inner validation only.
- **Re-scoring outputs:** lag-0 and lag-1 (§V7, LC8).

### V6.1. V1: decision structure

**Target-exposure actions and exact cost-structured head (subsumes counterfactual replay):**
- **Actions:** choose the target exposure e′ ∈ {0, ¼, ½, ¾, 1}.
- **Head:** Q(m, p, e′) = U(m, e′) − κ(m)·|e′ − p|, with κ = (c + s/2)/σ̂(m). The network outputs **U(m, e′)** only.
  - This is exact, because prices don't depend on the position.
  - The position-input branch is no longer needed; keep it behind a flag.
- **Targets for all five U values from every market transition:**
  y(e′) = e′·rₜ₊₁/σ̂ₜ + γ·maxₑ″[U⁻(mₜ₊₁, e″) − κ(mₜ₊₁)·|e″ − e′|]
  The online network chooses e″ and the target network evaluates it (Double DQN).
- **Why it is valid:**
  - Under zero market impact, rewards of unchosen actions are computable after the next price. In FX, "action augmentation" beat ε-greedy by 6.4 %/yr on average (Huang 2018).
  - Hindsight Learning in Exo-MDPs formalises this idea (Sinclair et al. 2023).
- **Effect:** the agent switches only if U(e′) − U(p) > κ·|e′ − p|. That is a no-trade band, in line with "trade partially towards the aim" (Gârleanu & Pedersen 2013).
- **Requires a Markov reward:** use `vol_scaled_pnl`, not `diff_sharpe`.

**Lazy buy-and-hold anchor:**
- In the *training* reward only, subtract η = **0.01** vol-units on every bar with e′ ≠ 1 (fixed a priori). Initialise the U-bias so that full exposure is preferred.
- A deviation must then be expected to beat holding by about 0.16 annual Sharpe units.
- **Literature:** Lazy-MDPs (Jacq et al. 2022) learn *when* to take control from a default policy. Residual Policy Learning (Silver et al. 2018) learns corrections to an existing controller and is more data-efficient than learning from scratch.

**ε-greedy:** fixed at 0.05. It only decides which days are visited and has no cost consequence.

**Owner ideas of 2026-10-06 (recorded before any v2 run):**
- *"Buy more than one unit per decision / buy based on the remaining capital."* This is the target-exposure action space above: any exposure from 0 to 100 % of the sleeve in **one** transaction. In R0 the old one-step limit rarely binds: 89 % of its moves are single ¼-steps, and merging multi-step moves would remove only ≈ 12 % of its transactions. R0's real problem is jitter between 75 % and 100 %, which the cost-structured head and the anchor address. Measured on the stored exposures; see `README.md` change report, 2026-10-06.
- *Costs "per transaction, not per unit".* A broker's fixed fee is charged per transaction; the bid-ask spread is always paid on every euro traded. For **neo-broker training variants only**, the head therefore gets an exact fixed-fee term: Q = U(m, e′) − κ(m)·|e′ − p| − φ(m)·1[e′ ≠ p], with φ = (fee / sleeve capital) / σ̂(m). Under the H1 cost model (proportional costs) φ = 0, so V1 itself is unchanged.
- *"Avoid a bias towards cheap or expensive stocks: buy capital/x instead of units."* This has been the design since M2. Exposure is a fraction of each sleeve's capital, P&L is computed from returns, costs are proportional to the traded value, and every input is scale-free. The share price never enters. Whole-share rounding is not modelled; fractional shares are assumed, a realism note for the neo-broker scenario.
- **Deferred, needing their own protocol:**
  - exposure above 100 % (leverage): with rf = 0 it leaves the Sharpe ratio unchanged, and Part I fixes long-only with a cap of 1;
  - a continuous fraction x: needs policy-gradient or actor-critic methods (§V11);
  - one shared capital pool across ETFs: portfolio allocation, which needs a new action space and a new benchmark.

### V6.2. V2: robust value learning and internal ensemble

- **HL-Gauss loss:** cross-entropy over 51 value bins with Gaussian label smoothing, spanning ±5 SD of the U targets on F1's inner slice. Categorical cross-entropy "mitigates issues inherent to value-based RL, such as noisy targets and non-stationarity" (Farebrother et al. 2024).
- **LayerNorm in every hidden layer.** It gives provably convergent TD learning even off-policy (Gallici et al. 2025, PQN).
- **Bootstrapped heads with gating:**
  - K = 10 U-heads on a shared trunk, each with its own bootstrap mask (Bootstrapped DQN, Osband et al. 2016).
  - Execution rule: switch only if mean_k[ΔU] − κ|e′ − p| > z·sd_k[ΔU], with z = 1.
  - The heads are the **final ensemble**, so no 50-seed retraining is needed.
  - The averaging logic is in the spirit of Averaged-DQN (Anschel et al. 2017).

### V6.3. V3: signal pack

**Leakage rules:**
- Aggregates use only the **26 training tickers + ^VIX**, never the leave-out tickers.
- Tickers whose exchange closes before 16:00 ET get **t−1** values of all US-based context features (Q1).

**D1 cross-asset context** (computable from existing data):
- bond momentum (IEF+TLT, 1/3/12 months) → equities;
- equity momentum (SPY+EFA) → bonds;
- credit proxy (LQD − IEF, 20/60 days);
- term proxy (TLT − IEF, 20/60 days);
- breadth (share of the 26 above their 200-day SMA);
- dispersion and average 60-day pairwise correlation;
- the ticker's own 3- and 12-month momentum rank.

Basis: past bond returns predict equities positively and equity returns predict bonds negatively, giving a 45 % higher Sharpe than standard time-series momentum in 20 countries (Pitkäjärvi, Suominen & Vaittinen 2020). Optional replacement for the credit proxy: FRED BAA10Y, daily since 1986-01-02, lagged one bar. That is a new data source, so it needs an owner decision.

**D2 multi-horizon trend and changepoints:**
- vol-scaled returns over 21, 63, 126 and 252 days, and MACD at three timescales (the exact set is my choice, in the spirit of Lim, Zohren & Roberts 2019);
- **Changepoint inputs (fixed now): Bayesian online changepoint detection** (BOCPD; Adams & MacKay 2007). It computes online, exactly, the probability distribution of the current run length.
  - **Input series:** the ticker's vol-scaled daily return xₜ = rₜ/σ̂ₜ₋₁ (60-day σ̂, lagged).
  - **Model:** Gaussian with unknown mean and precision, using a Normal–Gamma conjugate prior: μ₀ = 0, κ₀ = 1, α₀ = 1, β₀ = 1.
  - **Hazard:** constant, H = 1/λ, with **λ ∈ {21, 126}** (two timescales).
  - **Run length:** truncated at R_max = 500 and renormalised.
  - **Features per λ (4 in total):** P(run length ≤ 5 | x₁:ₜ), a recent-changepoint score; and min(E[run length | x₁:ₜ]/λ, 1), the regime age.
  - **Properties:** filtering only (no smoothing), so it is causal by construction. Cost is O(T·R_max) per ticker and λ, about 2 million updates, computed once in the feature cache.
  - **Relation to Wood et al.:** their CPD module "outputs a changepoint location and severity score" (Wood, Roberts & Zohren 2021), and multi-timescale CPD "can complement multi-headed attention" (Wood et al. 2021). BOCPD is a deliberately simpler, pre-registered substitute. I didn't check their method's details or cost; you report it as computationally heavy.
- Time-series momentum is documented over 1–12 months (Moskowitz, Ooi & Pedersen 2012).

**D3 auxiliary heads** (encoder only, weight 0.1):
- next 5- and 20-day vol-scaled return;
- next 20-day log realised volatility.

Their 20-day labels set H_max = 20, which is why P = 40 (§V6.0). A unit test asserts that no training sample's label window crosses a purge boundary.

Basis: self-predictive objectives (SPR, Schwarzer et al. 2021), and signal extraction as the separating element (Guijarro-Ordonez, Pelger & Zanotti).

### V6.4. Step 4 (conditional): more history, more regimes

**V4 long-history pretraining.**

*Data:*
- Kenneth French daily **industry portfolios**, available from 1926-07-01. Add the daily market factor too if the audit confirms it covers the same span.
- Window: **1926-07-01 → 2007-06-29**, which ends before the development range. So there is **no overlap with any validation or test block**, and one pretrained checkpoint per seed serves **all five folds**.
- This adds about 81 years of daily US data, including roughly a dozen major bear markets (my approximate count; the audit counts them).

*Inputs:* only return-derived features. VIX does not exist before 1990, so that input is masked with an availability flag.

*Procedure:*
1. Pretrain the V-winner on the long history with exogenous replay: 10 seeds, once.
2. Fine-tune on each ETF fold with 50 % of the normal budget.

*Caveat:* older regimes differ (non-stationarity). Fine-tuning handles the domain shift, and the go rule decides.

**E2 stationary-bootstrap augmentation** (run only if best checkpoints are still < 50 % of training): 50 % of training episodes on stationary-bootstrap paths of the training fold (mean block 20 days; Politis & Romano 1994).

**Moved to reserve or descriptive:**
- standalone counterfactual replay (now inside V1);
- vol-target anchor (now a descriptive baseline);
- hindsight demonstrations (DQfD);
- Munchausen;
- CVaR heads;
- learned skip-policy.

### V6.5. Synthetic positive controls (Step 0c), full specification

**Common structure (all three worlds):**
- **Tickers and factor structure:** 26 synthetic tickers with a one-factor structure: rᵢ,ₜ = μᵢ,ₜ + sᵢ·σₜ·(√ρ·zₜ + √(1−ρ)·uᵢ,ₜ).
  - zₜ and uᵢ,ₜ are independent standardised Student-t shocks with ν = 5.
  - ρ = the average pairwise correlation measured in the data audit (default 0.5 if the audit hasn't run).
  - The ticker scales sᵢ are log-uniform on [0.5, 2], mimicking bonds through to emerging-market equities.
- **Data generated:** prices compound from 100. No volume is generated, so volume features are masked, as in Q4.
- **VIX proxy:** VIXₜ = 100·√252·σₜ₊₁|ₜ·exp(0.25·ηₜ), with ηₜ ~ N(0,1).
  - σₜ₊₁|ₜ is the true conditional daily vol of the common factor: the GARCH forecast in W-null and W-vol, the current regime's vol in W-regime.
  - It is lagged one bar exactly as in the real pipeline. It is informative but deliberately noisy: in W-regime the bull/bear vol ratio is about 2.3 noise-SDs.
- **Paths and evaluation:**
  - The **training path** has development length (≈ 4,090 bars plus 504 warm-up bars). The agent selects checkpoints on the last 252 bars of the training path, purged.
  - Evaluation runs on an **independent 20-year path** from the same data-generating process, which is never used for training.
- **Seeds:** agent seeds 0–4; training-path seeds 1000–1004; evaluation-path seeds 2000–2004.
- **Costs and budget:** same costs (10 bp + 1 bp) and the same training budget as the real configuration.
- **Calibration rule** (checked before any agent run; not a trial): in W-vol and W-regime, the oracle's net Sharpe gain over buy-and-hold on the evaluation paths must be **≥ 0.15**. If it isn't, raise GARCH persistence (W-vol) or regime separation (W-regime) until it is. Then freeze the parameters.

| World | Data-generating process | Benchmark policy ("oracle", knows the true process) | Pass criterion |
|---|---|---|---|
| **W-null** | GARCH(1,1) common variance (α = 0.08, β = 0.90, unconditional daily vol 1 %). **Risk–return trade-off:** μᵢ,ₜ = λᵢ·(sᵢσₜ)², with λᵢ set so the unconditional Sharpe is ≈ 0.6. The mean–variance weight μ/σ² is then constant, so **no timing beats buy-and-hold**. | Buy-and-hold (100 % from the first bar) | Turnover ≤ 2/yr **and** net Sharpe ≥ buy-and-hold − 0.05 |
| **W-vol** | The same GARCH, but with **constant** μᵢ (Sharpe ≈ 0.6 at unconditional vol). The conditional Sharpe is high in calm periods, so de-risking in high volatility pays (the Moreira & Muir mechanism). | eₜ = min(1, σ̄²/σ²ₜ₊₁\|ₜ), using the true conditional variance and the median σ̄. Quantised to the ¼ grid; trades only when the quantised target changes. | Agent's net Sharpe gain over buy-and-hold ≥ 50 % of the oracle's |
| **W-regime** | **2-state Markov switching** in the common factor (Hamilton 1989-type). Bull: μ = +15 %/yr, σ = 14 %/yr, P(stay) = 0.998 (expected 500 days). Bear: μ = −20 %/yr, σ = 25 %/yr, P(stay) = 0.99 (expected 100 days). Unconditional Sharpe ≈ 0.56. A development-length path holds ≈ 6–7 bear episodes, about as scarce as the real data (§V2.1). No GARCH. | Hamilton filter with the true parameters gives πₜ = P(bull \| data). Target = clip(wₜ/w_bull, 0, 1), with wₜ = (πₜμ_b + (1−πₜ)μ_r)/(πₜσ_b² + (1−πₜ)σ_r²). Quantised to the ¼ grid with one-step hysteresis. | Agent's net Sharpe gain over buy-and-hold ≥ 50 % of the oracle's |

**Reading the outcome:**
- R0′ overtrading in W-null means the machinery manufactures trades from noise (RC2).
- Failing W-vol and W-regime means the agent can't learn timing that exists, and stopping rule 1 applies.
- V1 passing where R0′ fails is evidence for the decision-structure changes, independent of real data.

---

## V7. Learning-curve diagnostics (logged for every run; existing runs re-scored in Step 0a)

| ID | Diagnostic | Why |
|---|---|---|
| LC1 | Inner-validation net Sharpe every 5k transitions: area under the curve, last-half slope, best-checkpoint position | The learning curve |
| LC2 | Sharpe at 0 bp vs 10 bp | Skill vs cost |
| LC3 | **Timing IC** = Spearman ρ(eₜ, rₜ₊₁/σ̂ₜ); Sharpe of (eₜ − ē)·rₜ₊₁ | Direct measure of timing skill |
| LC4 | Attribution vs buy-and-hold: timing = gross − B&H; cost = net − gross | Explains every gap |
| LC5 | Action-gap ratio = median \|U(best) − U(2nd)\| ÷ SD(TD error) | Below 1 means noise-driven decisions |
| LC6 | Switches per 100 bars, time at anchor | Overtrading |
| LC7 | Predicted vs realised discounted return | Over-estimation |
| LC8 | **Lag-1 Sharpe** (execution at the next close) | Live realism (§V10) |

---

## V8. Adoption, ablation, stopping and selection rules

**Adopt** a bundle over the current reference if, at 10 bp on the 26-ETF validation set (50 seed-fold pairs):
- **(a)** median Δ net Sharpe ≥ +0.05 and one-sided paired permutation p < 0.10; **or**
- **(b)** median Δ ≥ −0.02 **and** turnover −30 %, or seed IQR −25 %, or timing IC +0.01.

**Ablate** (screening scale, 3 seeds × 5 folds) only for an adopted bundle. Remove one component per ablation, at most 3 in total. Keep a component only if removing it lowers median net Sharpe by ≥ 0.03 **or** raises turnover by ≥ 20 %. Otherwise drop it, giving a simpler model and cheaper inference. Ablation results guide simplification only, never H1 selection.

**Stopping rules:**
1. **After Step 0c:** if R0′ and V1 both fail W-vol and W-regime, pause real-data trials (except R0′, which is needed anyway) and iterate on synthetic data (not trials).
2. **After Step 3:** if no configuration reaches **gross** median Sharpe > buy-and-hold, stop development. Report "no exploitable timing signal with these inputs" and recommend not spending the test period.
3. **Budget:** at most 10 new trials (cumulative ≤ 26).

**Selection for M5:** the adopted configuration with the highest median validation net Sharpe. It must pass **gate G-lag**: its lag-1 median net Sharpe is no more than 0.05 below its lag-0 value. If it fails, it is not promoted. The M5 request is made only if its validation median net Sharpe ≥ buy-and-hold.

---

## V9. Compute and trial budget

| | v2.0 | **v2.1** |
|---|---|---|
| New trials | ≤ 20 (+1 for R0′ = 21) | **≤ 10** (incl. R0′) |
| Cumulative N_trials | ≤ 37 | **≤ 26** |
| Training runs on real data | ≈ 1,250 (20 configs × 50 + 50-seed final + R0′) | **≈ 200–355** |
| Synthetic runs | ≈ 180 | **30** |
| Total | ≈ 1,430 | **≈ 230–385 (≈ −73 % to −84 %)** |

**Engineering savings (no change of substance; apply to all v2 runs equally):**
1. **Feature cache:** compute features once per ticker and fold and share them across all configs.
2. **Exogenous replay is a static dataset.** Train in large vectorised mini-batches over (date, ticker) with no environment loop. Measure the time per run in Step 1 and report it.
3. **One pretraining checkpoint per seed is shared by all folds** (Step 4).
4. Parallel seeds and mixed precision where the hardware allows.

*(The draft's "early stopping with patience" was removed by the owner on 2026-10-06. Each one-year inner-validation score has a standard error of about 1 Sharpe unit, so stopping on it would often react to noise.)*

**Not recommended:** warm-starting fold k+1 from fold k. It saves compute but makes folds dependent, and it may amplify the overfitting seen in M2.

---

## V10. M6: live forward test ("today's data → tomorrow's decision") — DEFERRED

> **Owner decision 2026-10-06: future work, not part of v2 execution.** The core aim of v2 is to finish the research and improve the agent. This section is kept as a specification for later; nothing in it is built or run now. A future implementation must also respect the project rule that network access and git pushes are triggered by the owner only: steps 1 (fetch) and 5 (push) of §V10.2 would each need the owner's explicit approval.

**Purpose.** Prove that the agent works as a real process on data no one has seen: implementability, fidelity to the backtest, and consistency. A forward test cannot prove *skill* quickly. Per §V2.2, a 12-month test can only detect an information ratio of about 2. It can, however, catch a broken pipeline, hidden look-ahead, or a backtest that doesn't hold in reality.

**Preconditions:**
- M5 is done (or the owner decides to skip it).
- The final recipe (code hash, config hash) is frozen.
- The model is **refit once** with the frozen recipe on all data up to a fixed cut date (purged), then frozen for the whole forward test.

### V10.1. Execution timing

- The v1 harness takes the first decision at a bar's close and earns the following returns. Live, nobody can trade at the exact close that produced the signal.
- Two realistic options:
  - (i) **Market-on-close orders** sent shortly before the close, using near-close prices as a proxy;
  - (ii) **lag 1**: decide after today's close and trade at tomorrow's close.
- **Option (ii) is the default for M6.** Step 0a re-scores every strategy with lag 1 (LC8), and gate G-lag (§V8) makes sure the selected agent doesn't depend on unattainable timing.

### V10.2. Daily pipeline (runs after the US close, e.g. 22:30 CET)

1. **Fetch** today's bars for all 33 tickers and ^VIX.
2. **Validate:** completeness, stale prices, > 8σ moves without a corporate-action flag, the Q1 close-time rule. If validation fails, **no new signal**: keep the current positions and log the incident.
3. **Compute features** with the *same* code as the backtest (hash-checked).
4. **Run inference:** U-head ensemble → target exposure per ticker for the next session.
5. **Publish and timestamp:** write `signals/YYYY-MM-DD.csv` with date, ticker, exposure, model hash and data hash. Commit and push it to a remote repository **before the next session opens**. The pushed commit is the proof that there was no hindsight.
6. **Execute on paper** at the next close (lag 1) in a ledger. Charge costs at the primary level (10 bp + 1 bp) and under the neo-broker scenario.
   - Optional second check: a free broker paper account (e.g. Alpaca). It simulates fills against real-time NBBO quotes but ignores market impact, slippage from latency, queue position and dividends.
7. **Run the baselines on the same ledger:** buy-and-hold, momentum-60, MACD, vol-target. Pairs are by day.
8. **Replay check, weekly:** rerun the backtest engine over the same dates and compare it with the live ledger.

### V10.3. Pre-registered criteria (minimum 12 months, preferably 24)

| ID | Criterion | Pass |
|---|---|---|
| L1 | **Operational** | Signal pushed before the cut-off on ≥ 98 % of trading days; zero manual overrides |
| L2 | **Implementation fidelity** (live vs replay) | Median \|daily return difference\| ≤ 1 bp per sleeve; annual tracking difference ≤ 0.5 % |
| L3 | **Consistency with the backtest** | The live 12-month Δ Sharpe (agent − buy-and-hold) lies within the 5th–95th percentile of 12-month Δ Sharpe windows from the validation distribution (seeds × folds) |
| L4 | **Effect estimate** (reported, not tested) | Active-return IR with a 95 % CI, and drawdown vs **exposure-matched** buy-and-hold (H3 in live form) |

If L1–L3 pass, the evidence is **"the backtest is a faithful forecast of live behaviour"**. Skill evidence then combines the M5 result with the growing live record.

### V10.4. Real-money pilot (optional, only after L1–L3 pass)

- **Small capital and hard limits:**
  - stop if live drawdown relative to buy-and-hold exceeds the backtest's 99th percentile;
  - stop after 3 operational failures in a month.
- **Instruments for an EU retail investor:**
  - As of a 2021 European Parliament question, EU retail investors cannot buy **US-domiciled ETFs** such as SPY or EFA. The reason is the PRIIPs key-information-document requirement, which US issuers can't meet.
  - A real-money pilot would therefore need **UCITS equivalents**, which have different listings, trading hours and currency (EUR), so FX exposure appears.
  - Treat that mapping as a separate step. Paper-trade on the US tickers the model was trained on.
- Use the neo-broker cost scenario. A 26-ETF portfolio is unrealistic at EUR 10k (RESULTS: one ¼ step ≈ 104 bp), so use a small deployment set (`deploy2` or its UCITS mapping).
- Research code only; this is not investment advice.

---

## V11. Deferred or not recommended

| Idea | Status | Reason |
|---|---|---|
| Time-series foundation-model embeddings (e.g. Chronos) | Not on the H1 path | Pretraining corpora may overlap validation or test periods, which is an information-set violation (Ansari et al. 2024; Zhang et al. 2026: later-trained models were worse than point-in-time ones in 18 of 20 US cases) |
| Hindsight demonstrations (DQfD), Munchausen, CVaR heads, learned skip-policy | Reserve | Run only if a core bundle passes and a specific diagnostic points to them (DQfD: LC1; Munchausen: LC5; CVaR: H3) |
| Policy-gradient methods (PPO, SAC) | Out of scope | Outside the DQN framework, and M2 showed the algorithm is not the bottleneck |
| Intraday data, news or LLM sentiment | Out of scope | No free point-in-time history |
| Resets and larger networks (BBF) | Not planned | My reading: they target lost plasticity, but v1's problem is overfitting |
| Track B, market-neutral residuals (plan §7, hypothesis H2) | **Deferred** to its own protocol (owner decision 2026-10-06) | Needs shorts, a cash benchmark and short-side costs. M4 found residual features add nothing for broad ETFs (§V0, item 2) |

---

## V12. Implementation notes for Claude Code

1. Map reward, replay, action selection, encoder, feature pipeline and the harness's execution timing. Record them in `docs/CODEBASE_NOTES.md` §v2.
2. **Step 0a needs stored daily exposures per run.** If they aren't stored, re-run inference from the saved checkpoints (cheap). Re-scoring only re-evaluates finished strategies, so it creates no new trials. The one exception is weekly decisions (Step 3w), which are a new strategy and therefore a trial.
3. `replay.mode: agent | exogenous`. The exogenous mode samples (ticker, date) and computes all five U targets inside the loss.
4. New modules:
   - `harness/synthetic.py` (worlds, fixed seeds)
   - `data/longhistory/` (French files, read-only, with their own manifest hashes)
   - `live/`: **not built under v2** (M6 deferred)
   - `scripts/download_external.py`: owner-run downloads (Q2 cross-check, French portfolios, FRED); never called by the code
5. Tests:
   - point-in-time behaviour for every feature;
   - leave-out exclusion in aggregates;
   - the Q1 close-time lag;
   - U-head target computation;
   - gating rule;
   - purge horizon: no training label window (n-step, auxiliary, demonstration) crosses a boundary with P = 40;
   - warm-up: availability flags are 0 until 450 bars of history exist; the expanding z-score is used before 200 bars;
   - BOCPD causality: perturbing x after t leaves the features at t unchanged;
   - exposure-matched benchmark: equals ē·buy-and-hold per sleeve and block.
6. One YAML per bundle (`config/v2/V1.yaml` …), with hashes logged as in v1.
7. **Network rule:** no code downloads, pulls or pushes on its own. Data needed by v2 comes from `scripts/download_external.py`, run by the owner. wandb online logging is the one approved exception.

### V12.1. Implementation clarifications (2026-10-06, after Steps 0a/0b, before any v2 training run)

Where the text above leaves a detail open, the code does the following. No rule is changed. The owner can still override any item before Step 0d.

| # | Item | Implementation |
|---|---|---|
| 1 | Execution lag (LC8) | `harness/backtest.backtest(lag=L)`: the exposure decided at the close of bar t is held from the close of t+L, and the first L days of a block are flat. Costs are charged whenever the held exposure changes, so a lag-1 buy-and-hold pays its entry one day later. |
| 2 | Gate G-lag (§V8) | Median lag-1 net Sharpe − median lag-0 net Sharpe ≥ −0.05. Both medians are taken over the 50 seed–fold pairs, as worded (difference of medians). The paired median difference is reported alongside. |
| 3 | LC4 attribution | timing = median gross Sharpe − median buy-and-hold net Sharpe; cost = median net − median gross, so the reported parts add up. The same in %/yr (CAGR). Paired medians are kept in the CSVs. |
| 4 | LC3 timing | Timing IC = Spearman ρ(eₜ, rₜ₊₁/σ̂ₜ) per sleeve and block, averaged over the sleeves of a set. σ̂ is the ex-ante EWMA-60 volatility (`rl/features.ex_ante_vol`). The IC is undefined for a constant exposure. Timing Sharpe = annualised Sharpe of the equal-weight portfolio of (eₜ − ē)·rₜ₊₁, with ē the sleeve's mean exposure in that block. |
| 5 | LC5 action gap | Median over decision bars of Q(best valid) − Q(second-best valid) at the selected checkpoint, divided by SD(TD error). SD(TD error) ≈ mean\|TD error\|·√(π/2) (Gaussian approximation), from the training log at that checkpoint. |
| 6 | LC7 over-estimation | G = Σ γᵏ rₜ₊ₖ / reward_scale along the same greedy path, with the agent's own reward, truncated at the block end. Bars with fewer than 300 bars left are excluded (γ³⁰⁰ ≈ 0.05). Reported scale-free as (mean Q − mean G)/SD(G), plus Q − G and corr(Q, G). |
| 7 | H3 benchmarks | **Exposure-matched:** ē·rₜ per sleeve, costless, aggregated with equal weights like the agent. **Vol-matched:** the equal-weight buy-and-hold return scaled to the agent's realised net volatility in the block. In the Step 0a preview, "lower drawdown" means lower by more than 0.1 pp. |
| 8 | Vol-target baseline | eₜ = min(1, σ̄ₜ²/σₜ²) on the K = 4 grid, with σₜ the EWMA(60) SD of daily log returns and σ̄ₜ its expanding median (≥ 252 bars). It is the estimated counterpart of the W-vol oracle (§V6.5), logged as a baseline row, not a trial. |
| 9 | Pipeline v2, Q1 | For early-close tickers (only ^GDAXI), `resid` and `resid_cum30` use the t−1 value. VIX features were already lagged one bar for every ticker, and own-price features need no lag. |
| 10 | Pipeline v2, Q4 | `vol_rel20` is masked (0 after z-scoring) where volume is zero or its 20-bar SD is zero (^GDAXI: 9 bars in development). |
| 11 | Warm-up (§V0 item 6) | In R0′'s feature set, the residual pair is the only long-lookback group. Its flag `resid_avail` = 1 only when both z-scored residual features are valid **and** the ticker has ≥ 450 bars of history; otherwise both features are 0. The z-score needs at least 126 bars (expanding until the window is full). |
| 12 | z-score window | §V0 item 6 mentions a "200-bar rolling z-score", but the pipeline's z-score has used a 252-bar window since M4. v2 keeps 252 bars, so R0′ changes only what §V0 and §V6.0 require. **The owner can switch to 200 before Step 0d** (one line in `config/v2/R0prime.yaml`). |
| 13 | Purge | Every v2 config sets `splits.purge_horizon: 20`, so P = 20 + 20 = 40 at every boundary that `harness/splits.fold_ranges` cuts, including both sides of the inner-validation slice. |
| 14 | Configurations | One YAML per v2 configuration in `config/v2/` (R0′: `config/v2/R0prime.yaml`), inheriting from the v1 chain. |
| 15 | Q2 second source | The stooq spot check could not run (no access rights). Owner decision 2026-10-06: not repeated. Q2 rests on the Yahoo-internal checks (`DATA_AUDIT.md`). |
| 16 | MinBTL | `DATA_AUDIT.md` reports the §V2.2 table for both the development length and the validation length (9.70 years). Trials are compared on the validation blocks, so the second is the relevant one. |
| 17 | Synthetic worlds (§V6.5; `harness/synthetic.py`), written 2026-10-07 before any synthetic agent run | 26 tickers S00–S25 with one fixed draw of the scales sᵢ; ρ = 0.48 (DATA_AUDIT). In W-regime the factor drift is scaled by sᵢ, so every ticker has the factor's Sharpe ratio. The evaluation path also gets 504 warm-up bars. Training path and evaluation path follow each other per ticker; the fold trains on the first path (cut at the junction, P = 40, inner validation = its last 252 bars) and scores the 20 years after the second path's warm-up. W-vol oracle: σ̄ = the path median of σₜ₊₁\|ₜ. W-regime oracle: Hamilton filter on the observable proxy xₜ = meanᵢ(rᵢ,ₜ/sᵢ) (Gaussian likelihood, true parameters), using the one-bar-ahead P(bull). "One-step hysteresis" is implemented as a dead band: the exposure moves to the rounded target only once the target is ≥ ¾ step away. A full-step band would never return to 100 %, because the target stays just below 1 while P(bull) < 1. Pass criteria use medians over the 5 runs. wandb is off for synthetic runs; they are logged in `experiments/synthetic.csv`. |
| 18 | Calibration rule (§V6.5), applied 2026-10-07 | With β = 0.90 the W-vol oracle gained only +0.08 Sharpe over buy-and-hold (median of the 5 evaluation paths); the rule needs ≥ 0.15. Raised to **β = 0.91 (persistence 0.99), the smallest step tried that passes: +0.19**. W-null keeps the same GARCH, so the two worlds differ only in the drift. W-regime passes with the specified parameters (+0.36). Frozen before any agent run. |
| 19 | V1 (§V6.1; `rl/exogenous.py`, `config/v2/V1.yaml`) | **Budget:** transitions × update_ratio gradient steps of 256 (ticker, day) samples, the same number of steps as R0′; all 5 U outputs are trained on every sample (Huber, Double-DQN target with the cost matrix, soft target updates). The immediate term is clipped like the env reward (`reward_clip`). **Anchor:** η = 0.01 enters the training reward only. The output biases start at each exposure's buy-and-hold value (mean training reward / (1 − γ)). No reward normalisation: vol-scaled rewards are unit-scale, and κ must be in the same units. ε-greedy has nothing to act on, because there is no environment loop. For V1's Q diagnostics (LC5, LC7), Q = U − the trading cost, and G uses the training reward. |
| 20 | Stopping rule 1 (§V8) | Enforced in the run queue: V1's real-data run starts only if `harness.synthetic.real_data_allowed()` is true, i.e. R0′ and V1 do not both fail W-vol and W-regime. Missing results also pause. |
| 21 | Synthetic iteration after stopping rule 1 (2026-10-07, 00:58) | R0′ and V1 both failed W-vol and W-regime, so real-data trials pause and V1 is iterated on synthetic data only. Variants are run with `scripts/run_agent.py --synthetic --set …` and logged in `experiments/synthetic.csv` with their config hash. **None is a trial.** The candidate levers are those that make the U-head learn mean value differences faster and switch less on noise: MSE instead of Huber; n-step "hold" targets with n ≤ 20 (inside P = 40); no anchor (diagnostic only); a 10× learning rate; and V2-type bootstrapped heads with the z = 1 gate, ± LayerNorm. A variant goes to real data only as a YAML in `config/v2/` that the owner has approved, and it is then a trial. |
| 22 | Step 4 (§V6.4; `harness/longhistory.py`, `config/v2/V4.yaml`), written 2026-10-07 before its run | **Data:** the 5 value-weighted French industry portfolios plus the market (Mkt-RF + RF), daily, 1926-07-01 → 2007-06-29 (21,495 bars), compounded into prices. Rows after the window are cut on load. **Inputs:** V1b's features plus a new flag `vix_avail` (VIX is masked before 1990; constant 1 on the ETF data); volume masked; residual features against the PCA of the 6 long-history series. **Pretraining:** V1b's agent with its full budget (100k transitions), 10 seeds, once. It trains from bar 504 up to P = 40 bars before the window's last 63 bars, and selects its checkpoint on its last 252 training bars. wandb is off; it is not a trial on its own. **Fine-tuning:** every ETF fold starts from its seed's selected checkpoint, without the buy-and-hold prior, with 50 % of the budget (50k transitions). It starts only if the pretraining finished (queue guard). **Reference:** V1b, the parent, so the pretraining is the only change apart from `vix_avail`. The protocol says "pretrain the V-winner"; V1b is used because the owner ordered Step 4 before its real-data result is known. |
| 23 | Hyperparameter search and the 4 trials it feeds (owner decision 2026-10-07; `harness/hpo.py`), written before any search run | **Objective (owner):** the agent should time markets on a 1–4-week horizon at small costs instead of holding for years. H1, its cost level, the folds and the test lock are unchanged. **New world W-swing** (`harness/synthetic.py`, §V6.5 structure): constant vol of 1 %/day, ρ = 0.48; each ticker's drift is μ̄ + xᵢ,ₜ, with μ̄ giving a Sharpe ratio of 0.6 and x an AR(1) with a 10-day half-life (φ = 0.933), independent across tickers. Oracle: a scalar Kalman filter per ticker on its own scaled returns, with the true parameters; long once the predicted next-bar return exceeds b = c(1 − φ), flat once it falls below −b (c = 5 bp, the search cost), unchanged in between. **Calibration rule** (applied 2026-10-07, oracle vs buy-and-hold only, seeds 0–4): the smallest drift SD of {1.0, 1.25, 1.5, 1.75, 2.0}·10⁻³ per day with a median oracle net gain ≥ 0.15 at 5 bp and an oracle turnover of 12–50/yr. Results: +0.04 (5.6/yr), +0.14 (12.4/yr), **+0.32 (17.0/yr) → 1.5·10⁻³**, frozen. **Search (synthetic only, no trials):** every setting keeps V1b's structure without a buy-and-hold bias (`anchor_eta` 0, `agent.prior: none`) at 2 bp + 3 bp half-spread. Worlds W-null, W-regime and W-swing; W-vol is dropped from the objective. Space: loss {MSE, HL-Gauss}, n {5, 10, 20, 40}, γ {0.9, 0.95, 0.99}, width ×1 / ×4 (`transformer_dim` 8 / 32, hidden [64, 32] / [256, 128]), AdamW weight decay {0, 0.1}, heads {1, 10, 20}, random-prior scale {0, 3}, gate z {0, 0.5, 1}, learning rate {1e-4, 3e-4}. **Score** = mean over W-swing and W-regime of median(gain) / median(oracle gain) over the seeds, minus 1 if the median W-null gain is below −0.05. **Successive halving:** 32 settings (a balanced random design: every option equally often, seed 20261007) × seeds 0–4 at ¼ of the budget → best 8 × seeds 0–9 at ½ → best 2 × fresh seeds 10–19 at the full budget; about 5 checkpoints per run. HL-Gauss: 51 bins, σ = 0.75 bin width, support = each exposure's hold value ± 6 SD of its immediate term on the training samples (stored per run). Random priors: a frozen prior network added to the output, β·prior(m) (Osband et al. 2018). **Real data, 4 trials (N_trials 19 → 23 of 26):** the best two settings of the last round as `config/v2/H1.yaml` and `H2.yaml`, written by `harness/hpo.write_configs` before any real-data run (V1b + the setting at the full budget with the last round's checkpoint spacing, anchor 0, no prior), trained at the protocol's 10 + 1 bp (owner decision), plus `H1nb.yaml` and `H2nb.yaml`, trained under `neo_broker` with `agent.env.cost_positions: 2` (the EUR 1 fee sized to a 2 × EUR 5,000 account, ≈ 2 bp, in training and in the decision band). They are reported against V4 with the §V8 rule (`scripts/analyze_v2.py --step H`). A winner with n = 40 would need P = 60 on real data; that is decided only if it happens (owner, 2026-10-07). **Chaining (owner decision 2026-10-07, 22:20):** the 4 trials start right after the search in the queue, but only if both finalists have a final score > 0 and n ≤ 20 (`harness/hpo.real_data_allowed`); otherwise they pause for the owner. |

---

## V13. References (retrieved and checked; details in `evidence-dqn-trading.md`)

**Data sufficiency and statistics**
- Bailey, Borwein, López de Prado & Zhu 2014, minimum backtest length, Notices AMS — https://carmamaths.org/jon/backtest.pdf
- Lo 2002, The Statistics of Sharpe Ratios, FAJ — https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios
- Israel, Kelly & Moskowitz 2020, Can Machines "Learn" Finance?, JOIM — https://papers.ssrn.com/abstract=3624052
- Kenneth French Data Library, industry portfolios (daily from 1926-07-01) — https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/Data_Library/det_5_ind_port.html
- FRED BAA10Y (daily from 1986-01-02) — https://fred.stlouisfed.org/series/BAA10Y

**Synthetic worlds and changepoints**
- Adams & MacKay 2007, Bayesian Online Changepoint Detection — https://lips.cs.princeton.edu/bibliography/adams2007changepoint
- Hamilton 1989, Econometrica 57(2):357–384, regime switching — https://ideas.repec.org/a/ecm/emetrp/v57y1989i2p357-84.html
- Wood, Roberts & Zohren 2021, Slow Momentum with Fast Reversion (online CPD module) — https://ideas.repec.org/p/arx/papers/2105.13727.html

**Live testing and implementation**
- Alpaca paper trading documentation — https://docs.alpaca.markets/docs/paper-trading
- European Parliament question E-004745/2021, US ETFs and PRIIPs — https://www.europarl.europa.eu/doceo/document/E-9-2021-004745_EN.html

**Method (as in v2.0)**
- Huang 2018 — https://arxiv.org/abs/1807.02787
- Sinclair et al. 2023 — https://proceedings.mlr.press/v202/sinclair23a.html
- Jacq et al. 2022 — https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2022/pdfs/p669.pdf
- Silver et al. 2018 — https://arxiv.org/abs/1812.06298
- Gârleanu & Pedersen 2013 — https://www.nber.org/papers/w15205
- Farebrother et al. 2024 — https://proceedings.mlr.press/v235/farebrother24a.html
- Gallici et al. 2025 — https://proceedings.iclr.cc/paper_files/paper/2025/hash/c23f3852601f6dd7f0b39223d031806f-Abstract-Conference.html
- Osband et al. 2016 (metadata only) — https://papers.nips.cc/paper/6501-deep-exploration-via-bootstrapped-dqn
- Anschel et al. 2017 (metadata only) — https://www.arxiv.org/abs/1611.01929
- Pitkäjärvi, Suominen & Vaittinen 2020 — https://tinbergen.nl/publication/168343/cross-asset-signals-and-time-series-momentum
- Moskowitz, Ooi & Pedersen 2012 — https://papers.ssrn.com/abstract=2089463
- Lim, Zohren & Roberts 2019 — https://arxiv.org/pdf/1904.04912
- Wood et al. 2021 — https://deepai.org/publication/trading-with-the-momentum-transformer-an-intelligent-and-interpretable-architecture
- Guijarro-Ordonez, Pelger & Zanotti 2021 — https://cdar.berkeley.edu/sites/default/files/deep_learning_statistical_arbitrage.pdf
- Schwarzer et al. 2021 (metadata only) — https://mlanthology.org/iclr/2021/schwarzer2021iclr-dataefficient
- Politis & Romano 1994 (metadata only) — https://gnosis.library.ucy.ac.cy/handle/7/57533
- Moreira & Muir 2017 — https://www.nber.org/papers/22208
- Cederburg et al. 2020 — https://www.lehigh.edu/~xuy219/research/COWY.pdf
- Ansari et al. 2024 — https://amazon.science/publications/chronos-learning-the-language-of-time-series
- Zhang et al. 2026 — https://pith.science/paper/2609.20554

---

## V14. Change log

| Date | Version | Change |
|---|---|---|
| 2026-10-06 | v2.0-draft | First draft from the v1.2 results. |
| 2026-10-06 | v2.1-draft | Added the data-sufficiency analysis and audit (§V2). Condensed the programme to 3 core bundles + 1 conditional step, with re-scoring and an internal ensemble (≤ 9 trials, −75 % to −87 % training runs). Added long-history pretraining, the lag-1 gate and M6 live forward test (§V10). Still awaits owner approval; no v2 run has started. |
| 2026-10-06 | v2.2-draft | Resolved the pre-freeze points (§V0): Q1 leak confirmed → clean reference R0′ (+1 trial, V1 vs R0′); H2 reserved for Track B (not run under v2, owner decision pending), drawdown hypothesis renamed H3 and tested against exposure-matched buy-and-hold; purge generalised to P = W + H_max = 40 for all v2 configs; synthetic worlds and BOCPD changepoint features fully specified; warm-up/availability rule for long-lookback features added. Budget ≤ 10 new trials (cumulative ≤ 26). No v2 run has started. |
| 2026-10-06 | **v2.0 APPROVED** | Owner approval with changes: early stopping removed; M6 deferred; Track B deferred to its own protocol; no automatic network/pushes (wandb online is the exception); Q2 not skipped; owner ideas on target exposure, per-transaction fees and price-level bias recorded in §V6.1. Merged into PROTOCOL.md as Part II and frozen before any v2 run. |
| 2026-10-06 | v2.0.1 | Implementation clarifications §V12.1: execution lag; G-lag and LC4 as differences of medians; diagnostic definitions; H3 benchmarks; vol-target baseline; pipeline v2 details; z-score window kept at 252 bars; v2 configs in `config/v2/`; Q2 second source dropped by owner decision; MinBTL for the validation length. Written after Steps 0a and 0b and before any v2 training run. No rule changed. |
| 2026-10-07 | v2.0.2 | Implementation clarifications §V12.1 items 17–20: synthetic worlds, calibration (W-vol GARCH β 0.90 → 0.91, the smallest change that meets the rule), V1 implementation, automatic stopping rule 1. Written before any synthetic agent run and before V1's trial. No rule changed. |
| 2026-10-07 | v2.0.3 | §V12.1 item 21: stopping rule 1 fired; how the synthetic iteration is run and logged (no trials), and that a real-data variant needs owner approval. Written while the screen runs, before its results. No rule changed. |
| 2026-10-07 | v2.0.4 | **Owner decisions:** (1) V1b (`config/v2/V1b.yaml`, = synthetic variant `v2_V1_mse_n20_h10g1`) is approved for Step 1 on real data (+1 trial). It is compared against R0′ with the §V8 adoption rule. The robustness caveat of the synthetic screen is noted in RESULTS. (2) Step 4 (long-history pretraining, §V6.4) runs next, before Steps 2/3. It is compared against its parent (V1b), so the pretraining effect is isolated. (3) A hyperparameter-optimisation strategy is planned only after V1b-real and Step 4. (4) Literature-based architecture, hyperparameter and network-size ideas are screened on synthetic data first (no trials). Longer targets (n = 40) would need P = 60 on real data, which is a protocol change still to be decided. |
| 2026-10-07 | v2.0.5 | §V12.1 item 22: Step 4 implementation (data, inputs incl. `vix_avail`, pretraining and fine-tuning budgets, reference V1b). Written before its run. No rule changed. |
| 2026-10-07 | v2.0.6 | **Owner decision:** §V12.1 item 23. The objective of 1–4-week timing at small costs; the synthetic world W-swing and its calibration (drift SD 1.5·10⁻³); a hyperparameter search on synthetic worlds only (no trials); and the 4 real-data trials it feeds (H1, H2, H1nb, H2nb; cumulative budget 23 of 26). Written before any search run. H1, its cost level, the folds and the test lock unchanged. |
