# Plan v2: Risk-Aware Cross-Asset DQN Trading Agent

**Version:** v2, 2026-10-02. This replaces v1, which only extracted the plan from the screenshots.
**Inputs:** the Gemini conversation of 9–10 July 2026 (transcript: `01_stitched_text.md`) and a literature review (`evidence-dqn-trading.md`).
**Status:** I wrote this without seeing the code base. The line numbers (15, 24/25, 34) come from the screenshots.

**How to read the tags in this document**

- **[G]** = taken from the Gemini conversation
- **[R]** = changed or added after review
- **[L]** = supported by a retrieved source (see the reference list at the end)

---

## 0. Project aim, sharpened

**Original aim (implicit in [G]):** "turn a toy DQN into a robust, market-ready algorithm."

**Revised aim [R]:** build a DQN agent trained on many assets, and test with a pre-registered, multi-seed, overfitting-aware protocol whether it delivers **net-of-cost, risk-adjusted performance that beats simple baselines out of sample**. The research claim must survive:

- seed variation,
- correction for multiple testing,
- a time split the agent never saw during development.

Why the aim had to change: most published DRL trading gains shrink once you account for random seeds and for how many configurations were tried. In a 20-seed study, **none of 10 pairwise DRL algorithm comparisons stayed significant after multiplicity correction**, and identically configured runs ended up with very different portfolios [L: Grądzki 2026]. Deep RL results generally swing with seeds and implementation details [L: Henderson 2018]. Finance also offers little data and many chances to overfit [L: Arnott, Harvey & Markowitz 2018].

> **Analogy:** treat this like a clinical trial. The agent is the drug, buy-and-hold is the placebo arm, seeds are patients, and the hold-out period is the confirmation cohort. If the protocol is written after you have seen the results, the trial doesn't count.

---

## 1. Starting point (the user's code)

| Aspect | Before | Diagnosis |
|---|---|---|
| Reward | Absolute profit at sell time | Ignores variance [G] |
| Loss | `loss="mse"` (line 34) | Fat tails make gradients unstable [G] |
| Conv encoder | `GlobalAveragePooling1D()` (line 15) | Discards the order of time steps [G] |
| Inventory | Late fusion: `pos_in` + `Concatenate` (lines 24/25) | **Correct, keep it** [G] |
| Replay | batch 2048, buffer 100–200k | Oversized for one asset on daily data [G] |
| Training | Plateau at ~800 episodes, then a collapse | Too much LR or target drift [G], **or overfitting to the training period** [R]. Watching validation will tell which. |
| Data | 8 years of daily data, one asset, prices as features | Non-stationary and narrow [G] |

---

## 2. Weaknesses in the Gemini plan, and the fixes

| # | Weakness | Why it matters | Fix |
|---|---|---|---|
| W1 | **No evaluation protocol**: one run, no seeds, no correction for multiple tests | DRL trading claims usually fail exactly here [L: Grądzki 2026; Henderson 2018] | Section 3: ≥10 seeds, walk-forward with purge/embargo, PBO, Deflated Sharpe, frozen test period |
| W2 | **The "DSR" acronym is ambiguous**: *Differential* Sharpe Ratio (a reward) [L: Moody & Saffell 1998] vs *Deflated* Sharpe Ratio (an evaluation statistic) [L: Bailey & López de Prado 2014] | Mixing them up leads to wrong code and wrong reports | Name them `diff_sharpe` (reward) and `deflated_sharpe` (metric) everywhere |
| W3 | **The shuffled master tensor breaks DQN** | DQN needs transitions (s, a, r, s′) produced by stepping forward. Shuffling windows destroys the link from s to s′. The replay buffer already decorrelates samples. | A multi-asset environment: each episode draws a random ticker and start date within the training period |
| W4 | **"2500 non-overlapping samples" is wrong** | Stride-1 windows overlap by W−1 steps, which overstates the effective sample size and leaks across split boundaries | Purge W bars plus the reward horizon at every split boundary, and add an embargo |
| W5 | **Survivorship and selection bias**: AAPL, TSLA and MSFT were picked with hindsight | Picking today's winners creates spurious predictability [L: Brown et al. 1992] | Define the universe as of each period's start (e.g. index members at start), or use a broad list of liquid ETFs and futures chosen before seeing any results |
| W6 | **Wrong H4 arithmetic for US stocks**: "4 bars per day, 20 bars = 1 week" | A US regular session lasts 6.5 h, so it doesn't split into four 4-hour bars. That arithmetic only holds for roughly 16 h/day markets. | Use session-aligned bars (e.g. 1h, or 2 bars per session) or 24h markets (FX, futures). Recompute W from that. |
| W7 | **The data source can't supply the history** | yfinance limits 1h data to the **last 730 days** [L: yfinance #2451]. Alpha Vantage's free tier allows **25 requests/day** [L: Alpha Vantage docs]. So 8 years of H4 equity bars can't come from these. | Dukascopy (indices/FX), a paid equities vendor, or daily data for the long history plus intraday data only for recent validation |
| W8 | **Look-ahead through context features** | A daily VIX close joined to intraday bars of the same day uses future information | Join as-of with a lag: use only VIX values known before the bar closes |
| W9 | **The mean-variance step reward penalises upside as much as downside**, and its scale interacts with Huber δ | δ = 1.0 only makes sense if TD errors are near unit scale. With raw PnL, Huber behaves like pure L1 or pure L2. | Volatility-scale positions or returns [L: Zhang et al. 2019]. Normalise the reward. Make the Differential Sharpe reward an equal candidate [L: Moody & Saffell 1998]. Treat a downside-risk variant as optional. |
| W10 | **The exponential holding penalty is ad hoc** | It can suppress trend-following, which is the main source of edge in Zhang et al. 2019 [L] | Start with a small linear or zero penalty and let transaction costs discourage churn. Tune the penalty only with validation-based model selection. |
| W11 | **No baselines** | Without baselines you can't tell skill from market drift | Buy-and-hold, random agent, sign-of-past-return momentum, MACD crossover [L: Zhang et al. 2019 uses this kind of baseline set], all net of costs |
| W12 | **Hyperparameter tuning isn't counted as multiple testing** | Every config you try inflates the best Sharpe [L: Bailey & López de Prado 2014] | Log every trial. Report N_trials and the Deflated Sharpe. Estimate PBO with CSCV [L: Bailey et al. 2015; Gort et al. 2022]. |
| W13 | **The buffer-size advice contradicts itself** (≤50k, later 100–250k) | Confusing | Make it a config value with default 100k (with PER) and tune it on validation |
| W14 | **Augmentation is presented as harmless** | Augmenting validation or test data, or blocks that cross split boundaries, leaks information | Augment only the training period. Prefer the **stationary bootstrap** (random block lengths) [L: Politis & Romano 1994]. TDQN trains on artificial trajectories [L: Théate & Ernst 2021]. Keep GANs for later [L: Yoon et al. 2019; Wiese et al. 2020]. |
| W15 | **Risk is controlled only through the reward** | A mean-variance penalty doesn't control tail risk | Optional: distributional DQN (QR/C51) with CVaR-based action selection, using the corrected operator from Lim & Malik 2022 [L] |

---

## 3. Evaluation protocol (new, and mandatory) [R]

1. **Freeze a final test period** (e.g. the last 12–18 months). Don't touch it until the very end, and use it only once.
2. **Development with walk-forward:** rolling train → validation folds. At each boundary, purge (window length plus the reward horizon) and add an embargo.
3. **Seeds:** at least 10 (preferably 20) independent seeds per configuration. Report the median and IQR of Sharpe, return, max drawdown, turnover and holding time, plus the full distribution [L: Grądzki 2026; Henderson 2018].
4. **Overfitting statistics:**
   - **PBO via CSCV** across all configurations tried [L: Bailey et al. 2015; applied to DRL in Gort et al. 2022].
   - **Deflated Sharpe Ratio** for the selected configuration, using N_trials and the skewness and kurtosis of returns [L: Bailey & López de Prado 2014].
5. **Comparisons:** paired tests over seeds and folds (e.g. a permutation test) against each baseline, with family-wise correction (Holm).
6. **Costs:** run every result at 0, 5, 10 and 25 bp plus a spread model. Zhang et al. report robustness up to 25 bp on futures [L].
7. **Pre-registration:** write a `PROTOCOL.md` with hypotheses, metrics, the trial budget and the test date *before* the first training run [L: Arnott, Harvey & Markowitz 2018].

**Primary hypothesis H1:** the cross-asset DQN with a risk-adjusted reward has a higher net-of-cost Sharpe ratio than buy-and-hold and than momentum, over seeds and walk-forward folds. It must reach a Deflated Sharpe Ratio > 0.95 and PBO < 0.5.

---

## 4. Implementation plan v2 (phases)

### Phase A: Model and algorithm [G + R]

1. Replace `GlobalAveragePooling1D` with `Flatten` in the conv branch [G]. Keep the late-fusion two-input model [G].
2. Use Huber loss (δ configurable, default 1.0) **after reward normalisation** [G + R].
3. Use Double DQN targets [G; L: van Hasselt et al.].
4. Mask actions **on the Q-values**, both at ε-greedy selection and inside the target's argmax and max [G + R; L: Huang & Ontañón].
5. Add an LR schedule, and soft target updates (τ) or hard updates every K steps [G].
6. Optional "Rainbow-lite" upgrades, one at a time, each ablated: dueling head, n-step returns (n = 3–5), PER [G; L: Schaul et al.; Hessel et al.].
7. Optional: a distributional head (QR-DQN) to enable CVaR action selection [R; L: Lim & Malik 2022].

### Phase B: Reward [G + R]

1. A pluggable `reward_type`:
   - `mean_variance` [G]: ΔPnL − (λ/2)·ΔPnL² − ρ(h) − c
   - `diff_sharpe` [L: Moody & Saffell 1998]: an online update from EMA estimates of the first and second moments (parameter η)
   - `vol_scaled_pnl` [L: Zhang et al. 2019]: position × return / σ_target-scaled, minus costs
2. Normalise rewards (a running scale or vol-scaling) so Huber δ is meaningful.
3. Default holding penalty: none or small linear. The exponential form is an ablation, not the default.
4. Log each reward component separately.

### Phase C: Data and features [G + R]

1. **Universe:** choose it up front and without survivorship bias (W5). Document the choice in `PROTOCOL.md`.
2. **Frequency:** check what the data actually covers (W6, W7). Options: daily for 8+ years, or session-aligned intraday bars from a vendor with deep history.
3. **Features [G]:** log return, P/SMA − 1, Bollinger %B, RSI, normalised MACD histogram, relative volume, NATR, rolling σ of returns, lagged VIX, sin/cos of time-of-day (only if intraday). Optional: day-of-week encoding and a 50-period volatility "beta proxy".
4. **Normalisation:** rolling z-scores using only past data [G]. Unit-test for no look-ahead [R].
5. Context columns are joined **as-of with a lag** (W8).
6. Features are computed **per ticker**, then stored for lookup [G]. They are **not** shuffled into one training tensor (W3).

### Phase D: Cross-asset training [G + R]

1. Pooling assets is supported: a universal model trained on many stocks beat stock-specific models, even on stocks it had never seen [L: Sirignano & Cont 2018].
2. **Multi-asset environment:** each episode samples a ticker (uniformly, or weighted by data length) and a start index inside the current training fold, then runs for a fixed horizon [R].
3. Optional asset embedding or asset-class one-hot. Ablate it against "no ID", which forces pure generalisation [R].
4. **Leave-assets-out test:** hold out 20% of tickers entirely and evaluate on them [R; motivated by Sirignano & Cont 2018].

### Phase E: Augmentation (optional, training data only) [G + R]

1. Stationary bootstrap of contiguous return blocks within the training fold [L: Politis & Romano 1994].
2. Small Gaussian jitter on the stationary features [G].
3. GAN-based synthetic paths only after 1–2 show value [G; L: Yoon et al. 2019; Wiese et al. 2020].

### Phase F: Hyperparameters (starting values, all tuned on validation) [G]

| Parameter | Start | Range to search |
|---|---|---|
| Window W | 20 bars | 10–60 |
| Batch | 256 | 128–512 |
| Replay buffer | 100k | 50k–250k |
| Huber δ | 1.0 (on normalised rewards) | 0.5–2 |
| γ | 0.99 | 0.9–0.99 |
| LR | 1e-4 with decay | 3e-5 – 3e-4 |
| τ (soft update) | 0.005 | 0.001–0.01 |
| λ (risk aversion) | 0.5 | 0–2 |
| Costs | 10 bp | 0–25 bp |

Keep the trial budget small and fixed in advance (e.g. ≤ 50 configurations), and count every trial (W12).

---

## 5. Milestones and kill criteria [R]

| M | Deliverable | Go / no-go |
|---|---|---|
| M1 | Fixed model (Flatten, Huber, DDQN, masking) + tests, single asset | Training is stable over seeds, with no collapse at ~800 episodes |
| M2 | Evaluation harness (walk-forward, seeds, baselines, PBO, Deflated Sharpe) | The harness reproduces baseline numbers |
| M3 | Reward variants compared on validation | At least one reward beats buy-and-hold on median validation Sharpe, net of costs |
| M4 | Multi-asset environment + leave-assets-out test | Cross-asset ≥ single-asset on held-out assets |
| M5 | One look at the frozen test period | H1 passes, or the project reports a well-documented negative result |

A clean negative result is a valid scientific outcome. Don't loosen the protocol to get a positive one.

---

## 6. References (retrieved and verified 2026-10-02; full table in `evidence-dqn-trading.md`)

- Moody & Saffell, NeurIPS 1998 — https://papers.nips.cc/paper/1998/hash/4e6cd95227cb0c280e99a195be5f6615-Abstract.html
- Zhang, Zohren & Roberts, arXiv 2019 [preprint] — https://arxiv.org/abs/1911.10107
- Théate & Ernst, Expert Syst. Appl. 2021 — https://arxiv.org/abs/2004.06627
- Lim & Malik, NeurIPS 2022 — https://proceedings.neurips.cc/paper_files/paper/2022/hash/c88a2bd0e793550d0e885aa6e31ca277-Abstract.html
- Bailey et al., J. Comput. Finance 2015 (PBO) — https://papers.ssrn.com/abstract=2326253
- Bailey & López de Prado, J. Portf. Manag. 2014 (Deflated Sharpe) — https://papers.ssrn.com/abstract=2460551
- Arnott, Harvey & Markowitz, SSRN 2018 — https://papers.ssrn.com/abstract=3275654
- Gort et al., arXiv 2022 [preprint] — https://arxiv.org/abs/2209.05559
- Henderson et al., 2018 — https://arxiv.org/abs/1709.06560
- Grądzki, J. Finance Data Sci. 2026 — https://doi.org/10.1016/j.jfds.2026.100205
- Brown et al., Rev. Financ. Stud. 1992 — https://ideas.repec.org/a/oup/rfinst/v5y1992i4p553-80.html
- van Hasselt et al. (Double DQN) — https://arxiv.org/abs/1509.06461
- Schaul et al. (PER) — https://arxiv.org/abs/1511.05952
- Hessel et al., AAAI 2018 (Rainbow) — https://arxiv.org/abs/1710.02298
- Huang & Ontañón (action masking) — https://arxiv.org/abs/2006.14171
- Sirignano & Cont, arXiv 2018 [preprint] — https://arxiv.org/abs/1803.06917
- Politis & Romano, JASA 1994 — https://gnosis.library.ucy.ac.cy/handle/7/57533
- Yoon et al., NeurIPS 2019 (TimeGAN) — https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks
- Wiese et al., Quant. Finance 2020 (Quant GANs) — https://ideas.repec.org/a/taf/quantf/v20y2020i9p1419-1440.html
- yfinance issue #2451 — https://github.com/ranaroussi/yfinance/issues/2451
- Alpha Vantage support — https://www.alphavantage.co/support/#api-key
- Guijarro-Ordonez, Pelger & Zanotti, Management Science 72(9) 2026 (Deep Learning Statistical Arbitrage) — https://doi.org/10.1287/mnsc.2022.03132 (added 2026-10-05, see §7)

*Research code only; this is not investment advice.*

---

## 7. Addendum (2026-10-05): lessons from "Deep Learning Statistical Arbitrage" and a fallback if H1 fails

**Source:** Guijarro-Ordonez, Pelger & Zanotti, *Management Science* 72(9), 2026 (https://doi.org/10.1287/mnsc.2022.03132). The full text read was arXiv v2 (2022). Details, caveats and what could not be verified are in `evidence-dqn-trading.md` (2026-10-05 entry).

**Ground rule.** H1 and PROTOCOL v1.2 are **not** changed by this addendum. Changing a pre-registered hypothesis after seeing results is the failure mode the protocol exists to prevent. The paper is used in two legitimate ways: (A) to choose development configurations *before* their results are seen, within the 50-trial budget; and (B) to pre-register an *additional* hypothesis H2, only with the owner's approval and only **before M5**.

### 7.1 What the paper implies for this project

| Paper finding | Implication here | Tag |
|---|---|---|
| Return **levels** are "extremely hard to predict"; trading raw returns gave Sharpe 1.64 vs 2.5–4.2 for factor **residuals** | H1 asks the agent to time the level of broad ETFs. M1 already shows no timing rule beats buy-and-hold at 10 bp. Expect H1 to be hard. | [L] |
| Signal extraction dominates; the allocation function adds little | The temporal encoder deserves more attention than RL extras (dueling / PER / n-step). The DQN's main job is cost-aware position control | [L] |
| Conv + transformer doubles performance vs a fixed Fourier filter; generic nets do worse | Worth one encoder ablation in M4 | [L] |
| End-to-end Sharpe objective, ‖w‖₁ = 1, long-short | Supports `diff_sharpe` as the reward; a long-short, market-neutral design needs `allow_short` and a borrow-cost model | [L] |
| Half of the Sharpe survives a 1-week hold; lookback 30 days | Our window W = 20 and daily decisions are in the right range | [L] |
| Costs tested only up to 5 bp (Sharpe 4.16 → 3.79); large-cap single stocks; sample ends 2016; no seed analysis | Their edge at our 10 bp + spread is unknown, and with 26 ETFs there are far fewer independent residual bets. Do not expect Sharpe ratios of that size | [R] |

### 7.2 Track A: inside the current protocol (agent improvements, decided before results)

These keep H1 unchanged. Each costs one trial and is fixed in a config **before** its validation results are seen:

- **A1, M4 features: residual signals.** For each ticker, the daily return minus its exposure to the first k principal components of the universe. Estimated on trailing windows only (PCA on 252 days, loadings on 60), cumulated over L = 30 days, added as extra input channels. The directional agent can then see whether an ETF is cheap or rich **relative** to the rest. A look-ahead unit test is required, as for every feature.
- **A2, M4 encoder ablation:** `network.arch: conv_transformer` (2 conv layers with 8 filters of size 2, then 4-head attention, as in the paper) vs the current conv-Flatten encoder.
- **A3, diagnostic (no extra trial):** report how much of the agent's net Sharpe comes from *exposure timing* vs *being long*, by comparing with the random agent at matched exposure. This tests the paper's "allocation adds little" point on our data.

### 7.3 Track B: contingency hypothesis H2, if H1 is likely to fail

Trigger: after M3 or M4, if the development results show no configuration with a median validation Sharpe above buy-and-hold net of costs (the M3 kill criterion in §5).

- **H2 (draft, not approved):** a **market-neutral residual strategy** on the training-universe ETFs has a positive net Sharpe with deflated Sharpe > 0.95. It must also beat two baselines that stay on the same residuals: (i) an Ornstein-Uhlenbeck threshold rule (the paper's parametric benchmark) and (ii) a 1-week residual reversal rule. Costs: the same 10 bp + 1 bp half-spread, plus a borrow fee on short positions (value to be fixed at pre-registration).
- **Design:**
  - long-short weights with ‖w‖₁ = 1 across tickers (a portfolio, not per-ticker sleeves);
  - residuals as in A1;
  - policy = the DQN with `allow_short` and residual inputs, compared with the paper's end-to-end Sharpe-maximising network as a non-RL alternative.
- **Test period:** the frozen test can be opened **once**. H2 must therefore be pre-registered (PROTOCOL v1.3) **before M5** and evaluated in the **same** `final_test.py` run as H1, with Holm correction across {H1, H2}. The alternative is a forward test on data after 2026-09-30, which needs months of new data.
- **Budget:** H2 development configurations count in the same trial log, so they raise N_trials for H1's deflated Sharpe as well. Keep H2 to a few configurations.
- **Known risk:** 26 broad ETFs carry little idiosyncratic risk, so residual mean-reversion may be too weak to survive 10 bp. A larger, survivorship-free universe of liquid ETFs would help, but needs a PROTOCOL change and new data.

**Decision needed from the owner (no later than before M5):** pre-register H2 or not. Until then Track B is a plan only; no H2 code or trials will be run.
