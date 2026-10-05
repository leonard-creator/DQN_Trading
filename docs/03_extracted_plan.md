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

*Research code only; this is not investment advice.*
