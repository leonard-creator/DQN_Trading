# Evidence log — Risk-aware cross-asset DQN trading agent

## 2026-10-02 — Literature review to strengthen the plan from the Gemini screenshots

**Channel:** web search, plus abstract pages fetched from arXiv, SSRN, NeurIPS, ScienceDirect and GitHub.

**Queries:** differential Sharpe ratio / Moody & Saffell; PBO and Deflated Sharpe; DRL-trading surveys and critiques; seed variance and multiplicity; Double DQN, PER, Rainbow; invalid action masking; TDQN and artificial trajectories; volatility-scaled DRL trading on futures; TimeGAN and Quant GANs; stationary bootstrap; survivorship bias; universal (cross-asset) models; CVaR distributional RL; yfinance and Alpha Vantage limits.

**How the sources were read:** Tier 2 means the abstract or landing page was read. Tier 1 means metadata only (title, venue, year); those rows are marked "metadata only".

### A. Reward design

| Paper | Year | Type | Finding (one line) | Link |
|---|---|---|---|---|
| Moody & Saffell, NeurIPS 11 | 1998 | primary | Introduces the **differential Sharpe ratio** as an online, risk-adjusted value function. Compares recurrent RL (immediate rewards) with Q-learning (discounted rewards) on S&P 500 data from 1970–1994. | https://papers.nips.cc/paper/1998/hash/4e6cd95227cb0c280e99a195be5f6615-Abstract.html |
| Moody & Saffell, IEEE TNN, "Learning to trade via direct reinforcement" | 2001 | primary | Follow-up journal paper; metadata only. | https://www.cs.utexas.edu/~shivaram/readings/b2hd-MoodySaffell2001.html |
| Zhang, Zohren & Roberts, arXiv (Oxford-Man) | 2019 | preprint | DQN, PG and A2C on 50 liquid futures (2011–2019). **Volatility scaling** of positions inside the reward. Beats time-series-momentum baselines and stays profitable with costs up to 25 bp. DQN performed best. | https://arxiv.org/abs/1911.10107 |
| Théate & Ernst, Expert Systems with Applications 173:114632 | 2021 | primary | TDQN maximises the Sharpe ratio. Training relies **entirely on artificial trajectories** generated from limited historical data. | https://arxiv.org/abs/2004.06627 |
| Lim & Malik, NeurIPS 35 | 2022 | primary | Standard distributional-RL action selection does not converge to **CVaR** objectives; the paper proposes a modified distributional Bellman operator. | https://proceedings.neurips.cc/paper_files/paper/2022/hash/c88a2bd0e793550d0e885aa6e31ca277-Abstract.html |

### B. Evaluation, overfitting and reproducibility

| Paper | Year | Type | Finding (one line) | Link |
|---|---|---|---|---|
| Bailey, Borwein, López de Prado & Zhu, J. Computational Finance | 2015 | primary | **Probability of Backtest Overfitting (PBO)** estimated with combinatorially symmetric cross-validation (CSCV). | https://papers.ssrn.com/abstract=2326253 |
| Bailey & López de Prado, J. Portfolio Management 40(5) | 2014 | primary | **Deflated Sharpe Ratio** corrects for multiple testing (selection bias) and non-normal returns. | https://papers.ssrn.com/abstract=2460551 |
| Arnott, Harvey & Markowitz, SSRN | 2018 | protocol paper | Backtesting protocol for ML in finance: small data, overfitting, selection bias, interpretability. | https://papers.ssrn.com/abstract=3275654 |
| Gort et al., arXiv | 2022 | preprint | Treats DRL backtest overfitting as a hypothesis test using **PBO** (α = 10%) over 50 hyperparameter trials. Tested on 10 cryptocurrencies at 5-minute frequency through two crashes. | https://arxiv.org/abs/2209.05559 |
| Henderson et al., AAAI | 2018 | primary | Deep RL results vary strongly with random seeds and implementation details; argues for multi-seed reporting. Metadata only. | https://arxiv.org/abs/1709.06560 |
| Grądzki, J. Finance and Data Science | 2026 | primary | 20 seeds × 3 settings. Sharpe ratios vary widely across seeds. **None of 10 pairwise DRL comparisons survived multiplicity correction.** Mean pairwise allocation distance between identically configured runs was 0.92. | https://doi.org/10.1016/j.jfds.2026.100205 |
| Brown, Goetzmann, Ibbotson & Ross, Review of Financial Studies 5(4) | 1992 | primary | **Survivorship bias** in truncated samples can create spurious predictability. | https://ideas.repec.org/a/oup/rfinst/v5y1992i4p553-80.html |

### C. DQN algorithm

| Paper | Year | Type | Finding (one line) | Link |
|---|---|---|---|---|
| van Hasselt, Guez & Silver, arXiv / AAAI | 2015 (arXiv) | primary | Double DQN reduces the overestimation bias of Q-learning. Metadata only. | https://arxiv.org/abs/1509.06461 |
| Schaul et al., arXiv | 2015/16 (arXiv) | primary | Prioritized Experience Replay. Metadata only. | https://arxiv.org/abs/1511.05952 |
| Hessel et al., AAAI | 2018 | primary | Rainbow combines Double, PER, dueling, n-step, distributional and noisy-net improvements. Metadata only. | https://arxiv.org/abs/1710.02298 |
| Huang & Ontañón, arXiv / FLAIRS | 2020 (arXiv) | primary | Analyses invalid action masking. Written for policy gradients, but the mechanics carry over to masking Q-values. Metadata only. | https://arxiv.org/abs/2006.14171 |

### D. Data, augmentation and cross-asset learning

| Paper | Year | Type | Finding (one line) | Link |
|---|---|---|---|---|
| Sirignano & Cont, arXiv | 2018 | preprint | A **universal model trained on all stocks** beats stock-specific models out of sample, including on stocks never seen in training. | https://arxiv.org/abs/1803.06917 |
| Politis & Romano, JASA | 1994 | primary | Stationary bootstrap: random block lengths keep the resampled series stationary. Metadata only. | https://gnosis.library.ucy.ac.cy/handle/7/57533 |
| Yoon, Jarrett & van der Schaar, NeurIPS | 2019 | primary | TimeGAN for synthetic time series. Metadata only. | https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks |
| Wiese et al., Quantitative Finance 20(9) | 2020 | primary | Quant GANs generate financial time series. Metadata only. | https://ideas.repec.org/a/taf/quantf/v20y2020i9p1419-1440.html |
| yfinance GitHub issue #2451 | 2025 | docs / issue | Yahoo limits 1m–90m bars to the **last 60 days** and 60m/1h bars to the **last 730 days**. | https://github.com/ranaroussi/yfinance/issues/2451 |
| Alpha Vantage support page | 2026 | vendor docs | Free tier allows **25 requests per day**. Realtime and 15-minute-delayed US data are premium. | https://www.alphavantage.co/support/#api-key |

### E. Surveys and frameworks

| Paper | Year | Type | Finding (one line) | Link |
|---|---|---|---|---|
| Hambly, Xu & Yang, Mathematical Finance 33(3) | 2023 | review | Broad survey of RL in finance. Metadata only. | https://arxiv.org/abs/2112.04553 |
| Millea, Data 6(11):119 | 2021 | review | Critical survey of DRL for trading. Metadata only. | https://ideas.repec.org/a/gam/jdataj/v6y2021i11p119-d680602.html |
| Liu et al., FinRL | 2021 | framework | Open-source DRL trading framework with environments and baselines. Metadata only. | https://arxiv.org/abs/2111.09395 |
| Korkmaz, arXiv | 2024 | review | Survey of generalization in deep RL. Metadata only. | https://arxiv.org/abs/2401.02349 |

**Conclusion:** The plan's components (risk-adjusted reward, Double DQN, PER, Huber loss, stationary features, cross-asset pooling) are supported by the literature. The weakest part is **evaluation**. Multi-seed, multiplicity-aware and overfitting-aware evaluation (PBO, Deflated Sharpe, survivorship-free universe) is missing from the Gemini plan. Recent evidence (Grądzki 2026) shows that this is where DRL trading claims usually fail.

**Open questions:**
- Does the differential Sharpe ratio beat the mean-variance step reward with Q-learning on H4 equity data? No head-to-head comparison was found.
- How much does augmentation (bootstrap or GAN) help a DQN on top of cross-asset pooling?

**Not found / could not verify:** a DOI for Moody & Saffell 2001 (only a metadata page); full texts. The Quant GANs abstract fetch timed out. The exact Alpha Vantage rules for free *historical* intraday data are ambiguous on the support page.

---

## 2026-10-05 — Guijarro-Ordonez, Pelger & Zanotti, "Deep Learning Statistical Arbitrage" (requested by the project owner)

**Channel:** web fetch (Crossref API for the published record; arXiv + ar5iv for the open full text). The INFORMS page returned HTTP 403 (paywall/bot block); the Berkeley CDAR PDF copy was unreachable (DNS).
**Queries:** DOI 10.1287/mnsc.2022.03132; "Deep Learning Statistical Arbitrage" Guijarro-Ordonez Pelger Zanotti.
**Read level:** Tier 3 (full text, targeted at methods / results / costs) of **arXiv v2 (2022-10-07)**. The published version's full text was NOT read. Its abstract (from Crossref) words the results more cautiously ("considerable ... outperform" vs the preprint's "consistently high ... substantially outperform"), so the published numbers may differ from those below.

| Paper | Year | Type | Finding (one line) | Link |
|---|---|---|---|---|
| Guijarro-Ordonez, Pelger & Zanotti, *Management Science* 72(9):7502–7549 | 2026 (issue Sept 2026; Crossref gives no online-first date) | primary | Trade **factor residuals**, not price levels: residual portfolios from conditional latent factors (IPCA), signals from a **convolutional transformer** over 30-day cumulative residuals, allocation by a network trained **end-to-end to maximise Sharpe**. Daily U.S. large caps, out of sample 2002–2016: Sharpe 4.16 (IPCA-5) vs 0.97 for an Ornstein-Uhlenbeck threshold model and 1.90 for a Fourier filter [arXiv v2 numbers] | https://doi.org/10.1287/mnsc.2022.03132 · preprint https://arxiv.org/abs/2106.04028 |

**Details extracted (arXiv v2; unverified against the published version)**

- **Data:** ~550 largest, most liquid U.S. stocks (market cap > 0.01 % of total), CRSP daily returns, 46 firm characteristics. Trading evaluated Jan 2002 – Dec 2016.
- **Arbitrage portfolios:** out-of-sample residuals from Fama-French (1–8 factors, 60-day loadings), PCA (1–15 factors, 252-day correlation, 60-day loadings) or IPCA (1–15 conditional factors, re-estimated yearly). Residuals are cumulated into a price-like path over a **lookback L = 30 days** (robust to L = 60).
- **Signal extractor:** 2 causal-style conv layers (8 filters, size 2, instance norm, residual connections), then a transformer (4 heads); the last time step's projection is the signal.
- **Policy:** a feed-forward net maps signals to weights. Weights are normalised to **‖w‖₁ = 1** (long-short, bounded leverage). Signal and allocation are trained **jointly** on the Sharpe ratio (mean-variance is an alternative). This is supervised end-to-end optimisation, **not** reinforcement learning.
- **Training:** rolling 1,000-day estimation window, network re-estimated every 125 days, hyperparameters from validation periods. **No multiple random seeds or ensembles are reported** (our reading).
- **Key results:**
  - Residuals matter. Trading raw returns (K = 0) gives Sharpe 1.64, "substantially worse than any type of residual".
  - The choice of factor model has only a minor effect (Sharpe 2.5–4.2 across models and factor counts).
- **Ablation:** "trading signal extraction is the most challenging and separating element". The conv-transformer doubles performance vs a fixed Fourier filter, while a flexible allocation function adds only minor gains over a simple parametric rule. Generic, non-temporal neural nets do substantially worse.
- **Costs:** Sharpe 4.16 → 4.01 at 2 bp → 3.79 at 5 bp. Results at our primary 10 bp level were not found. Turnover levels were not found in the parts read.
- **Horizon:** most mispricing is corrected within ~1 month; about half of the Sharpe survives a 1-week holding period.
- **Factor exposure:** alpha vs the Fama-French 8-factor model 8.3 %/yr (t ≈ 16), R² ≈ 4 %, i.e. close to market-neutral.

**Relevance to this project**

1. **Confirms:** a temporal encoder over a short daily window, a risk-adjusted objective trained directly on returns (same idea as `diff_sharpe` / Moody & Saffell), bounded leverage, and evaluation net of costs.
2. **Challenges H1's design:** the paper's edge comes from predicting **relative** moves (residuals across a large cross-section), and it states that the level of returns is "extremely hard to predict". H1 asks a long-only agent to time the **level** of 26 broad ETFs against buy-and-hold. That is the hard problem the paper avoids. M1 agrees: no timing baseline beats buy-and-hold at 10 bp.
3. **Not directly transferable:**
   - 550 single stocks give many idiosyncratic bets; 26 broad ETFs (sector / country / asset-class baskets) give few, and their residuals are not idiosyncratic risk.
   - CRSP data are paid.
   - Costs were tested only up to 5 bp.
   - The sample ends in 2016.
   - There is no seed-variance or multiple-testing analysis comparable to PROTOCOL §8.

**Conclusion:** strong evidence (one paper, peer-reviewed, preprint numbers) that relative-value / residual signals plus a learned temporal filter beat level-timing. It supports keeping a time-series encoder, and it gives a concrete fallback design if H1 fails (see `docs/03_extracted_plan.md` §7).

**Open questions:**
- Do ETF residuals (vs universe PCA or SPY) mean-revert enough to survive 10 bp + spread?
- Does Q-learning add anything over the paper's end-to-end Sharpe policy, given its finding that the allocation step adds little?

**Not found / could not verify:**
- the published version's full text and final numbers;
- turnover levels;
- results at ≥ 10 bp;
- the online-first publication date (the owner described it as published last year; Crossref lists only the September 2026 issue).

---

## 2026-10-06 — How can the DQN agent beat buy-and-hold after the v1 results (M1–M4)?

**Channel:** web search, plus abstract and landing pages fetched from PMLR, ICLR, AAAI, AAMAS, NBER, SSRN, Google Research, DeepAI, CDAR and pith.science.

**Queries:** Exo-MDP hindsight learning; action augmentation in trading; value learning as classification (HL-Gauss); PQN and LayerNorm; volatility-managed portfolios and their critique; residual policy learning; lazy-MDPs; TempoRL; Gârleanu–Pedersen aim portfolio; time-series momentum; cross-asset signals; Deep Momentum Networks; Momentum Transformer; IQN; Averaged-DQN; Bootstrapped DQN; Munchausen RL; DQfD; iRDPG; SPR; BBF; primacy bias; Chronos; look-ahead bias in pretrained models; deep-learning statistical arbitrage; recent DQN-trading papers from 2024–2026.

| Paper | Year | Type | Finding (one line) | Link |
|---|---|---|---|---|
| Huang, arXiv | 2018 | preprint | **Action augmentation**: rewards of unchosen actions are computable under zero market impact, so Q-values for all actions are updated at once. +6.4 %/yr on average vs ε-greedy on 12 FX pairs. | https://arxiv.org/abs/1807.02787 |
| Sinclair et al., ICML (PMLR 202) | 2023 | primary | **Hindsight Learning** for MDPs with exogenous inputs: counterfactuals from logged exogenous data. Beats heuristics and standard RL in resource-allocation tasks. | https://proceedings.mlr.press/v202/sinclair23a.html |
| Jacq, Ferret, Pietquin & Geist, AAMAS | 2022 | primary | **Lazy-MDPs**: the agent defers to a default policy and pays a penalty η for taking control. It learns *when* to act. | https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2022/pdfs/p669.pdf |
| Silver, Allen, Tenenbaum & Kaelbling, arXiv | 2018 | preprint | **Residual Policy Learning**: RL learns corrections to an existing controller and is much more data-efficient than learning from scratch. | https://arxiv.org/abs/1812.06298 |
| Gârleanu & Pedersen, J. Finance 68(6) | 2013 | primary | With costs: "aim in front of the target" and "trade partially towards the current aim". | https://www.nber.org/papers/w15205 |
| Biedenkapp et al., ICML | 2021 | primary | **TempoRL**: a skip-policy learns how long to repeat an action, up to an order of magnitude faster than vanilla Q-learning. | https://arxiv.org/abs/2106.05262 |
| Farebrother et al., ICML (PMLR 235) | 2024 | primary | Value functions trained with **categorical cross-entropy** instead of MSE; mitigates noisy targets and non-stationarity. | https://proceedings.mlr.press/v235/farebrother24a.html |
| Gallici et al., ICLR | 2025 | primary | **LayerNorm** gives provably convergent TD learning without a target network or replay buffer (PQN). | https://proceedings.iclr.cc/paper_files/paper/2025/hash/c23f3852601f6dd7f0b39223d031806f-Abstract-Conference.html |
| Osband et al., NeurIPS | 2016 | primary | Bootstrapped DQN. Metadata only. | https://papers.nips.cc/paper/6501-deep-exploration-via-bootstrapped-dqn |
| Anschel, Baram & Shimkin, ICML | 2017 | primary | Averaged-DQN for variance reduction. Metadata only; the abstract fetch was rate-limited (HTTP 429). | https://www.arxiv.org/abs/1611.01929 |
| Vieillard, Pietquin & Geist, NeurIPS | 2020 | primary | **Munchausen RL**: a scaled log-policy added to the reward gives implicit KL regularisation and a larger action gap. | https://research.google/pubs/munchausen-reinforcement-learning/ |
| Dabney et al., ICML (PMLR 80) | 2018 | primary | IQN: distributional DQN that enables a large class of risk-sensitive policies. | https://proceedings.mlr.press/v80/dabney18a.html |
| Moreira & Muir, J. Finance 72(4) | 2017 | primary | Taking less risk when volatility is high raises factor and market Sharpe ratios (in-sample spanning). | https://www.nber.org/papers/22208 |
| Cederburg, O'Doherty, Wang & Yan, JFE | 2020 | primary | Out of sample, vol management wins 53 vs 50 times across 103 strategies. Real-time combinations often underperform; the market factor's Sharpe goes from 0.46 to 0.42. | https://www.lehigh.edu/~xuy219/research/COWY.pdf |
| Moskowitz, Ooi & Pedersen, SSRN / JFE | 2011/2012 | primary | Time-series momentum over 1–12 months in 58 futures, with partial reversal at longer horizons. | https://papers.ssrn.com/abstract=2089463 |
| Pitkäjärvi, Suominen & Vaittinen, JFE 136(1) | 2020 | primary | Bond returns predict equities positively and equities predict bonds negatively. Sharpe 45 % above standard TSMOM. | https://tinbergen.nl/publication/168343/cross-asset-signals-and-time-series-momentum |
| Lim, Zohren & Roberts, arXiv | 2019 | preprint | Deep Momentum Networks: Sharpe-optimised trend and position sizing with turnover regularisation; still outperform at 2–3 bp costs. | https://arxiv.org/pdf/1904.04912 |
| Wood et al., arXiv | 2021 | preprint | Momentum Transformer: changepoint detection at multiple timescales complements attention. | https://deepai.org/publication/trading-with-the-momentum-transformer-an-intelligent-and-interpretable-architecture |
| Guijarro-Ordonez, Pelger & Zanotti, working paper | 2021 | preprint | Signal extraction (CNN + Transformer) is the "most challenging and separating element"; a flexible allocation can't compensate for a weak signal. | https://cdar.berkeley.edu/sites/default/files/deep_learning_statistical_arbitrage.pdf |
| Hester et al., AAAI | 2018 | primary | **DQfD**: TD updates plus supervised classification of demonstrator actions; better early learning on 41 of 42 Atari games. | https://ojs.aaai.org/index.php/AAAI/article/view/11757 |
| Liu et al., AAAI | 2020 | primary | iRDPG: imitation of classical trading strategies inside DRL (POMDP formulation). | https://ojs.aaai.org/index.php/AAAI/article/view/5587 |
| Schwarzer et al., ICLR | 2021 | primary | SPR: self-predictive representations for data-efficient RL. Metadata only. | https://mlanthology.org/iclr/2021/schwarzer2021iclr-dataefficient |
| Schwarzer et al., ICML (PMLR 202) | 2023 | primary | BBF: scaled value networks for Atari 100K. | https://proceedings.mlr.press/v202/schwarzer23a.html |
| Nikishin et al., ICML (PMLR 162) | 2022 | primary | Primacy bias and resets. Metadata only. | https://proceedings.mlr.press/v162/nikishin22a |
| Ansari et al., TMLR | 2024 | primary | Chronos: pretrained on public datasets plus synthetic Gaussian-process data. | https://amazon.science/publications/chronos-learning-the-language-of-time-series |
| Zhang, Huang, Chen, Chen & Chen, arXiv 2609.20554 | 2026 | preprint | Look-ahead bias in pretrained financial forecasters: later-trained models were *worse* than point-in-time ones in 18 of 20 US model-horizon cases. Exposure to future data is still an information-set violation. | https://pith.science/paper/2609.20554 |

**Conclusion:**
- Methods that fit the v1 diagnosis (no gross timing skill, cost-driven losses, seed noise as large as the effects) and stay inside DQN:
  1. exogenous / counterfactual replay with a cost-structured value head, which yields a no-trade band;
  2. default-anchored ("lazy" / residual) action spaces;
  3. ensemble-gated switching;
  4. noise-robust value losses (cross-entropy, LayerNorm);
  5. slower cross-asset inputs (bond↔equity momentum, multi-horizon trend, changepoints).
- The literature supports each mechanism, but **none has been shown to beat buy-and-hold on diversified ETFs under a multi-seed, multiplicity-aware protocol.** The recent DQN-trading papers I found (2024–2026) report mostly single-split results.

**Open questions:**
- Does vol-timing survive costs and real-time estimation on these 26 ETFs? (Cederburg et al. suggest probably only weakly.)
- Is there any learnable daily or weekly timing signal at all? The synthetic positive controls in NEW_PROTOCOL §4.1 are designed to answer this.

**Not found / could not verify:**
- Averaged-DQN abstract (rate-limited).
- A ResearchGate page on time-series foundation models in finance (HTTP 429).
- The HL-Gauss specifics are not in the Farebrother et al. abstract; only categorical cross-entropy is confirmed.
- The exact input set of Deep Momentum Networks was not verified.

**Further references cited in `PROTOCOL.md` Part II §V13** (added by the owner; listed here so that every citation in the protocol has an entry):

| Source | Year | Type | Used for | Link |
|---|---|---|---|---|
| Bailey, Borwein, López de Prado & Zhu, Notices of the AMS | 2014 | primary | Minimum backtest length (MinBTL), §V2.2 | https://carmamaths.org/jon/backtest.pdf |
| Lo, Financial Analysts Journal | 2002 | primary | The statistics of Sharpe ratios; serial correlation, §V2.2 | https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios |
| Israel, Kelly & Moskowitz, JOIM | 2020 | primary | Can machines "learn" finance? Small-data limits | https://papers.ssrn.com/abstract=3624052 |
| Kenneth French Data Library | – | data | Daily industry portfolios from 1926-07-01, Step 4 | https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/Data_Library/det_5_ind_port.html |
| FRED, series BAA10Y | – | data | Optional credit-spread input, daily from 1986-01-02 | https://fred.stlouisfed.org/series/BAA10Y |
| Adams & MacKay | 2007 | preprint | Bayesian online changepoint detection, §V6.3 | https://lips.cs.princeton.edu/bibliography/adams2007changepoint |
| Hamilton, Econometrica 57(2):357–384 | 1989 | primary | Regime switching, synthetic world W-regime, §V6.5 | https://ideas.repec.org/a/ecm/emetrp/v57y1989i2p357-84.html |
| Wood, Roberts & Zohren | 2021 | preprint | Slow momentum with fast reversion (online CPD module) | https://ideas.repec.org/p/arx/papers/2105.13727.html |
| Alpaca documentation | – | vendor docs | Paper trading (M6, deferred) | https://docs.alpaca.markets/docs/paper-trading |
| European Parliament question E-004745/2021 | 2021 | official | US-domiciled ETFs and PRIIPs for EU retail investors (M6, deferred) | https://www.europarl.europa.eu/doceo/document/E-9-2021-004745_EN.html |

*Provenance note (2026-10-06):* this entry and the table above were researched and verified by the project owner (source file `_templates/Evidence_for_research.md`) and copied here unchanged, without re-verification, at the owner's instruction.
