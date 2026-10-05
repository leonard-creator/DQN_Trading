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
