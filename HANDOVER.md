# HANDOVER — DQN_Trading, state of 2026-10-15 ~12:15

This file hands the project to a new agent. Read it first. Then read, in this order:
1. `PROTOCOL.md` Part II: the binding research rules (v2.0 → v2.0.8; §V12.1 holds the implementation clarifications 1–24, the change log is §V14).
2. `RESULTS.md`: every result, newest at the end (Step H, the AHL screen, the real-data H trials, the euro illustration).
3. `README.md`: status and the change report (newest entry first).
4. `EXPLANATIONS.md`: plain-language explanations and the glossary of the winning setting (§5).
5. The memory files in `~/.claude/projects/-raid-users-tilingl/memory/` (owner rules).

---

## 1. Working rules set by the owner (must be followed)

- **Scope:**
  - Work only inside `/raid/users/tilingl/Documents/ML/DQN_Trading`. No sudo.
  - Do not use `/tmp` for project work. Use the git-ignored `.dev_v2/` (dev copy of the code, `logs/`, `scratch/`).
- **Resources:** never more than 90 % of CPU, RAM or GPU (40 cores, 247 GB, 4× V100). The job pool caps itself at 12 workers × 3 threads.
- **Environment:** conda env `dqn_ml`. Tests: `CUDA_VISIBLE_DEVICES="" python -m pytest -o addopts="" -q` (135 passed on 2026-10-08).
- **Network:**
  - No automatic downloads, platform pulls or git pushes by code. The owner runs downloads (`scripts/download_external.py`).
  - Exception: wandb logs online for real-data training.
  - Literature search on request is fine; never download files to the server.
- **Git:**
  - The owner commits; Claude only drafts commit messages.
  - The owner prefers **results-only commits** (RESULTS.md, reports) separate from code commits.
  - Never commit `data/raw`, `experiments/runs/`, models, graphs or the legacy `train_data/`/`test_data/`.
- **Documentation:**
  - Every change goes into the README change report. Results go into RESULTS.md, rule changes into PROTOCOL.md (§V12.1 item + §V14 row, written **before** the run).
  - `EXPLANATIONS.md` is updated **only for new terms or on the owner's request**.
  - **`HANDOVER.md` is updated only when the owner commands it**, built from the other .md files and change logs.
- **Decisions:**
  - Ask the owner before anything that spends a trial or changes the protocol.
  - **Before the final M5 test: ask the owner again and show the overview of the steps still missing** (owner, 2026-10-15).
- **Long GPU runs:** always detached through the queue (§7), so they survive chat disconnects.
- **Mindset:** every analysis also looks for ways to improve the data and the agent; measurable profit in a later live test is the goal. Never mention optional MCP connectors to the owner.
- **Style:**
  - Reuse the harness and scripts. Code must be short, safe and commented; new options are off by default and unit-tested.
  - One-off analyses go to `.dev_v2/scratch/`.
  - Messages to the owner: short, factual, research terms, numbers next to the baseline.
- **Owner's goal (2026-10-07):** an agent that times markets on a **1–4-week horizon** net of small costs (or the `neo_broker` costs), not one that holds for years. H1 stays the confirmatory test and is never changed.

---

## 2. Research question and protocol in one paragraph

H1: does the DQN-type agent beat buy-and-hold (B&H) net of costs over 10 seeds × 5 walk-forward folds?
- **Data:** 26 training ETFs (^GDAXI included), ^VIX as context, 7 leave-out ETFs (QQQ, XLE, XLF, EEM, EWZ, **IEF**, SLV).
- **Periods:** validation 2014-01 → 2023-09; frozen test period 2023-10 → 2026-09, **untouched and locked** (`scripts/final_test.py --i-am-sure`, only for M5).
- **Every real-data configuration is a trial** (`experiments/trials.csv`). **Budget ≤ 26; used 24; 2 left.**
- **Statistics:** deflated Sharpe, PBO, paired sign-flip tests with Holm.
- **Costs:** 10 bp + 1 bp half-spread (H1). Secondary scenario `neo_broker`: EUR 1 per order on EUR 10k, 3 bp half-spread.
- **Synthetic worlds** (`harness/synthetic.py`) are never trials:
  - W-null: no timing exists;
  - W-vol: de-risk in high volatility;
  - W-regime: bull/bear;
  - W-swing: 1–4-week trends per ticker.

**Trials:**
- 1–16 = v1 (M2–M4);
- 17 R0′;
- 18 V1b;
- 19 V4;
- 20–23 H1, H2, H1nb, H2nb;
- 24 V3.

**Adopted reference: V4** (0.67 = B&H).

---

## 3. Results so far (median Sharpe at 10 bp on the 26 ETFs; B&H = 0.67)

| Step | What | Result |
|---|---|---|
| M1–M4 (v1) | baselines, DQN fixes, rewards, cross-asset conv-transformer | best v1 0.44; no timing skill; losses from overtrading |
| 0a/0b/0d | re-scoring, data audit, clean reference R0′ | timing IC ≈ 0 everywhere; data clean (Q1 leak fixed); R0′ 0.43 |
| 0c + iteration | synthetic positive controls, 13 V1 variants | V1b (MSE + 20-step targets + 10 gated heads) is the only variant to pass W-null and W-regime (weak: 16 % of the oracle over 10 seeds) |
| Step 1 / Step 4 | V1b (trial 18), V4 = V1b + 1926–2007 pretraining (trial 19) | 0.64 / **0.67**; both ≈ B&H (97 % invested, timing IC 0); V4 adopted |
| Owner-idea screens | 40-day targets, 4× wider network (synthetic) | 40-day ≈ V1b; wide = 2× gross timing but noise trading |
| **Step H search** | 32 settings, successive halving, no buy-and-hold head start, 5 bp, W-null/W-regime/W-swing (synthetic) | winner **c00** (HL-Gauss, 5-day targets, γ 0.99, 4× network, 10 heads gate z 1, lr 1e-4): 77 % / 60 % of the oracle's gain |
| AHL screen | adaptive HL-Gauss range (Chen et al. 2025) vs fixed, c00/c19/c25, seeds 20–29 | AHL not adopted; c00 with the fixed range is the most robust |
| **H trials (real data)** | H1 = c00, H2 = c25 + neo-broker twins (trials 20–23) | **0.49 / 0.43 / 0.36 / 0.27**, none adopted; timing IC ≈ 0.005; below B&H even at 0 bp; G-lag fails for H1 |
| Euro view | EUR 10k in SPY + EFA, neo-broker, 2014 → 2023-09 | B&H EUR 19,568; H1 EUR 14,594; H2 EUR 12,754 (RESULTS.md "In euros") |
| **Realism check** (2026-10-15, no trials) | real ETFs mean-revert (VR(60) 0.66 ex-crisis), commodities trend; W-revert world; estimated contrarian rule on real data | c00 learns 82 % of the W-revert oracle; the estimated rule on real data loses even gross (0.51 vs 0.67) → **the data, not the learner, is the limit** |
| **Step 3: V3** (trial 24) | V4 + signal pack (D1 context, D2 trend + BOCPD, D3 auxiliary heads) | **0.34**, not adopted; 17 of 50 runs trade and lose −0.32; gross 0.54 < 0.67 → **stopping rule 2 reached** |

Reports: `experiments/reports/` (`V2_0a_rescore.md`, `V2_0c.md`, `V2_0d.md`, `V2_1.md`, `V2_4.md`, `V2_hpo.md`, `V2_H.md`), plus `DATA_AUDIT.md` at the root. Literature: `docs/evidence-dqn-trading.md`.

## 4. Key learnings

1. **The v1 failure was the learning machinery.** It overtraded noise and could not learn timing even where it provably exists (synthetic worlds).
2. **The V1 decision structure stops noise trading:**
   - target exposures, an exact cost band, exogenous replay, gated heads;
   - with the buy-and-hold head start (anchor + prior) the agent converges to B&H on real data, which is the best safe policy it can justify.
3. **What makes timing learnable on synthetic data:**
   - HL-Gauss loss, short n-step targets, a slow learning rate, an ensemble of 10 heads with a strict gate, the 4× network;
   - MSE and lr 3e-4 drift into noise trading with longer training;
   - random priors hurt; weight decay has no effect (RESULTS "Which ingredients pay off").
4. **The synthetic skill does not transfer to the real ETFs.**
   - Real ETFs **mean-revert** at 1–4 weeks: variance ratio 0.87 / 0.82 / 0.83 at 5/10/20 days on development data.
   - W-swing trends (1.24), so the search optimised for the wrong structure. This fits M1, where no momentum rule beat B&H.
5. **Without the buy-and-hold head start, the agent acts on noise:**
   - real-data action gaps are 0.10 of the TD noise;
   - the early folds (models trained on 6–10 years) trade weekly and lose;
   - later folds mostly hold.
6. **Cheaper training costs → more trading → worse at 10 bp** (neo-broker twins).
7. **Selection effects are large:** seed spread on real data is about ±0.5 Sharpe, and single agents can look good by luck (1 of 20 winner agents beat B&H in the euro view). The hard −0.05 W-null cut-off flips borderline settings, so report the loss continuously.
8. **Engineering pitfalls already solved:**
   - spawn-based worker pools need an `if __name__ == "__main__":` guard in scripts;
   - soft target updates must skip frozen weights (bit-exactness);
   - the queue imports code at each line's start, so never edit `harness/`, `rl/` or `scripts/` while it runs.

---

## 5. Current plan (owner command 2026-10-15) — status

1. ~~Realism check~~ — **done** (PROTOCOL item 25; RESULTS "v2 realism check").
2. ~~V3 signal pack~~ — **done** (item 26, trial 24; RESULTS "v2 Step 3"). Not adopted; **stopping rule 2 is reached**.
3. **Wrap-up toward M5:** written (RESULTS "v2 wrap-up toward M5").
   - §V8 selection: V4, validation median 0.6673 = buy-and-hold 0.6673; it passes G-lag and is identical to buy-and-hold in 35 of 50 runs.
   - The protocol recommends **not** spending the test period.
   - **Waiting for the owner's decision on M5** (ask before anything touches the test period; show the overview of missing steps): Step 3w, ablations, Track B/H2 in its own protocol, a risk-control objective (H3), M6.

## 6. How to run things

- **Real-data trial:** `python -u scripts/run_agent.py --config config/v2/<X>.yaml --workers 12` (logged in `experiments/trials.csv`, wandb online).
- **Report:** `python -u scripts/analyze_v2.py --step <label> --trials <names> --reference v2_V4 --workers 12 --threads 2`.
- **Synthetic screen** (no trial):
  - `--synthetic --worlds ... --seeds ... --set key=value`;
  - search: `--hpo`;
  - follow-up screens: `--hpo-screen NAME` (defined in `harness/hpo.SCREENS`);
  - calibration: `--calibrate`.
- **Smoke run** (mechanics only, nothing logged, no outer numbers): `--smoke --smoke-transitions 8000`.
- **Long-history pretraining:** `--pretrain` (Step 4).

## 7. Queue

- **Files:** queue `experiments/v2_queue.txt` (74 lines, all done; idle since 2026-10-15 11:50), runner `scripts/run_queue.sh`, log `experiments/v2_queue.log`, position `experiments/v2_queue.pos`.
- **Start:** `setsid nohup scripts/run_queue.sh experiments/v2_queue.txt > /dev/null 2>&1 < /dev/null &`. It resumes after the last finished line.
- **Rules:** append lines only, never edit lines before the position. Check with `grep "end   line" experiments/v2_queue.log | tail`.

## 8. Git state

- **Committed by the owner:** `e02a0e3`, `010733b`, `4269a9a` (2026-10-08: Steps 1 and 4, the screens, search tooling, W-swing, search results, the H configs).
- **Uncommitted** (2026-10-08 → 2026-10-15, incl. the realism check and V3):
  - code: AHL (`rl/exogenous.py`, `harness/hpo.py`, `scripts/run_agent.py`, tests);
  - docs: PROTOCOL items 24 + v2.0.7/v2.0.8; RESULTS (Step H, AHL screen, real-data H trials, euro view); README; EXPLANATIONS (glossary); CODEBASE_NOTES; this HANDOVER;
  - results: `experiments/trials.csv`, `experiments/v2_queue.txt`, `experiments/reports/V2_hpo.md`, `V2_H.md`, `v2_H/`, `v2_hpo/screen_support.csv`, `runs_screen_support.csv`.
- **Untracked owner files:** `NEW_PROTOCOL.md`, `_templates/`, `docs/4535_Adaptive_HL_Gaussian_A_Va.pdf` (suggested: keep the PDF out of git).

## 9. File map (most used)

- `harness/`:
  - `experiment.py` (`run_experiment`, `prices=` for synthetic);
  - `backtest.py` (costs, scenarios);
  - `splits.py`, `data.py`, `trials.py`, `stats.py`, `metrics.py`;
  - `report.py` (shared tables);
  - `diagnostics.py` (§V7);
  - `synthetic.py` (worlds, oracles, `run_many`, calibration);
  - `hpo.py` (search, screens, `write_configs`, guard);
  - `longhistory.py` (Step 4).
- `rl/`:
  - `policy.py` (jobs, `train_jobs` pool, `run_job`, `cost_function` incl. `cost_positions`);
  - `exogenous.py` (V1 learner: U-head, exogenous replay, anchor, heads + gate, n-step, HL-Gauss, AHL, random priors, AdamW, `agent.prior`);
  - `features_m4.py` (pipeline v2, flags `resid_avail` / `vix_avail`);
  - `networks.py` (conv-transformer);
  - `diagnostics.py` (LC5/LC7);
  - `trainer.py` / `agent.py` / `env.py` (old environment-loop DQN).
- `scripts/`: `run_agent.py`, `analyze_v2.py`, `rescore_v2.py`, `audit_data.py`, `run_queue.sh`, `download_external.py`, `final_test.py` (M5 only).
- `config/v2/`: `R0prime`, `V1`, `V1b`, `V4`, `H1`, `H2`, `H1nb`, `H2nb`.
