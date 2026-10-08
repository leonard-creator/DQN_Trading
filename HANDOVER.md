# HANDOVER — DQN_Trading, state of 2026-10-07 ~10:50

This file hands the project to a new chat. Read it first. Then read, in this order:
1. `PROTOCOL.md` Part II: the binding research rules (v2.0–v2.0.5, §V12.1 holds the implementation clarifications 1–22).
2. `RESULTS.md`: every result.
3. `EXPLANATIONS.md`: the key decisions in plain words.
4. `README.md`: status and change report.
5. The memory files in `~/.claude/projects/-raid-users-tilingl/memory/`.

---

## 1. Working rules set by the owner (must be followed)

- **Scope:**
  - Work only inside `/raid/users/tilingl/Documents/ML/DQN_Trading`.
  - Do not use `/tmp` for project work. Use project-local folders (e.g. `.dev_v2/`, which is git-ignored).
  - Not sudo.
- **Resources:** never use more than 90 % of CPU, RAM or GPU (40 cores, 4× V100). Check before heavy jobs.
- **Environment:** conda env `dqn_ml`.
- **Network:**
  - No automatic network access, downloads, platform pulls or git pushes by code. The owner runs downloads (`scripts/download_external.py`).
  - Exception: wandb logs **online** for real-data training runs.
  - Literature search by Claude on request is fine; never download files to the server.
- **Git:** the owner commits. Claude only drafts commit messages and never commits or pushes. Never commit `data/raw`, `experiments/runs/`, models or graphs.
- **Style:**
  - Reuse the harness and scripts; avoid single-use files.
  - Code must be short, safe and clearly commented.
  - Document every change in the README change report.
  - Keep `EXPLANATIONS.md` (plain language) updated with every crucial decision.
- **Decisions:** ask the owner before critical decisions (anything that spends a trial or changes the protocol).
- **Long GPU runs:** always detached, through the queue (see §6), so they survive chat disconnects.
- **Mindset:**
  - Every analysis should also look for ways to improve the agent; a measurable gain in the later live test is the goal.
  - Never mention optional MCP connectors to the owner.
- **Owner's goal for the agent (2026-10-07, important):**
  - Holding one position for years is **not** what the agent should learn.
  - It should decide on a **1–4 week horizon** and place trades more often where that adds value net of costs and shows an actual timing learning instead of buy and hold.
  - Costs are small or as in the `neo_broker` scenario.
  - H1 (beat buy-and-hold net of costs) stays the confirmatory test and is never changed.

---

## 2. Research question and protocol in one paragraph

Does a DQN-type agent beat buy-and-hold (B&H) net of costs, over 10 seeds × 5 walk-forward folds?
- **Data:** 26 training ETFs plus ^GDAXI, ^VIX as context, 7 leave-out ETFs.
- **Periods:** validation 2014-01 → 2023-09; frozen test period 2023-10 → 2026-09, **untouched and locked**.
- **Pre-registration:** H1 is fixed in `PROTOCOL.md`. Every configuration evaluated on real validation data is a **trial**, logged in `experiments/trials.csv`; the budget is ≤ 26.
- **Statistics:** deflated Sharpe, PBO, paired sign-flip tests with Holm.
- **Costs:** primary 10 bp + 1 bp half-spread. The secondary scenario `neo_broker` (`config/cost_scenarios.yaml`) has a fixed fee of 1 unit per transaction on a 10,000-unit account, a 3 bp half-spread, and a TER on ^GDAXI only.
- **Trials used: 19 of 26.**

---

## 3. Results so far (numbers: median Sharpe at 10 bp on the 26 ETFs)

| Step | What | Result |
|---|---|---|
| M1–M4 (v1) | baselines, DQN fixes, rewards, cross-asset + conv-transformer | best v1 = 0.44, B&H = 0.67; no timing skill; losses come from overtrading |
| 0a | re-scoring of 16 trials | timing IC ≈ 0 everywhere; action gaps smaller than the TD noise (noise-driven decisions) |
| 0b | data audit | data clean; ^GDAXI close-time leak (residual features) found and fixed (pipeline v2); about 2 effective assets, about 6 bear markets |
| 0d | R0′ = clean reference | 0.43 (trial 17) |
| 0c | synthetic worlds W-null / W-vol / W-regime (known oracle) | R0′ fails all three; V1 passes W-null, fails the others → stopping rule 1 → synthetic iteration |
| iteration | 13 V1 variants (no trials) | only **V1b** = V1 + MSE + 20-step hold targets + 10 bootstrapped heads with gate z = 1 passes W-null and W-regime (+0.26 = 72 % of the oracle, seeds 0–4). Robustness: seeds 5–9 only +0.05; pooled 10 seeds +0.06 (16 %) |
| Step 1 | V1b on real data (trial 18) | **0.64**, adopted over R0′ (+0.12, p < 0.001); but ≈ B&H: 97 % invested, 0.5 turns/yr, timing IC 0 |
| Step 4 | V4 = V1b + pretraining on French 1926–2007 (trial 19) | **0.67** = B&H; adopted (seed IQR −91 %); still no timing |

**Key learnings**
1. The v1 failure was the learning machinery: it overtraded noise and could not learn timing even where the timing is known to exist.
2. V1's exact cost head and gate stopped the noise trading. But the learned value gaps are too small and noisy, so on real data the agent converges to holding.
3. Huber loss (δ = 1) slowed the learning of small mean gaps. MSE + 20-step targets fixed this on synthetic data. Gated heads cut the noise trades.
4. The learner is short of *events*. More history (Step 4) made it consistent, but not better at timing.
5. W-vol is unsolved, partly by design: a risk-neutral reward plus the anchor hurdle.

All reports are in `experiments/reports/` (`V2_0a_rescore.md`, `DATA_AUDIT.md` at the root, `V2_0c.md`, `V2_0d.md`, `V2_1.md`, `V2_4.md`). The literature is in `docs/evidence-dqn-trading.md` (section "2026-10-07 — Tricks …").

---

## 4. The approved next plan (owner, 2026-10-07)

**Status 2026-10-07 22:15:**
- §4.3 is implemented and tested (suite: 131 passed). The W-swing drift SD is calibrated to 1.5·10⁻³ (oracle +0.32 at 5 bp, 17 turns/yr).
- PROTOCOL §V12.1 item 23 / v2.0.6 and the EXPLANATIONS entry are written. `harness/hpo.write_configs()` writes H1, H2, H1nb and H2nb from the final round.
- **Owner approved item 23 (22:20); the chain is queued** (queue lines 50–58, started 22:19):
  1. the search (`--hpo`, ≈ 3.5 h; log `experiments/runs/v2_hpo.log`, results `experiments/reports/v2_hpo/` + `V2_hpo.md`);
  2. `write_configs()` (H1, H2, H1nb, H2nb in `config/v2/`);
  3. the 4 real-data trials and the step-H report (`V2_H.md`), each guarded by `hpo.real_data_allowed()`: both finalists need a final score > 0 and n ≤ 20, else they pause for the owner.
- Owner decisions of 22:00: H1/H2 train at 10 + 1 bp; if a finalist has n = 40, ask the owner about P = 60 before its real-data run.
- **While the queue runs, do not edit `harness/`, `rl/` or `scripts/` in place.** Each round and each trial imports the code at its start. Develop in `.dev_v2/` (copy the repo code there first).
- Next for Claude: when the report exists, write up RESULTS (new step "H"), README, EXPLANATIONS and the improvement ideas; give the owner a commit draft.

### 4.1 Hyperparameter optimisation (HPO) on synthetic worlds only (no trials)

- **Objective (new owner goal):** learn timing on **1–4 week horizons** at **small costs**.
  - Score = the mean over W-swing and W-regime of the median (over seeds) share of the oracle's net gain the agent captures.
  - Penalty if the median gain in W-null is below −0.05 (losses from noise trading).
  - Drop W-vol from the objective.
- **New synthetic world W-swing** (to add to `harness/synthetic.py`; a protocol amendment is needed):
  - Each ticker has its own latent drift d_t, AR(1) with a half-life of about 10 days, around a positive mean drift μ̄ (B&H Sharpe ≈ 0.6). Common shocks as in the other worlds (ρ = 0.48).
  - The oracle runs a scalar Kalman filter per ticker on its own returns (true parameters): long if the predicted next-day return beats a cost band b ≈ cost × (1 − φ), else flat, with hysteresis.
  - Calibrate the drift SD so that the oracle's net gain is ≥ 0.15 at the HPO cost and its turnover is about 12–50/yr (1–4 week holds).
- **Common settings for all candidates:**
  - V1b's structure (exogenous U-head, exact cost band).
  - **No buy-and-hold bias:** `agent.anchor_eta: 0` and a new option `agent.prior: none` (no B&H output-bias start).
  - HPO costs ≈ 5 bp per unit traded (`evaluation.costs.primary_bps: 2`, `half_spread_bps: 3`).
- **Search space:**

| Setting | Options |
|---|---|
| loss | mse, hl_gauss |
| n_step | 5, 10, 20, 40 |
| gamma | 0.9, 0.95, 0.99 |
| width | ×1 (`transformer_dim` 8, `hidden` [64, 32]) or ×4 (32, [256, 128]) |
| weight_decay (AdamW) | 0, 0.1 |
| heads | 1, 10, 20 |
| prior_scale (random priors) | 0, 3 |
| gate_z | 0, 0.5, 1 |
| lr | 1e-4, 3e-4 |

- **Successive halving:**
  1. 32 random settings × 3 worlds × seeds 0–4 at ¼ budget → keep 8.
  2. 8 × seeds 0–9 at ½ budget → keep 2.
  3. 2 × fresh seeds 10–19 at full budget (confirmation).

  Scale `eval_every_updates` so each run keeps about 5 checkpoints. Expected cost: about 3–3.5 GPU-h with 10 workers.
- **Engineering:**
  - Cache the M4 features per (world, seed); they do not depend on the agent config.
  - Train all candidates of a round in **one** `train_jobs` pool (`synthetic.run_many`).
  - Log every candidate (overrides + scores per round) to `experiments/reports/v2_hpo/`.

### 4.2 Real data after the HPO (owner: the best **two**, not one, to see the scoring variance)

- Write the two winners as `config/v2/H1.yaml`, `H2.yaml` (inherit V1b; set the winning overrides, anchor 0, prior none).
- Write neo-broker-trained variants `H1nb.yaml`, `H2nb.yaml` (`agent.env.cost_scenario: neo_broker`).
  - Add a new env option `cost_positions: 2`. Then the per-transaction fee is sized to the owner's deployment account (2 positions of 5,000 units → about 2 bp per trade), not split over the 26 training sleeves (which would be about 26 bp per trade).
  - Implement it in `rl/policy.run_job`: `train_costs = [cost_of(t, a["env"].get("cost_positions") or len(train_t)) …]`.
- **4 trials → N_trials 23 of 26.** Report all four against V4 with `scripts/analyze_v2.py --step H --trials v2_H1 v2_H2 v2_H1nb v2_H2nb --reference v2_V4`.
- Owner: "just test the best two without overanalysing it".
- **Before running:**
  - Add PROTOCOL Part II §V12.1 item 23 (owner objective, W-swing, HPO plan, 4 real-data trials) and a change-log row v2.0.6.
  - Add an EXPLANATIONS.md entry in plain words.

### 4.3 Code still to implement (all options default OFF, unit-tested, short)

- `rl/exogenous.py`:
  - (a) `algo.loss: hl_gauss`:
    - The network outputs heads × E × `bins` (default 51) logits; U = Σ softmax · bin centres.
    - Loss = cross-entropy against the HL-Gauss target, σ = 0.75 × bin width.
    - Support = [min_e mean(imm_e)/(1−γⁿ) − 6·SD(imm), max_e … + 6·SD(imm)], computed from the training samples.
    - **Store the support in `info.json`** and pass it in `rl/diagnostics.py` when rebuilding UAgent.
  - (b) `algo.prior_scale` β: a frozen random prior network added inside the Keras model (`base(m) + β·prior(m)`, prior `trainable=False`), so the saved weights include it. Apply the B&H bias to the base's last layer.
  - (c) `algo.weight_decay` → `keras.optimizers.AdamW`.
  - (d) `agent.prior: none` → no bias start.
- `rl/policy.py`: the `cost_positions` env option (see 4.2).
- `harness/synthetic.py`:
  - add the W-swing world (`simulate` + Kalman oracle);
  - add `run_many(configs, …)` with the feature cache (keep `run()` as a wrapper);
  - keep calibration (`calibrate()`) working for W-swing.
- `harness/hpo.py` (new, small): the driver from §4.1. Plus `scripts/run_agent.py --hpo`.
- Tests in the existing test files (`tests/test_exogenous.py`, `tests/test_synthetic.py`). The full suite currently has **119 passed**.
- Development tip: the old chat used `.dev_v2/` (a git-ignored copy of the code inside the project, currently identical to the repo code) so that queued runs keep the tested code. Develop there and copy files back when no queue job is about to start. Or develop in place after the queue is idle.

---

## 5. Queue state (updated 2026-10-07 22:20)

- **Queue** `experiments/v2_queue.txt`, runner `scripts/run_queue.sh`, log `experiments/v2_queue.log`, position file `experiments/v2_queue.pos`.
- **Running since 22:19:** lines 50–58 (see §4 status): the search, then the guarded real-data chain. Measured pace in round 1: ~150 s per ¼-budget job, 12 at a time. Expected: search ≈ 02:30, the 4 trials ≈ 05:30–06:30, report shortly after.
- **Earlier:** lines 46–48 (synthetic screens `v2_V1b_n40`, `v2_V1b_wide`, `v2_V1b_wide_n40`, seeds 0–9) finished with exit 0 at 11:40, about 25 min each.
- **Results:** written up in RESULTS.md ("v2 synthetic screen — the owner's ideas"), README and EXPLANATIONS. 40-day targets ≈ V1b; the 4× wider network has twice the gross timing but trades noise (W-null −0.13).
- **Check:** `grep "end   line" experiments/v2_queue.log | tail` and `ps -ef | grep run_queue`.
- **Queue rules:**
  - Lines may be **appended** while the queue runs.
  - Never insert or delete lines before the current position.
  - If the queue has exited, restart it with `setsid nohup scripts/run_queue.sh experiments/v2_queue.txt > /dev/null 2>&1 < /dev/null &`; it resumes after the last finished line.

## 6. Git state

- Committed by the owner: `e02a0e3`, then `010733b` (2026-10-07 09:11: Steps 0c/0d, V1/V1b, shared report code, queue).
- Uncommitted (state 21:45): the tracked files changed after 09:11 and the new files of 2026-10-07, e.g. `EXPLANATIONS.md`, `HANDOVER.md`, `harness/longhistory.py`, `config/v2/V4.yaml`, `experiments/reports/V2_1.md`, `V2_4.md`, `v2_1/`, `v2_4/`.
- Give the owner a commit draft with `git add -u` plus an explicit `git add` of the new files. The owner commits; never commit yourself.

## 7. File map (most used)

- `harness/`:
  - `experiment.py` (`run_experiment`, `prices=` for synthetic);
  - `backtest.py`;
  - `report.py` (shared report tables);
  - `diagnostics.py` (§V7 scoring);
  - `synthetic.py` (0c worlds);
  - `longhistory.py` (Step 4);
  - `splits.py`, `data.py`, `trials.py`, `stats.py`.
- `rl/`:
  - `policy.py` (jobs, `train_jobs` pool, `run_job`);
  - `exogenous.py` (V1/V1b/V4 learner);
  - `trainer.py` + `agent.py` + `env.py` (old env-loop DQN);
  - `networks.py`;
  - `features_m4.py` (pipeline v2, flags `resid_avail` / `vix_avail`);
  - `diagnostics.py` (LC5/LC7).
- `scripts/`:
  - `run_agent.py` (`--synthetic`, `--calibrate`, `--pretrain`, `--set k=v`, `--worlds`, `--seeds`, `--smoke`);
  - `analyze_v2.py` (generic step report with the §V8 rule);
  - `rescore_v2.py`, `audit_data.py`, `run_queue.sh`.
- `config/v2/`: `R0prime`, `V1`, `V1b`, `V4`.

## 8. Note on the old chat

Keep messages short and factual, and describe the work in research terms (backtests, decision horizons, cost scenarios).
