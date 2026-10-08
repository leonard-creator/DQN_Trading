# Explanations — the key experiments and design decisions in plain words

This file explains the decisions that shaped the agent, in simple words with examples. It is meant to be read without the code. Numbers come from [`RESULTS.md`](RESULTS.md) and the reports in `experiments/reports/`; the binding rules are in [`PROTOCOL.md`](PROTOCOL.md). A new entry is added for every crucial experiment or design decision.

---

## 1. The road so far: v1 milestones (very brief)

- **M1 (baselines):** buy-and-hold (B&H) sets the bar: Sharpe 0.67 on the 26 training ETFs after 10 bp costs.
- **M2 (model fixes on ^GDAXI):** a standard modern DQN (Double DQN, Huber loss, action masking) trains stably, but it trades about 30 times a year without any timing skill.
- **M3 (reward designs):** no reward beats B&H. Before costs the agent is about as good as B&H, so the whole gap is trading costs.
- **M4 (26 ETFs, new features, conv-transformer):** the best v1 agent reaches 0.44. It gets there by trading less and staying invested, not by timing.
- **Neo-broker costs (€1 per trade):** same picture.

## 2. v2 Step 0: diagnosing why the agent fails (brief)

- **0a, re-scoring all 16 trials:** no agent shows timing skill (timing IC ≈ 0). The agents' value estimates for hold, buy and sell differ by less than the noise in their own training signal, so their choices are decided by noise.
- **0b, data audit:** the data is clean. One leak was found and fixed: ^GDAXI's residual feature used US prices from later the same day. The agents had never used it. The data is thin in what matters: 26 ETFs behave like about 2 independent assets, with only about 6 bear markets in 16 years.
- **0d, clean reference R0′:** the best v1 agent re-run with the fix scores 0.43 (against 0.44 before). This is the reference that every v2 agent must beat.

---

## 3. The synthetic test worlds and the variant exploration (2026-10-07)

*Written as explained to the owner on 2026-10-07, copied here as it was.*

### Why synthetic data and not our training sets

- **Every variant judged on real data counts as a trial.** 13 variants would have used the entire remaining budget (17 → 30 > 26). The best of 26 random strategies already shows a Sharpe of about 0.65 by pure luck, so a screen like this on real data would amount to overfitting.
- **Real data can't separate "no signal" from "can't learn".** A synthetic world has a known oracle: we know how much profit is available, so a failure clearly means the agent can't learn it.
- **The training sets can't judge variants either.** In-sample, the agent fits noise. The one-year inner-validation slice has an error of about ±1 Sharpe, far too noisy to rank variants.
- **We can generate fresh worlds on demand.** That is how the robustness check on seeds 5–9 exposed the weak result. History gives us only one past.
- The caveat: synthetic worlds are simplified, so the winner still has to prove itself on real data, as one trial.

### The variant exploration in simple words

The three test worlds:
- **W-null:** a market where timing cannot help, so the right answer is "just hold".
- **W-vol:** it pays to reduce exposure when markets get stormy.
- **W-regime:** calm bulls at +15%/yr alternate with bears at −20%/yr. The oracle steps aside in bears and roughly doubles the Sharpe ratio.

**Old architecture (R0′).** Actions are buy or sell ¼, it learns from its own trades, with Huber loss and one-day targets. It fails everywhere. Even in W-null it trades 11 times a year. Example from one W-regime world (seed 3):

| Strategy | Sharpe | Trades per year |
|---|---|---|
| Buy-and-hold | 0.50 | – |
| Oracle | 1.43 | – |
| R0′ | **−0.21** | 39 |

**V1 (new structure).** It picks a target exposure directly (0–100%), trading costs are built into the decision as a no-trade band, it learns from every ticker and day, and it defaults to buy-and-hold. The random trading stops (0.1 trades per year), but it never steps aside. A probe showed why: it learned the right direction, but its estimated value gap between 100% and 0% was **10× too small** (0.1–0.26 where about 2–3 was needed), so the band never opened.

**The three fixes:**
- **MSE instead of Huber.** The true daily difference between being in and out is tiny (about ±0.05), buried in noise of ±1–3. Huber treats every error above 1 as "one step up or down", like counting votes instead of averaging amounts, so it learns the small average about 4× more slowly.
- **20-day targets.** A bear market lasts about 100 days. One-day learning passes that bad news back one day at a time; a 20-day target shows 20 days of bear losses in each training example.
- **10 gated heads.** Ten "advisors" are each trained on a random half of the data, and the agent trades only when they agree the gain beats the cost by more than their disagreement. Faced with noise they disagree, so nothing happens; in a real bear they agree, so it sells.

How the fixes combined in W-regime:

| Variant | Gain over B&H | Trades per year | Verdict |
|---|---|---|---|
| MSE alone | −0.01 | – | nothing |
| 20-day targets alone | +0.05 | – | little |
| MSE + 20-day | +0.17 (gross +0.35, close to the oracle) | 12.4 | too much trading |
| **+ gated heads (V1b)** | **+0.26** (72% of the oracle) | 4.7 | **pass** |

In the seed-3 world above, V1b reaches **0.79** where R0′ got −0.21.

These did not help: LayerNorm (+0.00), a 10× learning rate (starts trading noise), removing the anchor (+0.00), the mean-variance reward (+0.14), and a stricter gate (z = 2, +0.25, about the same as z = 1).

| | Old R0′ | V1 | V1b |
|---|---|---|---|
| W-null: trades per year | 10.8 ❌ | 0.1 ✅ | 0.1 ✅ |
| W-regime: gain | −0.17 ❌ | +0.00 ❌ | +0.26 ✅ (10 worlds: +0.06) |
| W-vol: gain | −0.09 ❌ | 0.00 ❌ | 0.00 ❌ |

### Key learnings

1. The old agent's problem is the **learning machinery, not only the data**. It even fails worlds where we know a profitable signal exists.
2. The new decision structure (V1) stops the random trading. Three fixes then made it learn regime timing: MSE loss, 20-day targets, and gated heads.
3. That success is **not yet robust**: +0.26 Sharpe on the 5 pre-registered worlds, only +0.05 on 5 new worlds, and +0.06 pooled over all 10. It was positive in 8 of 10 runs and never clearly negative, so it is safe but weak.
4. The learner is short of **events, not rows**. Each history contains only about 6 bear markets, just like the real data. That points straight at Step 4 (80 more years of history).
5. Volatility timing (W-vol) is unsolved in every variant. The signal is small (+0.19 Sharpe for the oracle) and sits below the anchor's hurdle (about 0.16).

### Why the Huber loss was used in the first place

- **It is the standard loss of DQN.** The original DQN (Mnih et al., Nature 2015) clipped the TD error to [−1, 1], which is the same as Huber with δ = 1. It stops a few huge errors from causing huge gradient steps and destabilising training.
- **That protection mattered for the old agent.** Its differential-Sharpe reward spiked while its estimates warmed up (hence `reward_clip`), and its targets bootstrapped from its own noisy, drifting estimates. The plan (Phase 2.2) therefore prescribed Huber for M2, together with reward normalisation, so that δ = 1 meant "one typical reward". It was the right choice there: M2 trained stably, and its failure came from elsewhere (overtrading).
- **It became a brake only in the new V1 structure.** There, the target of every exposure contains the full vol-scaled return of the day, with noise of ±1 per day and about ±4.5 over 20 days. So almost every error is larger than δ = 1, and Huber then behaves like an absolute-value loss: every sample pushes by the same fixed step, whatever its size.
- **Example:** suppose the truth is "being invested today is worth +0.05 more than being out", but each observation is +0.05 plus noise of size about 3. Only about a quarter of the observations fall inside Huber's ±1 zone, where it averages properly. The rest only vote "up" or "down", so the small average is learned about 4× more slowly. After the fixed training budget the estimate is still close to its starting point (buy-and-hold everywhere), which is exactly what the V1 probe showed.
- **Lesson:** a loss that makes training robust to noise can also hide a small but real signal. HL-Gauss (Step 2 of the plan) aims at both: robust to noise *and* the correct average.

---

## 4. Decision log (newest first)

### 2026-10-07 night: the hyperparameter search for a 1–4-week trader (your objective)

- **Your goal:** the agent should not hold one position for years. It should trade on a 1–4-week horizon where that pays after small costs. H1 (beat buy-and-hold at 10 bp) stays the confirmatory test.
- **A new test world, W-swing.** Each of the 26 synthetic tickers gets its own short-lived trend: its expected return drifts up or down and half of every drift fades within 10 trading days. A good trader can ride the up-drifts and step aside in the down-drifts.
  - The oracle is a textbook filter (Kalman) that knows the true process and watches each ticker's own returns. It trades only when the expected move over the drift's lifetime beats the 5 bp cost.
  - **How strong the drifts are** was set by a rule fixed before any agent ran: the weakest strength at which the oracle beats buy-and-hold by at least 0.15 Sharpe at 5 bp, trading 12–50 times a year. Result: the oracle gains +0.32 Sharpe and trades 17 times a year, i.e. it holds for about 3–4 weeks. Weaker drifts (+0.14) missed the bar.
- **The search, in simple words:** 32 different learner settings, with the knobs from the literature (loss type, look-ahead length, discount, network size, weight decay, number of advisors, random "opinions" for the advisors, the gate, learning rate).
  - Every setting starts neutral: no buy-and-hold head start and no anchor penalty, so stepping aside is not discouraged.
  - Each is scored on how much of the oracle's gain it captures in W-swing and in the bull/bear world. A setting that trades on noise in the no-timing world is pushed to the bottom.
  - Like a tournament: all 32 play a short game on 5 worlds each, the best 8 a longer game on 10 worlds, the best 2 the full game on 10 brand-new worlds.
- **Why synthetic and not real data:** the search compares 32 settings, and every comparison on real data would count as a trial (the budget has only 7 left). On synthetic worlds we know the possible profit, so we can tell "learns timing" from "got lucky".
- **What follows:** the best two settings go to the real data as four trials (each also in a version trained with your neo-broker costs, sized for a 2-ETF account of 2 × EUR 5,000).

### 2026-10-07 evening: your two ideas on the synthetic worlds (40-day targets, a 4× bigger network)

- **40-day targets: no measurable change.** In the bull/bear world the agent captures +0.08 instead of +0.06 Sharpe, both about a fifth of the oracle's +0.40, and single worlds move by ±0.2 between the two runs, so this is noise. The 20-day targets already pass the bear-market news back fast enough.
- **4× bigger network: a better pattern-finder, including patterns that are not there.**
  - Before costs it captures more than twice as much timing (+0.22 vs +0.10), so the small network was a real limit.
  - But in the world where timing cannot help (W-null) it now trades about 9 times a year and loses in 6 of 10 worlds.
  - The "ten advisors" gate no longer protects, because all ten share the same bigger brain and therefore make the same mistakes.
- **Both together:** worse than either.
- **What follows:** a bigger network only together with brakes (weight decay, random "opinions" for the advisors, a stricter gate). These are exactly the knobs of the planned hyperparameter search. The 40-day targets stay in that search; whether real data would then need a 60-day gap is decided only if a 40-day setting wins (your decision of 2026-10-07).

### 2026-10-07: Step 4 (81 years of pretraining): more consistent, but still buy-and-hold

- **Result:** V4 = V1b + pretraining on 1926–2007 scores 0.67 on the 26 ETFs, the same as buy-and-hold, against 0.64 for V1b.
- **What changed:** the 10 seeds now agree almost perfectly (their spread fell by 91 %), so the pretraining removed randomness. It did not add timing skill.
- **In simple words:** the agent practised on 81 years of markets and came back with one firm conclusion, "staying invested is best". With its current inputs and learning rules, it cannot find daily signals that would justify stepping aside often enough to pay off.
- **What follows:** more data alone is not the answer. The next step is to improve *how* it learns (the literature ideas) and *what* it sees (better signals), chosen systematically on synthetic worlds before any further real-data trial.

### 2026-10-07: V1b on real data: the overtrading is fixed, the timing is not there yet

- **Result:** on the 26 ETFs V1b scores 0.64. The old reference R0′ scores 0.43 and buy-and-hold 0.67.
- **It is adopted over R0′** (+0.12 Sharpe, a clear statistical win), because it stopped wasting money on trading. It trades about once per two years (the initial purchase) where R0′ traded 16 times a year.
- **In simple words: V1b has learned "don't trade unless you are sure", and on real data it is never sure.** It holds about 97 % all the time, which is buy-and-hold. That is exactly what the synthetic robustness check predicted, since its regime timing was weak. Real history has even fewer and noisier bear markets than the synthetic worlds.
- **Why that is still progress:** the agent now has the right default. Any future improvement in *learning* (more history in Step 4, the literature ideas) can only add timing on top of buy-and-hold, instead of first having to undo the losses from noise trading.

### 2026-10-07: Step 4, learning from 81 years instead of 16 (implemented, queued after V1b)

- **The idea in one sentence:** let the agent first practise on 81 years of US stock-market history (1926–2007), which contain many more bear markets than our 16 ETF years, and only then fine-tune it on the ETFs.
- **Why:** the synthetic worlds showed that the agent is short of *events* (bear markets), not of rows. The only way to get more real events without touching the validation or test years is to go further back in time.
- **What exactly:**
  - The data is the daily returns of 5 US industry portfolios plus the whole market, from Kenneth French's data library, which you downloaded.
  - Each of the 10 seeds pretrains once with the full budget.
  - Every ETF fold then starts from that seed's pretrained network and trains with half the usual budget.
- **One detail:** the VIX did not exist before 1990, so in the old data the agent sees "VIX unknown" through an extra on/off input (`vix_avail`) instead of a fake value.
- **The fair comparison:** V4 (V1b + pretraining) against V1b itself, so the only real change is the pretraining.

### 2026-10-07: what the deep-learning literature suggests next (sources: `docs/evidence-dqn-trading.md`)

Our diagnosis, small value gaps hidden in noisy targets, is a known problem in deep RL, and the literature has recipes for it:

| Idea | What it does, in simple words | Source |
|---|---|---|
| **HL-Gauss loss** | The network predicts a small histogram of possible values instead of one number. It is robust to noise like Huber, but keeps the correct average like MSE, and it is what lets *bigger* networks keep improving. | Farebrother et al. 2024 |
| **Bigger networks with brakes** | 4× wider networks helped, but only together with weight decay (shrinks unneeded weights), periodic partial resets (fights getting stuck), and LayerNorm inside residual blocks. Our plain LayerNorm test did not help, which fits this: the residual form matters. | BBF (Schwarzer et al. 2023), SimBa (Lee et al. 2025) |
| **Long targets first, shorter later** | BBF starts with 10-step targets and shrinks them to 3, while lengthening the time horizon (discount 0.97 → 0.997). For us that could mean 40 → 10 days: fast learning of regimes early, less bias later. | BBF; Fedus et al. 2020 (n-step returns help) |
| **Self-prediction side task** | Train the network to also predict its own next state or future returns. In BBF this was the single most important ingredient. It is already planned as V3's "auxiliary heads". | BBF, SPR |
| **Random priors for the advisors** | Give each of the 10 gated heads a fixed random "opinion". Their disagreement then stays meaningful even after much training, so the gate keeps working. | Osband et al. 2018 |
| **Learning from similar past regimes** | A trend model that looks up similar episodes across other assets and periods adapts faster to new regimes; it recovered twice as fast after COVID. | X-Trend (Wood et al. 2023) |

**How they will be tested:**
- Always on the synthetic worlds first (no trials), with all 10 seeds, because robustness is the open problem.
- Your two ideas, 40-day targets and a 4× wider network, are already queued this way.
- Once V1b's real-data result and Step 4 are in, the remaining ideas go into one **hyperparameter-optimisation plan**:
  - *where:* only on synthetic worlds and the 1926–2007 history, never on the 2014–2023 validation years;
  - *how:* many settings with a small training budget; only the better half continues with more budget, and so on (successive halving);
  - *what counts:* the share of the oracle's gain captured in W-regime and W-vol, with no noise trading allowed in W-null;
  - *end:* only the single final configuration goes to real data, as one trial.

### 2026-10-07 morning: V1b to real data, then Step 4, then a hyperparameter strategy

- **Decision (owner):**
  1. Run V1b on the real development data as Step 1 (one trial). It is compared against R0′ with the pre-registered adoption rule.
  2. Then Step 4: pretrain on 80 years of US industry data, 1926–2007, to give the agent more bear markets to learn from.
  3. Only after both: a proper hyperparameter-optimisation strategy.
- **Why this order:** V1b is safe (it does not trade noise) and passed the pre-registered synthetic rule. Real-data evidence is needed before spending more effort, and the clearest remaining lever is more regime events (learning 4 above).
- **Owner idea: doubling the 20-day targets to 40 days and using bigger networks.** Bigger networks are not a problem for the GPUs. The 40-day targets have one catch:
  - Every 20-day target looks 20 days into the future. To make sure no training example ever peeks across the boundary into the validation period, we leave a gap of 40 trading days between training and validation data: the 20-day observation window plus the 20-day look-ahead.
  - 40-day targets would need a 60-day gap. That is a protocol change, because all v2 runs must use the same gap to stay comparable.
  - So 40-day targets and bigger networks are tested on the synthetic worlds first (no gap problem there, no trials). The gap is changed for real data only if they clearly help.
