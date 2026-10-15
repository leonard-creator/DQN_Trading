"""Provenance of the synthetic numbers in RESULTS.md that the generated reports (V2_0c.md, V2_hpo.md) do not show
directly (2026-10-07 -> 2026-10-15). No training; reads experiments/synthetic.csv and experiments/reports/v2_hpo/.
    PYTHONPATH=. python experiments/reports/v2_provenance/synthetic_stats.py
Sections: (a) the owner-idea screen, V1b pooled over seeds 0-9; (b) "Which ingredients pay off" (search round 1,
balanced design); (c) round 3: W-null worlds losing > 0.02; (d) the AHL screen (W-null counts, paired changes);
(e) the realism screens on W-revert / W-null.
"""

import pandas as pd

from harness import hpo

pd.set_option("display.width", 200)
S = pd.read_csv("experiments/synthetic.csv")
S = S[S["timestamp"] == S.groupby("config")["timestamp"].transform("max")]           # latest run of each config
S = S.assign(gain=S.agent_sharpe - S.bh_sharpe, gross=S.agent_sharpe_gross - S.bh_sharpe, ogain=S.oracle_sharpe - S.bh_sharpe)
count = lambda cond: (lambda s: int(cond(s).sum()))                                     # noqa: E731


def summary(d):
    return d.groupby(["config", "world"]).agg(gain=("gain", "median"), gross=("gross", "median"),
                                              oracle_gain=("ogain", "median"), turnover=("agent_turnover", "median"),
                                              oracle_turnover=("oracle_turnover", "median"),
                                              up=("gain", count(lambda s: s > 0.02)),
                                              down=("gain", count(lambda s: s < -0.02))).round(2)


print("(a) owner-idea screen, seeds 0-9")
v1b = S[S.config.isin(["v2_V1_mse_n20_h10g1", "v2_V1_mse_n20_h10g1_s5-9"])].assign(config="V1b (pooled)")
print(summary(pd.concat([v1b, S[S.config.isin(["v2_V1b_n40", "v2_V1b_wide", "v2_V1b_wide_n40"])]])).to_string())

print("\n(b) search round 1: mean per option (balanced design)")
r1 = pd.read_csv("experiments/reports/v2_hpo/round1.csv")
r1 = r1.assign(noise=r1.null_gain < hpo.NULL_FLOOR, capture=r1[["capture_swing", "capture_regime"]].mean(axis=1),
               top8=r1.index < 8)                                                       # round1.csv is sorted by score
for k in hpo.SPACE:
    print(r1.groupby(k).agg(n=("id", "size"), capture=("capture", "mean"), noise_trading=("noise", "mean"),
                            in_top8=("top8", "sum")).round(2).to_string(), "\n")

print("(c) round 3: W-null worlds losing > 0.02")
r3 = pd.read_csv("experiments/reports/v2_hpo/runs_r3.csv")
r3 = r3.assign(gain=r3.agent_sharpe - r3.bh_sharpe)
print(r3[r3.world == "W-null"].groupby("config").gain.apply(count(lambda s: s < -0.02)).to_string())

print("\n(d) AHL screen: W-null counts and paired median change vs the fixed range")
d = pd.read_csv("experiments/reports/v2_hpo/runs_screen_support.csv")
d = d.assign(gain=d.agent_sharpe - d.bh_sharpe, variant=d.config.str.split("_").str[1], id=d.config.str.split("_").str[2])
w = d[d.world == "W-null"].groupby(["id", "variant"])
print(w.gain.apply(count(lambda s: s < -0.02)).unstack().to_string())
for world in ("W-swing", "W-regime", "W-null"):
    t = d[d.world == world].pivot_table(index=["id", "seed"], columns="variant", values="gain")
    for v in ("ahl", "ahlpaper"):
        print(world, v, (t[v] - t["hl"]).groupby(level=0).median().round(3).to_dict())

print("\n(e) realism screens (5 bp; 'up' = seeds gaining > +0.02)")
print(summary(S[S.config.isin(["v2_realism_c00", "v2_realism_V1b"])]).to_string())
