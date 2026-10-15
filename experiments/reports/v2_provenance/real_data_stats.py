"""Provenance of the real-data numbers in RESULTS.md that the generated step reports do not show directly
(2026-10-15). No training; reads experiments/trials.csv and the stored run outputs (experiments/runs/).
    PYTHONPATH=. python experiments/reports/v2_provenance/real_data_stats.py
Sections: (a) the wrap-up table: gross (0 bp) and net (10 bp) medians on the 26 ETFs; (b) V4 vs buy-and-hold (exactly and within 0.005 Sharpe);
(c) V3's deviating runs against V4; (d) the descriptive baseline mean_reversion (realism check) per set and per
fold.
"""

import pandas as pd

pd.set_option("display.width", 200)
T = pd.read_csv("experiments/trials.csv")


def metrics(name):
    m = pd.read_csv(T[T.name == name].iloc[-1].output_dir + "/metrics.csv")
    return m[m.eval_set == "train"]


print("(a) median Sharpe on the 26 ETFs")
rows = []
for n in ("baseline_buy_and_hold", "m4_cross_resid_transformer", "v2_R0prime", "v2_V1b", "v2_V4", "v2_H1", "v2_H2",
          "v2_H1nb", "v2_H2nb", "v2_V3", "baseline_mean_reversion"):
    m = metrics(n)
    rows.append({"config": n, "gross_0bp": m[m.cost_bps == 0].sharpe.median(), "net_10bp": m[m.cost_bps == 10].sharpe.median(),
                 "turnover": m[m.cost_bps == 10].turnover.median()})
print(pd.DataFrame(rows).round(4).to_string(index=False))

k = ["seed", "fold"]
at10 = lambda n: metrics(n).query("cost_bps == 10")[k + ["sharpe", "turnover", "exposure"]]   # noqa: E731
same = lambda d: f"{int((d.abs() < 0.005).sum())} within 0.005 Sharpe, {int((d.abs() < 1e-9).sum())} exactly"  # noqa: E731
print("\n(b) V4 vs buy-and-hold, 50 (seed, fold) pairs at 10 bp")
m = at10("v2_V4").merge(at10("baseline_buy_and_hold"), on=k, suffixes=("_v4", "_bh"))
print("medians", round(m.sharpe_v4.median(), 4), round(m.sharpe_bh.median(), 4), f"| identical to buy-and-hold in "
      f"{same(m.sharpe_v4 - m.sharpe_bh)} (of {len(m)})")

print("\n(c) V3 against V4")
m = at10("v2_V3").merge(at10("v2_V4")[k + ["sharpe"]], on=k, suffixes=("_v3", "_v4"))
m["d"] = m.sharpe_v3 - m.sharpe_v4
dev = m[m.turnover > 2]
print(f"identical to V4: {same(m.d)} (of {len(m)}); trading > 2/yr: {len(dev)} (mean d {dev.d.mean():+.2f}, "
      f"median turnover {dev.turnover.median():.1f}, exposure {dev.exposure.mean():.2f}); all runs: mean d {m.d.mean():+.3f}, "
      f"median d {m.d.median():+.3f}")
print(dev.groupby("fold").d.agg(["size", "mean"]).round(2).to_string())

print("\n(d) baseline mean_reversion (deterministic: seed 0); buy-and-hold on the other sets: experiments/reports/V2_3.md")
mr = pd.read_csv(T[T.name == "baseline_mean_reversion"].iloc[-1].output_dir + "/metrics.csv").query("seed == 0")
print(mr[mr.cost.isin(["0bp", "10bp", "neo_broker"])].groupby(["eval_set", "cost"])
      [["sharpe", "turnover", "exposure", "avg_holding"]].median().round(2).to_string())
f = mr.query("eval_set == 'train' and cost == '10bp'").set_index("fold").sharpe.rename("mean_reversion").to_frame()
f["buy_and_hold"] = metrics("baseline_buy_and_hold").query("cost_bps == 10 and seed == 0").set_index("fold").sharpe
print(f.round(2).T.to_string())
