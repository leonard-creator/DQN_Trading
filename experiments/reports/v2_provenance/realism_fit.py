"""Provenance of the W-revert calibration (PROTOCOL Part II §V12.1 item 25, `harness/synthetic.REVERT`) and of the
synthetic variance ratios in RESULTS.md, "v2 realism check" and "v2 Step H on real data" (2026-10-15).
(1) Least-squares fit of the temporary-component model VR(q) = 1 - s + s (1 - phi^q) / (q (1 - phi)) to the measured
variance-ratio curves (all years; without the crises -> s = 0.407, phi = 0.904; targets from realism_measure.py).
(2) Variance ratios of the implemented worlds: one 40,000-bar path (seed 1), median over the 26 tickers.
(3) The W-revert oracle's gain over buy-and-hold at 5 bp and 11 bp (`harness.synthetic.calibrate`, seeds 0-4).
    PYTHONPATH=. python experiments/reports/v2_provenance/realism_fit.py
"""
import warnings

import numpy as np
from scipy.optimize import minimize

from harness import synthetic as sy
from harness.config import load_config

warnings.filterwarnings("ignore")
Q = np.array([2, 5, 10, 20, 60])
TARGETS = {"all": [0.92, 0.87, 0.82, 0.83, 0.76], "ex-crisis": [0.97, 0.93, 0.85, 0.79, 0.66]}


def vr_model(s, phi, q):
    return 1 - s + s * (1 - phi ** q) / (q * (1 - phi))


def fit(target):
    f = lambda x: np.sum((vr_model(x[0], x[1], Q) - target) ** 2)                      # noqa: E731
    return minimize(f, [0.2, 0.9], bounds=[(0.01, 0.9), (0.0, 0.995)]).x


def vr(x, q):
    return np.convolve(x - x.mean(), np.ones(q), "valid").var() / (q * x.var())


print("(1) fit of the temporary-component model")
for name, target in TARGETS.items():
    s, phi = fit(np.array(target))
    print(f"{name}: s = {s:.3f} (transitory share of daily variance), phi = {phi:.3f} (half-life "
          f"{np.log(0.5) / np.log(phi):.1f} days); model VR {np.round(vr_model(s, phi, Q), 2)} vs measured {target}")

print("\n(2) variance ratios of the implemented worlds (2/5/10/20/60 days)")
for w in ("W-revert", "W-swing", "W-null"):
    lr = np.log1p(sy.simulate(w, 40_000, 1)[0])
    print(f"{w:9s}", " / ".join(f"{np.median([vr(lr[:, i], q) for i in range(lr.shape[1])]):.2f}" for q in Q))

print("\n(3) W-revert oracle vs buy-and-hold")
for bps, hs in ((2, 3), (10, 1)):
    cfg = load_config("config/v2/V1b.yaml", {"evaluation": {"costs": {"primary_bps": bps, "half_spread_bps": hs}}})
    print(f"cost {bps + hs} bp:", flush=True)
    sy.calibrate(cfg, worlds=["W-revert"])
