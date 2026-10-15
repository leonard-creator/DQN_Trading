"""Provenance of the W-swing calibration grid (PROTOCOL Part II §V12.1 item 23, `harness/synthetic.SWING`, 2026-10-07):
oracle vs buy-and-hold gain and turnover at 5 bp for drift SD 1.0 ... 2.0e-3 (no agent).
    PYTHONPATH=. python experiments/reports/v2_provenance/calibrate_swing.py
"""
from harness import synthetic as sy
from harness.config import load_config
cfg = load_config("config/v2/V1b.yaml", {"evaluation": {"costs": {"primary_bps": 2, "half_spread_bps": 3}}})
for sd in [1.0e-3, 1.25e-3, 1.5e-3, 1.75e-3, 2.0e-3]:
    sy.SWING["drift_sd"] = sd
    print(f"drift_sd {sd:.2e}:", flush=True)
    sy.calibrate(cfg, worlds=["W-swing"])
