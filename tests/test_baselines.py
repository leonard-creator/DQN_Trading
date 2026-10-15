"""Baselines: no look-ahead, correct grid, reproducible randomness."""

import numpy as np
import pytest

from harness.baselines import buy_and_hold, macd_crossover, mean_reversion, momentum, random_agent, stable_seed

RNG = np.random.default_rng(7)
CLOSE = 100 * np.exp(np.cumsum(0.01 * RNG.standard_normal(600)))


@pytest.mark.parametrize("fn", [lambda c: momentum(c, 60), lambda c: momentum(c, 20, allow_short=True),
                                lambda c: macd_crossover(c, 12, 26, 9),
                                # parameters are fitted on the returns up to the first decision (bar 300 < t)
                                lambda c: mean_reversion(c, np.array([300]), 11e-4, min_history=100)])
def test_no_lookahead_future_prices_do_not_change_past_exposure(fn):
    t = 400
    base = fn(CLOSE)
    perturbed = CLOSE.copy()
    perturbed[t + 1:] *= np.exp(0.5 * RNG.standard_normal(len(CLOSE) - t - 1))   # rewrite the future
    np.testing.assert_array_equal(base[: t + 1], fn(perturbed)[: t + 1])


def test_momentum_sign_and_warmup():
    c = np.r_[np.full(10, 100.0), np.linspace(100, 120, 20), np.linspace(120, 90, 20)]
    e = momentum(c, 5)
    assert np.all(e[:5] == 0)                       # not enough history
    assert e[20] == 1.0 and e[45] == 0.0
    assert set(np.unique(momentum(c, 5, allow_short=True))) <= {-1.0, 0.0, 1.0}


def test_buy_and_hold_is_always_fully_invested():
    assert np.all(buy_and_hold(CLOSE) == 1.0)


def test_random_agent_stays_on_the_grid_and_respects_masking():
    dec = np.arange(100, 400)
    e = random_agent(len(CLOSE), dec, levels=4, seed=stable_seed(0, "SPY", "F1"))
    x = e[dec]
    assert np.all(np.isnan(np.delete(e, dec)))                 # only decision bars are set
    assert set(np.unique(x)) <= {0.0, 0.25, 0.5, 0.75, 1.0}
    assert np.all(np.abs(np.diff(np.r_[0.0, x])) <= 0.25 + 1e-12)  # one step of 1/K at a time
    short = random_agent(len(CLOSE), dec, levels=4, seed=1, allow_short=True)[dec]
    assert short.min() >= -1.0 and short.max() <= 1.0


def test_random_agent_is_reproducible_and_seed_dependent():
    dec = np.arange(50, 300)
    a = random_agent(len(CLOSE), dec, 4, stable_seed(3, "SPY", "F2"))
    b = random_agent(len(CLOSE), dec, 4, stable_seed(3, "SPY", "F2"))
    c = random_agent(len(CLOSE), dec, 4, stable_seed(4, "SPY", "F2"))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a[dec], c[dec])


def test_stable_seed_does_not_depend_on_python_hash_randomisation():
    assert stable_seed(0, "SPY", "F1") == [zlib_crc("0"), zlib_crc("SPY"), zlib_crc("F1")]


def zlib_crc(s):
    import zlib
    return zlib.crc32(s.encode())


def test_mean_reversion_holds_without_reversion_and_steps_aside_after_rallies():
    """Realism-check baseline (PROTOCOL Part II §V12.1 item 25): a trending random walk is simply held; on a
    mean-reverting path (synthetic W-revert) the estimated rule is flat mainly after rallies."""
    from harness import synthetic as sy
    rng = np.random.default_rng(11)
    trend = 100 * np.exp(np.cumsum(0.001 + 0.01 * rng.standard_normal(2000)))
    assert mean_reversion(trend, np.array([1500]), 11e-4)[1500:].mean() > 0.8
    r = sy.simulate("W-revert", 3000, 3)[0][:, 0]
    close = 100 * np.cumprod(1 + r)
    e = mean_reversion(close, np.array([2000]), 5e-4)[2000:]
    past5 = np.log(close[2000:] / close[1995:-5])                      # return of the 5 days up to each decision
    assert 0.3 < e.mean() < 0.98 and past5[e == 0].mean() > past5[e == 1].mean()
