"""Evaluation harness for the DQN trading project (spec Phase 1, PROTOCOL.md).

The harness is built BEFORE the agent is changed, so that every later change
is measured the same way. Modules:

    config      load YAML configs, merge them, hash them (trial identity)
    data        download / load per-ticker daily history, test-period guard
    splits      walk-forward folds with purge, embargo and inner validation
    backtest    exposure series -> net daily returns (one cost model for all)
    baselines   buy-and-hold, random agent, momentum, MACD crossover
    metrics     Sharpe, Sortino, drawdown, turnover, holding period, ...
    stats       deflated Sharpe, PBO via CSCV, permutation test, Holm
    trials      append-only experiments/trials.csv
    experiment  run_experiment(config, seeds): the single entry point

Naming rule (spec W2): the REWARD is called `diff_sharpe` (differential Sharpe,
Moody & Saffell 1998); the METRIC is called `deflated_sharpe` (Bailey &
Lopez de Prado 2014). The bare acronym "DSR" is never used.
"""
