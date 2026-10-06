"""Risk-aware DQN agent (spec Phase 2 onwards), evaluated through harness/.

    features  per-ticker feature matrices (reuses scrape_data + functions)
    env       vectorised trading environment: capital-fraction positions,
              action masking, pluggable rewards, random-start episodes
    replay    uniform and prioritised replay storing indices, not windows
    networks  two-input Q-network (late fusion), optional dueling head
    agent     Double DQN with masked targets, Huber/MSE, LR schedule, soft/hard target
    trainer   one training run: collect, learn, select checkpoint on inner validation
    policy    DQNPolicy: the harness interface, runs (seed, fold) jobs in parallel

The original single-asset code (env.py, agent/agent.py, train.py) is left
unchanged; config/legacy.yaml reproduces its algorithmic choices here.
"""
