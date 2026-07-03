"""Greedy evaluation of a trained agent.

`evaluate()` is used for BOTH in-training validation and standalone testing, so
the two can never disagree. Run standalone:

    python evaluate.py SundPGI_test.csv <model_name> --test
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless-safe: render/save without a display
import matplotlib.pyplot as plt

from functions import load_features, FeatureScaler, formatPrice
from env import TradingEnv
from agent.agent import Agent


def evaluate(agent, env, plotting=False, title="model", save_path=None, verbose=False):
    """Roll the greedy policy through `env` once and report performance.

    Returns a metrics dict: realized P&L (euro), return %, Sharpe of the reward
    stream, and trade count.
    """
    obs = env.reset()
    buy_signals, sell_signals, rewards = [], [], []
    info = {"realized_pnl": 0.0}
    done = False
    while not done:
        t = env.t                                   # day the action is taken on
        action = agent.act(obs, greedy=True)        # greedy: no exploration
        obs, reward, done, info = env.step(action)
        rewards.append(reward)
        if info["trade_type"] == "buy":
            buy_signals.append((t, info["price"]))
            if verbose:
                print("Buy  @ " + formatPrice(info["price"]))
        elif info["trade_type"] == "sell":
            sell_signals.append((t, info["price"]))
            if verbose:
                print("Sell @ " + formatPrice(info["price"]))

    realized = info["realized_pnl"]
    invested = sum(price for _, price in buy_signals)
    return_pct = 100.0 * realized / (invested + 1e-9)
    r = np.asarray(rewards)
    sharpe = float(np.mean(r) / (np.std(r) + 1e-9)) if len(r) else 0.0

    print("--------------------------------")
    print(f"{title}: realized {formatPrice(realized)} | return {return_pct:.2f}% "
          f"| trades {env.trade_count} | reward Sharpe {sharpe:.3f}")
    print("--------------------------------")

    if plotting:
        prices = env.close
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(prices)), prices, label="Price")
        if buy_signals:
            bx, by = zip(*buy_signals)
            plt.scatter(bx, by, marker="^", color="g", label="Buy")
        if sell_signals:
            sx, sy = zip(*sell_signals)
            plt.scatter(sx, sy, marker="v", color="r", label="Sell")
        plt.legend(loc="upper left",
                   title=f"Realized: {formatPrice(realized)} ({return_pct:.1f}%)")
        plt.xlabel("Trading days")
        plt.ylabel("Close price")
        plt.title(title)
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            plt.savefig(save_path, dpi=120, bbox_inches="tight")
            print("saved graph ->", save_path)
        else:
            plt.show()
        plt.close()

    return {"realized_pnl": realized, "return_pct": return_pct,
            "sharpe": sharpe, "trades": env.trade_count}


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained DQN trader")
    p.add_argument("stock", help="CSV filename inside train_data/ or test_data/")
    p.add_argument("model_name", help="model dir under models/ (e.g. myrun9)")
    p.add_argument("--test", action="store_true", help="load from test_data/ (else train_data/)")
    p.add_argument("--max-position", type=int, default=10)
    p.add_argument("--transaction-cost", type=float, default=1.0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    agent = Agent(model_name=args.model_name, is_eval=True)
    window = int(agent.window)

    # models are saved as "<base><episode>.keras"; the scaler is saved once as
    # "<base>_scaler.json". Strip the ".keras" suffix and trailing episode
    # digits to recover the base name.
    base = args.model_name[:-6] if args.model_name.endswith(".keras") else args.model_name
    base = base.rstrip("0123456789")
    scaler = FeatureScaler.load(f"models/{base}_scaler.json")

    df, close = load_features(args.stock, scaler.features, test=args.test)
    feat = scaler.transform(df)
    env = TradingEnv(close, feat, window,
                     transaction_cost=args.transaction_cost, max_position=args.max_position)

    save_path = f"graphs/{args.model_name}_on_{os.path.splitext(args.stock)[0]}.png"
    evaluate(agent, env, plotting=True, save_path=save_path, verbose=args.verbose,
             title=f"{args.model_name} on {args.stock}")


if __name__ == "__main__":
    main()
