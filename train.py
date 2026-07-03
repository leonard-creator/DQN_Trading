"""Train the DQN trader.

Example:
    python train.py SundPGI_train.csv myrun --episodes 50 --window 10

Highlights vs. the old loop:
* One shared TradingEnv drives both training and (greedy) validation.
* Vectorized Double-DQN replay -> big GPU speedup.
* Epsilon decays on a schedule over the whole run, not per replay call.
* Feature scaler is fit on the training slice only and saved with the model.
"""

import argparse
import os
import random
import numpy as np

from functions import load_features, FeatureScaler, DEFAULT_FEATURES
from env import TradingEnv
from agent.agent import Agent, ReplayBuffer
from evaluate import evaluate


def set_seeds(seed):
    """Seed python/numpy/tensorflow for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="Train a DQN trader (multi-input, DSR reward)")
    p.add_argument("stock", help="CSV filename inside train_data/")
    p.add_argument("model_name", help="base name for saved models/ dirs")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--max-position", type=int, default=10)
    p.add_argument("--transaction-cost", type=float, default=1.0)
    p.add_argument("--dsr-eta", type=float, default=0.01)
    p.add_argument("--dsr-clip", type=float, default=5.0)
    p.add_argument("--inventory-penalty", type=float, default=0.0)
    p.add_argument("--target-sync", type=int, default=500, help="steps between target net updates")
    p.add_argument("--train-every", type=int, default=4, help="env steps between learn() calls")
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-min", type=float, default=0.01)
    p.add_argument("--decay-fraction", type=float, default=0.6)
    p.add_argument("--arch", choices=["conv", "lstm", "mlp"], default="conv")
    p.add_argument("--validation-days", type=int, default=50)
    p.add_argument("--val-stock", default=None,
                   help="explicit validation CSV in train_data/ (else carve --validation-days from the train tail)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--save-every", type=int, default=0,
                   help="also save a per-episode checkpoint every N episodes (0 = off)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)
    os.makedirs("models", exist_ok=True)

    # --- data + scaler (fit on training data only, no leakage) ---
    df, close = load_features(args.stock, args.features, test=False)
    if args.val_stock:
        # explicit validation file (e.g. produced by scrape_data.py): fit the
        # scaler on the whole training file; validation is its own series.
        scaler = FeatureScaler(args.features).fit(df)
        feat = scaler.transform(df)
        train_close, train_feat = close, feat
        vdf, vclose = load_features(args.val_stock, args.features, test=False)
        vfeat = scaler.transform(vdf)
    else:
        # carve the validation window off the tail of the training file
        split = len(close) - args.validation_days
        scaler = FeatureScaler(args.features).fit(df.iloc[:split])
        feat = scaler.transform(df)
        train_close, train_feat = close[:split], feat[:split]
        vclose, vfeat = close[split:], feat[split:]
    scaler.save(f"models/{args.model_name}_scaler.json")

    env_kwargs = dict(transaction_cost=args.transaction_cost, max_position=args.max_position,
                      dsr_eta=args.dsr_eta, dsr_clip=args.dsr_clip)
    train_env = TradingEnv(train_close, train_feat, args.window,
                           inventory_penalty=args.inventory_penalty, **env_kwargs)
    val_env = TradingEnv(vclose, vfeat, args.window, **env_kwargs)

    n_feat = feat.shape[1]
    agent = Agent(window=args.window, n_feat=n_feat, n_pos=train_env.n_pos,
                  action_size=3, model_name=args.model_name, arch=args.arch,
                  gamma=args.gamma, lr=args.lr,
                  epsilon_start=args.epsilon_start, epsilon_min=args.epsilon_min)
    buffer = ReplayBuffer(args.buffer_size, args.window, n_feat, train_env.n_pos)

    total_steps = args.episodes * train_env.length
    agent.set_epsilon_schedule(total_steps, args.decay_fraction)

    # wandb is optional: imported and used only when logging is enabled. Passing
    # --no-wandb (or simply not having wandb installed) trains without it. When
    # enabled it respects WANDB_MODE (online/offline).
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            mode = os.environ.get("WANDB_MODE", "online")
            run = wandb.init(project="Deep Q-learning trader", mode=mode,
                             config={**vars(args), "n_feat": n_feat, "reward": "differential_sharpe"})
        except ImportError:
            print("[train] wandb not installed; continuing without experiment logging.")
            use_wandb = False

    global_step = 0
    best_val = float("-inf")   # best validation net P&L seen -> drives _best checkpoint
    for episode in range(args.episodes):
        obs = train_env.reset()
        info = {"realized_pnl": 0.0, "net_pnl": 0.0, "position": 0}
        for _ in range(train_env.length):
            action = agent.act(obs)
            next_obs, reward, done, info = train_env.step(action)
            buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs

            # learn on a schedule; sync the target net periodically
            if len(buffer) >= args.batch_size and global_step % args.train_every == 0:
                agent.learn(buffer, args.batch_size)
            if global_step % args.target_sync == 0:
                agent.sync_target()
            agent.update_epsilon(global_step)
            global_step += 1

            if use_wandb and global_step % 200 == 0:
                wandb.log({"episode": episode, "global_step": global_step,
                           "epsilon": agent.epsilon, "train_net_pnl": info["net_pnl"],
                           "position": info["position"], "buffer": len(buffer)})
            if done:
                break

        # greedy validation on the held-out tail (no exploration pollution)
        val = evaluate(agent, val_env, plotting=False, title=f"val ep{episode}")
        print(f"Episode {episode + 1}/{args.episodes} | "
              f"train net €{info['net_pnl']:.2f} | "
              f"val net €{val['net_pnl']:.2f} ({val['return_pct']:.2f}%) | "
              f"val Sharpe {val['sharpe']:.3f} | eps {agent.epsilon:.3f}")
        if use_wandb:
            wandb.log({"episode": episode,
                       "train_net_pnl": info["net_pnl"], "train_realized_pnl": info["realized_pnl"],
                       "val_net_pnl": val["net_pnl"], "val_realized_pnl": val["realized_pnl"],
                       "val_return_pct": val["return_pct"], "val_sharpe": val["sharpe"]})

        # Checkpointing: keep the best-on-validation and the latest model under
        # stable names, so evaluation never depends on guessing an episode index.
        # --save-every optionally also keeps periodic per-episode snapshots.
        if val["net_pnl"] > best_val:
            best_val = val["net_pnl"]
            agent.model.save(f"models/{args.model_name}_best.keras")
            print(f"  ** new best validation net €{best_val:.2f} -> saved {args.model_name}_best.keras")
        if args.save_every and episode % args.save_every == 0:
            agent.model.save(f"models/{args.model_name}{episode}.keras")

    agent.model.save(f"models/{args.model_name}_last.keras")   # final model, stable name
    if use_wandb:
        run.finish()


if __name__ == "__main__":
    main()
