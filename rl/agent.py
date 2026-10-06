"""Double DQN agent with action masking (spec Phase 2.2-2.6).

Update rule for a sampled transition (s, a, r, s', discount, mask'):

    a*   = argmax_{a' valid in s'} Q_online(s', a')          (Double DQN: online picks...)
    y    = r / reward_scale + discount * Q_target(s', a*)     (...target evaluates)
    loss = mean_i  w_i * L(y_i - Q_online(s_i, a_i))

    L = Huber(delta) or squared error (legacy); w_i = PER importance weights
    (all 1 without PER); discount = gamma^n * (1 - terminal).

Without Double DQN, the target is max_{a' valid} Q_target(s', a').

Masking: invalid actions get -1e9 before every argmax/max, both when acting
(epsilon-greedy explores only among valid actions) and inside the target.
With env.action_masking = false (legacy) every mask is all-True, which gives
the old behaviour. tests/test_rl_agent.py checks both paths.

Target network: soft update theta' <- tau*theta + (1-tau)*theta' after every
gradient step, or a hard copy every `hard_every` steps.

Learning rate: cosine or exponential decay from `lr` to lr * lr_min_frac over
all planned updates, or constant. Gradients are clipped by global norm.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from rl.networks import build_q_network

NEG = -1e9


def make_schedule(algo, total_updates):
    lr = float(algo["lr"])
    kind = algo.get("lr_schedule", "constant")
    steps = max(1, int(total_updates))
    if kind == "cosine":
        return keras.optimizers.schedules.CosineDecay(lr, steps, alpha=float(algo.get("lr_min_frac", 0.1)))
    if kind == "exponential":
        return keras.optimizers.schedules.ExponentialDecay(lr, steps, float(algo.get("lr_min_frac", 0.1)))
    if kind == "constant":
        return lr
    raise ValueError(f"unknown lr_schedule '{kind}'")


class DQNAgent:
    def __init__(self, window, n_feat, pos_dim, n_actions, net_cfg, algo, total_updates):
        self.online = build_q_network(window, n_feat, pos_dim, n_actions, net_cfg)
        self.target = build_q_network(window, n_feat, pos_dim, n_actions, net_cfg)
        self.target.set_weights(self.online.get_weights())
        self.n_actions = n_actions

        self.gamma = float(algo["gamma"])
        self.double = bool(algo.get("double", True))
        self.loss_type = algo.get("loss", "huber")
        if self.loss_type not in ("huber", "mse"):
            raise ValueError(f"unknown loss '{self.loss_type}'")
        self.delta = float(algo.get("huber_delta", 1.0))
        self.soft = algo.get("target_update", "soft") == "soft"
        self.tau = float(algo.get("tau", 0.005))
        self.hard_every = int(algo.get("hard_every", 500))

        self.schedule = make_schedule(algo, total_updates)
        clip = algo.get("grad_clip_norm")
        self.opt = keras.optimizers.Adam(learning_rate=self.schedule,
                                         **({"global_clipnorm": float(clip)} if clip else {}))
        self.updates = 0
        self._train = tf.function(self._train_step, reduce_retracing=True)
        self._q = tf.function(lambda m, p: self.online([m, p], training=False), reduce_retracing=True)

    # ------------------------------------------------------------------ acting
    def q_values(self, market, pos):
        return self._q(tf.convert_to_tensor(market, tf.float32),
                       tf.convert_to_tensor(pos, tf.float32)).numpy()

    def act(self, market, pos, mask, epsilon, rng):
        """Masked epsilon-greedy. Returns int actions (B,)."""
        q = np.where(mask, self.q_values(market, pos), -np.inf)
        action = np.argmax(q, axis=1)
        if epsilon > 0:
            explore = rng.random(len(action)) < epsilon
            # uniform choice among the VALID actions: random scores, invalid ones forced lowest
            random_valid = np.argmax(np.where(mask, rng.random(mask.shape), -1.0), axis=1)
            action = np.where(explore, random_valid, action)
        return action.astype(np.int32)

    def current_lr(self):
        s = self.schedule
        return float(s(self.opt.iterations)) if callable(s) else float(s)

    # ---------------------------------------------------------------- learning
    def targets(self, r, m2, p2, mask2, discount):
        """TD target y (exposed separately so tests can check it)."""
        q2_target = self.target([m2, p2], training=False)
        if self.double:
            q2_online = self.online([m2, p2], training=False)
            a2 = tf.argmax(tf.where(mask2, q2_online, NEG), axis=1, output_type=tf.int32)
            q_next = tf.gather(q2_target, a2, batch_dims=1)
        else:
            q_next = tf.reduce_max(tf.where(mask2, q2_target, NEG), axis=1)
        return r + discount * q_next

    def _train_step(self, m, p, a, r, m2, p2, mask2, discount, w):
        y = tf.stop_gradient(self.targets(r, m2, p2, mask2, discount))
        with tf.GradientTape() as tape:
            q = self.online([m, p], training=True)
            q_a = tf.gather(q, a, batch_dims=1)
            td = y - q_a
            if self.loss_type == "huber":
                abs_td = tf.abs(td)
                quad = tf.minimum(abs_td, self.delta)
                per_sample = 0.5 * quad ** 2 + self.delta * (abs_td - quad)
            else:
                per_sample = tf.square(td)
            loss = tf.reduce_mean(w * per_sample)
        grads = tape.gradient(loss, self.online.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.online.trainable_variables))
        if self.soft:
            for t_var, o_var in zip(self.target.weights, self.online.weights):
                t_var.assign(self.tau * o_var + (1.0 - self.tau) * t_var)
        return td, loss, tf.reduce_mean(q_a)

    def learn(self, batch, data, reward_scale=1.0):
        """One gradient step on a replay batch; returns (td errors, loss, mean Q)."""
        f32 = lambda x: tf.convert_to_tensor(x, tf.float32)        # noqa: E731
        td, loss, mean_q = self._train(
            f32(data.windows(batch["g"])), f32(batch["pos"]),
            tf.convert_to_tensor(batch["action"], tf.int32),
            f32(batch["reward"] / reward_scale),
            f32(data.windows(batch["g2"])), f32(batch["pos2"]),
            tf.convert_to_tensor(batch["mask2"], tf.bool),
            f32(batch["discount"]), f32(batch["weights"]))
        self.updates += 1
        if not self.soft and self.updates % self.hard_every == 0:
            self.target.set_weights(self.online.get_weights())
        return td.numpy(), float(loss), float(mean_q)
