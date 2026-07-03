"""DQN agent with a two-input Conv1D Q-network and a vectorized replay buffer.

Speed-critical change vs. the old code: experience replay used to call
`model.fit()` once per sampled transition inside a Python loop. Here the whole
minibatch is trained in a SINGLE forward/backward pass (`train_on_batch`) using
Double-DQN targets, which is what makes GPU training 10-50x faster.
"""

import random
import numpy as np
import keras
from keras.layers import (Input, Conv1D, Dense, Concatenate,
                          GlobalAveragePooling1D, Flatten, LSTM)
from keras.models import Model, load_model
from keras.optimizers import Adam


class ReplayBuffer:
    """Pre-allocated ring buffer.

    Stores the two observation components (market tensor + position vector)
    separately so a sampled minibatch comes out as contiguous NumPy arrays
    ready to feed straight into the network. Lives in host RAM; only the
    sampled minibatch is copied to the GPU.

    Memory ~= 2*(window*n_feat + n_pos)*4 bytes per transition. For
    window=10, n_feat=5, n_pos=3 that is ~440 B, so 100k transitions ~= 44 MB.
    """

    def __init__(self, capacity, window, n_feat, n_pos):
        self.capacity = int(capacity)
        self.market = np.zeros((self.capacity, window, n_feat), np.float32)
        self.pos = np.zeros((self.capacity, n_pos), np.float32)
        self.next_market = np.zeros((self.capacity, window, n_feat), np.float32)
        self.next_pos = np.zeros((self.capacity, n_pos), np.float32)
        self.action = np.zeros(self.capacity, np.int32)
        self.reward = np.zeros(self.capacity, np.float32)
        self.done = np.zeros(self.capacity, np.float32)
        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self.idx
        self.market[i], self.pos[i] = obs
        self.next_market[i], self.next_pos[i] = next_obs
        self.action[i] = action
        self.reward[i] = reward
        self.done[i] = float(done)
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.market[idx], self.pos[idx], self.action[idx], self.reward[idx],
                self.next_market[idx], self.next_pos[idx], self.done[idx])

    def __len__(self):
        return self.size


def build_model(arch, window, n_feat, n_pos, action_size, lr):
    """Two-input Q-network: a temporal encoder over the (window, n_feat) market
    tensor, concatenated with the position vector, then dense head -> Q-values.
    """
    market_in = Input(shape=(window, n_feat), name="market")
    if arch == "conv":
        # causal padding keeps the encoder from mixing in any future step
        x = Conv1D(32, 3, activation="relu", padding="causal")(market_in)
        x = Conv1D(64, 3, activation="relu", padding="causal")(x)
        x = GlobalAveragePooling1D()(x)
    elif arch == "lstm":
        x = LSTM(64)(market_in)
    else:  # "mlp" fallback: flatten the window, no temporal inductive bias
        x = Flatten()(market_in)
        x = Dense(128, activation="relu")(x)

    pos_in = Input(shape=(n_pos,), name="position")
    h = Concatenate()([x, pos_in])
    h = Dense(64, activation="relu")(h)
    h = Dense(32, activation="relu")(h)
    q = Dense(action_size, activation="linear")(h)

    model = Model([market_in, pos_in], q)
    model.compile(loss="mse", optimizer=Adam(learning_rate=lr))
    return model


class Agent:
    def __init__(self, window=None, n_feat=None, n_pos=3, action_size=3,
                 model_name=None, is_eval=False, use_target=True, arch="conv",
                 gamma=0.99, lr=0.001, epsilon_start=1.0, epsilon_min=0.01):
        self.model_name = model_name
        self.is_eval = is_eval
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate = lr
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.eps_decay_steps = 1

        if is_eval:
            # Load a trained model and read the input/output shapes back from it
            # so evaluation never needs the training hyperparameters.
            # Models are stored as portable single-file ".keras" archives.
            name = model_name if model_name.endswith(".keras") else model_name + ".keras"
            self.model = load_model("models/" + name)
            self.window = int(self.model.inputs[0].shape[1])
            self.n_feat = int(self.model.inputs[0].shape[2])
            self.n_pos = int(self.model.inputs[1].shape[1])
            self.action_size = int(self.model.outputs[0].shape[1])
            self.target_model = None
        else:
            self.window, self.n_feat, self.n_pos = window, n_feat, n_pos
            self.model = build_model(arch, window, n_feat, n_pos, action_size, lr)
            self.target_model = self._clone(self.model) if use_target else None

    def _clone(self, model):
        clone = keras.models.clone_model(model)   # same architecture, fresh weights
        clone.set_weights(model.get_weights())
        return clone

    def sync_target(self):
        """Hard update: copy online weights into the target network."""
        if self.target_model is not None:
            self.target_model.set_weights(self.model.get_weights())

    # --- exploration schedule -------------------------------------------
    # Linear decay from epsilon_start to epsilon_min over `decay_fraction` of
    # total training, updated ONCE PER ENV STEP in the training loop. The old
    # code decayed inside the replay step (every timestep), collapsing epsilon
    # to the minimum within the first episode.
    def set_epsilon_schedule(self, total_steps, decay_fraction=0.6):
        self.eps_decay_steps = max(1, int(total_steps * decay_fraction))

    def update_epsilon(self, global_step):
        frac = min(1.0, global_step / self.eps_decay_steps)
        self.epsilon = self.epsilon_start + frac * (self.epsilon_min - self.epsilon_start)

    # --- policy ----------------------------------------------------------
    def act(self, obs, greedy=False):
        """Epsilon-greedy action. `greedy=True` forces the learned policy,
        used for validation/evaluation so results are never polluted by
        exploration."""
        if not greedy and not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        market, pos = obs
        q = self.model.predict_on_batch([market[None, ...], pos[None, ...]])
        return int(np.argmax(np.asarray(q)[0]))

    # --- learning --------------------------------------------------------
    def learn(self, buffer, batch_size):
        """One vectorized Double-DQN update over a sampled minibatch."""
        m, p, a, r, nm, npv, d = buffer.sample(batch_size)

        # Double DQN: online net picks the next action, target net values it.
        next_online = np.asarray(self.model.predict_on_batch([nm, npv]))
        next_actions = np.argmax(next_online, axis=1)
        next_target = np.asarray(self.target_model.predict_on_batch([nm, npv]))
        q_next = next_target[np.arange(batch_size), next_actions]

        targets = r + self.gamma * q_next * (1.0 - d)   # 0 bootstrap on terminal

        q = np.asarray(self.model.predict_on_batch([m, p]))
        q[np.arange(batch_size), a] = targets            # only the taken action's target changes
        self.model.train_on_batch([m, p], q)
