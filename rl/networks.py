"""Two-input Q-network with late fusion (kept from the original design).

    market window (W x F) --> temporal encoder --+
                                                 +--> Dense ... --> Q(s, .)
    position vector       ----------------------+

Encoder options (agent.network):
    arch     conv  : two causal Conv1D layers (no step sees a later step), then
                     pooling = flatten (keeps the time order; spec Phase 2.1) or
                     gap (GlobalAveragePooling1D, discards the order; LEGACY)
             lstm  : one LSTM(64)
             mlp   : flatten + Dense(128), no temporal structure
             conv_transformer : convolutional transformer after Guijarro-Ordonez,
                     Pelger & Zanotti (2026), plan §7 Track A2 (see conv_transformer_encoder)
    dueling        : Q = V(s) + A(s, a) - mean_a A(s, a)   (Wang et al. 2016)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DuelingCombine(layers.Layer):
    """Combine value and advantage streams into Q-values."""

    def call(self, inputs):
        value, adv = inputs
        return value + adv - tf.reduce_mean(adv, axis=1, keepdims=True)


class InstanceNorm(layers.Layer):
    """Instance normalisation for sequences (B, T, C).

    Each channel of each sample is normalised over the time axis; learnable
    scale and shift are per CHANNEL (as in the paper's CNN block).
    """

    def build(self, input_shape):
        c = int(input_shape[-1])
        self.gamma = self.add_weight(name="gamma", shape=(c,), initializer="ones", trainable=True)
        self.beta = self.add_weight(name="beta", shape=(c,), initializer="zeros", trainable=True)

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1], keepdims=True)
        return self.gamma * (x - mean) * tf.math.rsqrt(var + 1e-5) + self.beta


class PositionEmbedding(layers.Layer):
    """Learned embedding of the time step (0 .. W-1) added to the sequence.

    Self-attention by itself ignores order; the causal mask and the conv layers
    give some order information, an explicit position embedding makes it exact.
    """

    def build(self, input_shape):
        self.pos = self.add_weight(name="pos", shape=(int(input_shape[1]), int(input_shape[2])),
                                   initializer="zeros", trainable=True)

    def call(self, x):
        return x + self.pos


def conv_transformer_encoder(market_in, net):
    """Convolutional transformer encoder (paper's CNN + transformer, plan §7 A2).

    1. CNN block: two causal Conv1D layers (D filters, kernel 2, ReLU) with
       instance normalisation (each channel normalised over the time axis,
       per-channel scale/shift) and a residual connection from a 1x1
       projection of the input. Captures local patterns.
    2. Transformer layer: position embedding, H-head self-attention over the
       whole window, residual + LayerNorm, position-wise feed-forward
       (2D, ReLU -> D), residual + LayerNorm. Captures global patterns.
    3. Signal = the last time step's D-dimensional output.
    Defaults D = 8, H = 4, kernel 2 as in the paper (dims per head D / H = 2).

    No look-ahead: the input window only contains bars up to the decision bar
    t (MarketData.windows), so any mixing INSIDE the window (instance norm,
    attention) uses past data only.
    """
    D = int(net.get("transformer_dim", 8))
    H = int(net.get("transformer_heads", 4))
    k = int(net.get("kernel_size", 2))
    W = int(market_in.shape[1])
    skip = layers.Conv1D(D, 1)(market_in)
    h = layers.Conv1D(D, k, padding="causal", activation="relu")(market_in)
    h = InstanceNorm()(h)
    h = layers.Conv1D(D, k, padding="causal")(h)
    h = InstanceNorm()(h)
    h = layers.Activation("relu")(layers.Add()([h, skip]))
    h = PositionEmbedding()(h)
    att = layers.MultiHeadAttention(num_heads=H, key_dim=max(1, D // H))(h, h)
    h = layers.LayerNormalization()(layers.Add()([h, att]))
    ff = layers.Dense(D)(layers.Dense(2 * D, activation="relu")(h))
    h = layers.LayerNormalization()(layers.Add()([h, ff]))
    last = layers.Cropping1D((W - 1, 0))(h)                       # keep only the last time step
    return layers.Flatten()(last), h


def build_q_network(window, n_feat, pos_dim, n_actions, net):
    """Market encoder (+ position vector, late fusion) -> hidden layers -> one output per action.
    pos_dim = 0: market input only, as the V1 U-head needs (PROTOCOL Part II §V6.1)."""
    market_in = keras.Input(shape=(window, n_feat), name="market")
    pos_in = keras.Input(shape=(pos_dim,), name="position") if pos_dim else None

    arch = net.get("arch", "conv")
    if arch == "conv":
        x = market_in
        for filters in net.get("conv_filters", [32, 64]):
            x = layers.Conv1D(filters, net.get("kernel_size", 3), padding="causal", activation="relu")(x)
        pooling = net.get("pooling", "flatten")
        if pooling == "flatten":
            x = layers.Flatten()(x)
        elif pooling == "gap":
            x = layers.GlobalAveragePooling1D()(x)
        else:
            raise ValueError(f"unknown pooling '{pooling}'")
    elif arch == "conv_transformer":
        x, _ = conv_transformer_encoder(market_in, net)
    elif arch == "lstm":
        x = layers.LSTM(64)(market_in)
    elif arch == "mlp":
        x = layers.Dense(128, activation="relu")(layers.Flatten()(market_in))
    else:
        raise ValueError(f"unknown arch '{arch}'")

    h = layers.Concatenate()([x, pos_in]) if pos_dim else x   # late fusion
    for units in net.get("hidden", [64, 32]):
        if net.get("layer_norm", False):                      # V2 (§V6.2): Dense -> LayerNorm -> ReLU
            h = layers.Activation("relu")(layers.LayerNormalization()(layers.Dense(units)(h)))
        else:
            h = layers.Dense(units, activation="relu")(h)
    if net.get("dueling", False):
        q = DuelingCombine()([layers.Dense(1)(h), layers.Dense(n_actions)(h)])
    else:
        q = layers.Dense(n_actions)(h)
    return keras.Model([market_in, pos_in] if pos_dim else market_in, q)
