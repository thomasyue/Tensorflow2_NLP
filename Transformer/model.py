"""
self implementation of <attentions is all you need> using tensorflow 2.0
haven't run any unit test or debug yet.
parameters are pretty much the same as the original paper.
"""
import tensorflow as tf


def scaled_dot_product_attention(query, key, value, mask):
    """
    Page 4: Figure 2 left.
    attention(Q,K,V) = softmax(QK/sqrt(d_k))*V

    :param query:   shape: [None, seq_len_q, depth]
    :param key:     shape: [None, seq_len_k, depth]
    :param value:   shape: [None, seq_len_v, depth_v]
    :param mask:    shape: [None, seq_len_q, seq_len_k]
    :return:        output, attention_weights
    """

    # QK  [None, seq_len_q, depth] * [ None, depth, seq_len_k] -> [None, seq_len_q, seq_len_k]
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    dk = tf.cast(tf.shape(key)[-1], dtype=tf.float32)
    scaled_attention_logits = matmul_qk/tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1) # softmax along seq_len_k.

    output = tf.matmul(attention_weights, value)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention", **kwargs):
        super(MultiHeadAttention, self).__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.linear = tf.keras.layers.Dense(d_model)

    def split_head(self, x, batch_size):
        # [None, seq_len_*, num_heads, depth]
        x = tf.reshape(x, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])


    def call(self, v, k, q, mask, training=None):
        batch_size = tf.shape(q[0]) # [None, ...] extract None in this case

        q = self.wq(q)  # [None, seq_len_q, d_model]
        k = self.wk(k)  # [None, seq_len_k, d_model]
        v = self.wv(v)  # [None, seq_len_v, d_model]

        q = self.split_head(q, batch_size)  # [None, num_heads, seq_len_q, depth]
        k = self.split_head(k, batch_size)  # [None, num_heads, seq_len_k, depth]
        v = self.split_head(v, batch_size)  # [None, num_heads, seq_len_v, depth]

        # scaled_attention: [None, num_heads, seq_len_q, depth]
        # attention_weights:[None, num_heads, seq_len_q, seq_len_k]
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3]) # [None, seq_len_q, num_heads, depth]
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model)) # [None, seq_len_q, d_model]

        output = self.linear(concat_attention) # [None, seq_len_q, d_model]

        return output, attention_weights

    def get_config(self):
        config = {'num_heads': self.num_heads,
                  'd_model': self.d_model}

        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), dtype=tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, vocab_size, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(vocab_size, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
                                     i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                                     d_model=d_model)

        # apply to even indices. 2i
        sines = tf.math.sin(angle_rads[:, 0::2])

        # apply to odd indices. 2i+1
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)

        pos_encoding = pos_encoding[tf.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs, training=None):
        out = inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]
        return out

    def get_config(self):
        config = {'vocab_size': self.vocab_size,
                  'd_model': self.d_model}

        base_config = super(PositionalEncoding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([tf.keras.layers.Dense(dff, activation=tf.nn.relu), # [None, seq_len, dff]
                                tf.keras.layers.Dense(d_model)])                   # [None, seq_len, d_model]


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout = dropout

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask, training=None):

        att_output, _ = self.mha(x, x, x, mask) # [None, input_seq_len, d_model]
        att_output = self.dropout1(att_output, training=training)
        out1 = self.layernorm1(x + att_output) # [None, input_seq_len, d_model]

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = {'d_model': self.d_model,
                  'num_heads': self.num_heads,
                  'dff': self.dff,
                  'dropout': self.dropout}

        base_config = super(EncoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout = dropout

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.dropout3 = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_output, look_ahead_mask, padding_mask, training=None):

        att1, att_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        att1 = self.dropout1(att1, training=training)
        out1 = self.layernorm1(att1+x)

        att2, att_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        att2 = self.dropout2(att2, training=training)
        out2 = self.layernorm2(att2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, att_weights_block1, att_weights_block2

    def get_config(self):
        config = {'d_model': self.d_model,
                  'num_heads': self.num_heads,
                  'dff': self.dff,
                  'dropout': self.dropout}

        base_config = super(DecoderLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, dropout, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.dropout = dropout

        self.embedding = tf.keras.layers.Embedding(self.input_vocab_size, self.d_model)
        self.pos_encoding = PositionalEncoding(self.input_vocab_size, self.d_model)
        self.enc_layers = [EncoderLayer(self.d_model, self.num_heads, self.dff, self.dropout)
                           for _ in range(self.num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask, training=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training)

        return x # [None, seq_len, d_model]

    def get_config(self):
        config = {'d_model': self.d_model,
                  'num_layers': self.num_layers,
                  'num_heads': self.num_heads,
                  'dff': self.dff,
                  'input_vocab_size': self.input_vocab_size,
                  'dropout': self.dropout}

        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, tgt_vocab_size, dropout, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.tgt_vocab_size = tgt_vocab_size
        self.dropout = dropout

        self.embedding = tf.keras.layers.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(tgt_vocab_size, d_model)
        self.dec_layer = [DecoderLayer(d_model, num_heads, dff, dropout) for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, enc_output, look_ahead_mask, padding_mask, training=None):
        att_weights = {}

        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, att_blk1, att_blk2 = self.dec_layer[i](x,
                                                      enc_output,
                                                      look_ahead_mask,
                                                      padding_mask,
                                                      training=training)
            att_weights['decoder_layer{}_block1'.format(i+1)] = att_blk1
            att_weights['decoder_layer{}_block2'.format(i+1)] = att_blk2

        # x: [None, tgt_seq_len, d_model]
        return x, att_weights

    def get_config(self):
        config = {'d_model': self.d_model,
                  'num_layers': self.num_layers,
                  'num_heads': self.num_heads,
                  'dff': self.dff,
                  'tgt_vocab_size': self.tgt_vocab_size,
                  'dropout': self.dropout}

        base_config = super(Decoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Transformer(tf.keras.Model):

    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, dropout, **kwargs):
        super(Transformer, self).__init__(**kwargs)

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, dropout)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask, training=None):
        enc_output = self.encoder(inp, enc_padding_mask, training=training)

        dec_output, _ = self.decoder(tar, enc_output, look_ahead_mask, dec_padding_mask, training=training)

        final_output = self.final_layer(dec_output)

        return final_output
