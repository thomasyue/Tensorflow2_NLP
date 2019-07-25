import tensorflow as tf
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(123)
tf.random.set_seed(123)

batch_size = 64
total_words = 10000
max_length = 150
embedding_dim = 100

(x, y), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=total_words)

x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_length)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length)
print(x.shape, y.shape, x_test.shape, y_test.shape)

temp = np.zeros((x.shape[0], max_length, total_words))
temp[np.expand_dims(np.arange(x.shape[0]), axis=0).reshape(x.shape[0], 1), np.repeat(
    np.array([np.arange(max_length)]), x.shape[0], axis=0), x] = 1

x_one_hot = temp

temp = np.zeros((x_test.shape[0], max_length, total_words))
temp[np.expand_dims(np.arange(x.shape[0]), axis=0).reshape(x_test.shape[0], 1), np.repeat(
    np.array([np.arange(max_length)]), x_test.shape[0], axis=0), x] = 1
x_test_one_hot = temp

db = tf.data.Dataset.from_tensor_slices((x, [x_one_hot, y]))
db = db.shuffle(1000).batch(batch_size)

test_db = tf.data.Dataset.from_tensor_slices((x_test, [x_test_one_hot, y_test]))
test_db = test_db.shuffle(1000).batch(batch_size)


class Encoder(tf.keras.layers.Layer):

    def __init__(self, lstm_units, hidden_dim, **kwargs):
        self.lstm_units = lstm_units
        self.hidden_dim = hidden_dim
        super(Encoder, self).__init__(**kwargs)

        self.lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units,
                                                                        return_sequences=True,
                                                                        name='ENCODE_BiLSTM_1'))
        self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units,
                                                                        return_sequences=False,
                                                                        name='ENCODE_BiLSTM_2'))
        self.hidden = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu)

    def call(self, inputs, training=None):
        h = self.lstm1(inputs)
        h = self.lstm2(h)
        encoded = self.hidden(h)

        return encoded

    def get_config(self):
        config = {'lstm_units': self.lstm_units,
                  'hidden_dim': self.hidden_dim}
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Decoder(tf.keras.layers.Layer):

    def __init__(self, max_len, lstm_units, vocab_size, **kwargs):
        self.max_len = max_len
        self.lstm_units = lstm_units
        self.vocab_size = vocab_size

        super(Decoder, self).__init__(**kwargs)

        # [None, hidden_dim] -> [None, total_words, hidden_dim]
        self.repeat = tf.keras.layers.RepeatVector(self.max_len)

        self.lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units,
                                                                        return_sequences=True,
                                                                        name='DECODE_BiLSTM_1'))
        self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.lstm_units,
                                                                        return_sequences=True,
                                                                        name='DECODE_BiLSTM_2'))
        self.out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.vocab_size,
                                                                         activation=tf.nn.softmax),
                                                   name='OUTPUT')

    def call(self, inputs, training=None):
        h = self.repeat(inputs)
        h = self.lstm1(h)
        h = self.lstm2(h)
        output = self.out(h)

        return output

    def get_config(self):
        config = {'max_len': self.max_len,
                  'lstm_units': self.lstm_units,
                  'vocab_size': self.vocab_size}
        base_config = super(Decoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Sentiment(tf.keras.layers.Layer):

    def __init__(self, hidden_dim, num_classes, **kwargs):
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        super(Sentiment, self).__init__(**kwargs)

        self.dense1 = tf.keras.layers.Dense(self.hidden_dim, activation=tf.nn.relu, name='SENTIMENT_FF_1')
        self.senti_pred = tf.keras.layers.Dense(self.num_classes, activation='sigmoid', name='SENTIMENT_OUT')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        out = self.senti_pred(x)

        return out

    def get_config(self):
        config = {'num_classes': self.num_classes,
                  'hidden_dim': self.hidden_dim}
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def VAE(lstm_units, hidden_dim, num_classes, vocab_size, max_len):
    input1 = tf.keras.layers.Input(shape=(max_len,), name='ENCODER_INPUT')

    embed_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    encoder_layer = Encoder(lstm_units, hidden_dim)
    sentiment_layer = Sentiment(hidden_dim, num_classes)
    decoder_layer = Decoder(max_len, lstm_units, vocab_size)

    embedded = embed_layer(input1)
    encoded = encoder_layer(embedded)

    sentiment_pred = sentiment_layer(encoded)

    decoded = decoder_layer(encoded)

    autoencoder = tf.keras.Model(inputs=input1, outputs=[decoded, sentiment_pred])

    return autoencoder


# layer_dict = {'Encoder': Encoder,
#               'Decoder': Decoder,
#               'Sentiment': Sentiment}


def main():
    lstm_units = 64
    num_classes = 1
    hidden_dim = 50
    vocab_size = total_words
    epochs = 5
    lr = 1e-3

    autoencoder = VAE(lstm_units, hidden_dim, num_classes, vocab_size, max_length)
    autoencoder.build(input_shape=(-1, max_length))
    autoencoder.summary()
    optimizer = tf.optimizers.Adam(learning_rate=lr)

    for epoch in range(epochs):



        for step, (x, y) in enumerate(db):

            with tf.GradientTape() as tape:
                decoded_logits, senti_logits = autoencoder(x)
                decoder_loss = tf.losses.sparse_categorical_crossentropy(x, decoded_logits, from_logits=True)
                senti_loss = tf.losses.binary_crossentropy(y, senti_logits, from_logits=True)
                total_loss = decoder_loss + senti_loss

            grads = tape.gradient(total_loss, autoencoder.trainable_variables)
            optimizer.apply(zip(grads, autoencoder.trainable_variables))

            if step%100 == 0:
                print('epoch: {} step: {}  decoder_loss: {} sentiment_loss: {}'.format(epoch,
                                                                                       step,
                                                                                       decoder_loss,
                                                                                       senti_loss))

if __name__ == '__main__':
    main()


