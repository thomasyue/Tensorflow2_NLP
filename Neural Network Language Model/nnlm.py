import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

inputs = ['i love you so much ha ha ha']
total_words = 100
embed_dim = 50
n_classes = 10
max_length = 40


class nnlm(tf.keras.Model):
    """
    y = b + Wx + U*tanh(d+Hx)
    """
    def __init__(self, units):
        super(nnlm, self).__init__()

        # [None, total_words] -> [None, total_words, embed_dim]
        self.embed = tf.keras.layers.Embedding(total_words, embed_dim, input_length=max_length)

        #
        self.H = tf.Variable(tf.random.normal([total_words*embed_dim, units]))
        self.W = tf.Variable(tf.random.normal([total_words*embed_dim, n_classes]))
        self.U = tf.Variable(tf.random.normal([units, n_classes]))
        self.d = tf.Variable(tf.random.normal([units]))
        self.b = tf.Variable(tf.random.normal([n_classes]))

    def call(self, inputs, training=None):
        # [None, total_words] -> [None, total_words, embed_dim]
        x = self.embed(inputs)

        # [None, total_words, embed_dim] -> [None, total_words*embed_dim]
        x = tf.reshape(x, [-1, total_words*embed_dim])

        # [None, total_words*embed_dim] -> [None, hidden_dim]
        tanh = tf.nn.tanh(self.d+tf.matmul(x, self.H))

        # [None, hidden_dim] -> [None, n_classes]
        output = tf.matmul(x, self.W) + tf.matmul(tanh, self.U) + self.b

        return output


def main():
    units = 64
    epochs = 5
    model = nnlm(units)
    model.build(input_shape=(None, total_words))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=['acc'],
                  loss=tf.losses.CategoricalCrossentropy())
    model.summary()


if __name__ == '__main__':
    main()