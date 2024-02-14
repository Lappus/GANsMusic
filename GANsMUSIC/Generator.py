import tensorflow as tf

class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__()

        self.layer_list = [
          tf.keras.layers.Dense(4*4*128, use_bias=False),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.LeakyReLU(),
          tf.keras.layers.Reshape((4, 4, 128)),

          tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=(2, 2), padding='same', use_bias=False),
          #tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.LeakyReLU(),

          tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=(2, 2), padding='same', use_bias=False),
          #tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.LeakyReLU(),

          tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=4, strides=(2, 2), padding='same', use_bias=False),
          #tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.LeakyReLU(),

          tf.keras.layers.Conv2DTranspose(filters=8, kernel_size=4, strides=(2, 2), padding='same', use_bias=False),
          #tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.LeakyReLU(),

          tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        ]

        self.metric_loss = tf.keras.metrics.Mean(name="loss")
     
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self.noise_dim = 100

    @tf.function
    def call(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x

