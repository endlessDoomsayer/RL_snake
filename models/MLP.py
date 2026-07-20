import tensorflow as tf
from tensorflow.keras import layers

class MLP(tf.keras.Model):
    def __init__(self, output_dim, activation='swish'):
        super().__init__()
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128, activation=activation)
        self.d2 = layers.Dense(64, activation=activation)
        self.d3 = layers.Dense(32, activation=activation)
        self.out = layers.Dense(output_dim)

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.out(x)