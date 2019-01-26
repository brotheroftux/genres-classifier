import tensorflow as tf
from tensorflow import keras
import numpy as np

class AutoEncoder:
  """ Autoencoder class """

  def __init__ (self, data_dimension=100, encoding_dimension=10, loss='mean_squared_error',
                optimizer='adadelta'):
    self.data_dimension = data_dimension
    self.encoding_dimension = encoding_dimension
    self.optimizer = optimizer
    self.loss = loss

  def initialize_model (self):
    input_layer = keras.layers.Input(shape=[self.data_dimension])
    encoded = keras.layers.Dense(self.encoding_dimension, 
                                 input_shape=[self.data_dimension], activation=tf.nn.relu)(input_layer)
    decoded = keras.layers.Dense(self.data_dimension, activation=tf.nn.sigmoid)(encoded)

    self.model = keras.Model(input_layer, decoded)

    self.encoder = keras.Model(input_layer, encoded)

    encoded_input = keras.layers.Input(shape=[self.encoding_dimension])
    self.decoder = keras.Model(encoded_input, self.model.layers[-1](encoded_input))

    self.model.compile(
      optimizer=self.optimizer,
      loss=self.loss
    )

  def run (self, train_data, validation_data, epochs=50):
    self.model.fit(train_data, train_data,
                   epochs=epochs,
                   validation_data=(validation_data, validation_data))
