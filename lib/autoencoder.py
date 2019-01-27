import tensorflow as tf
from tensorflow import keras
import numpy as np

class AutoEncoder:
  """ Autoencoder class """

  def __init__ (self, data_dimension=100, encoding_dimension=10, classes=5,
                loss='mean_squared_error', optimizer='adam', verbose=True):
    self.data_dimension = data_dimension
    self.encoding_dimension = encoding_dimension
    self.classes = classes
    self.optimizer = optimizer
    self.loss = loss
    self.verbose = verbose

  def initialize_model (self):
    input_layer = keras.layers.Input(shape=[self.data_dimension])
    encoded = keras.layers.Dense(self.encoding_dimension, activation=tf.nn.relu)(input_layer)
    decoded = keras.layers.Dense(self.data_dimension, activation=tf.nn.sigmoid)(encoded)

    self.model = keras.Model(input_layer, decoded)

    self.encoder = keras.Model(input_layer, encoded)

    encoded_input = keras.layers.Input(shape=[self.encoding_dimension])
    self.decoder = keras.Model(encoded_input, self.model.layers[-1](encoded_input))

    self.model.compile(
      optimizer=self.optimizer,
      loss=self.loss,
      metrics=['accuracy']
    )

    self.input_layer = input_layer
    self.encoded_layer = encoded

  def extend_model (self, loss_function='binary_crossentropy'):
    self.encoded_layer.trainable = False
    output_layer = keras.layers.Dense(self.classes, activation='sigmoid')(self.encoded_layer)

    self.full_model = keras.Model(self.input_layer, output_layer)

    self.full_model.compile(
      optimizer=self.optimizer,
      loss='binary_crossentropy',
      metrics=['accuracy']
    )
    
  def fit (self, train_data, validation_data, epochs=40):
    self.model.fit(train_data, train_data,
                   epochs=epochs,
                   validation_data=(validation_data, validation_data),
                   verbose=self.verbose)

  def fit_full_model (self, train_data, train_labels, test_data, test_labels, epochs=40):
    self.full_model.fit(train_data, train_labels, epochs=epochs, verbose=self.verbose)

    test_loss, test_acc = self.full_model.evaluate(test_data, test_labels)
    print('Test accuracy: {}, test loss: {}'.format(test_acc, test_loss))