import tensorflow as tf
import numpy as np
import data.dataset as data

from lib.autoencoder import AutoEncoder
from lib.colors import colors

from functools import partial

__dim__ = 100

known_labels = [
    'K-Pop', 
    'Drum and Bass', 
    'Symphonic Metal', 
    'Trance',
    'Progressive Rock'
  ]

def transform_labels (labels):
  global known_labels

  return [1 if x in labels else 0 for x in known_labels]

def decode_labels (labels, threshold=0.5):
  global known_labels

  return [known_labels[i] for i in range(len(labels)) if labels[i] > threshold]

def transform_input_one_hot (input, depth):
  """
  Transforms a vector of features into a one-hot vector.

  Ex:

  input = [3, 4, 1, 1]
  depth = 5

  # [0, 1, 0, 1, 1]
  """

  result = tf.zeros(depth)

  for scalar in input:
    result += tf.one_hot(scalar, depth)

  return result

""" Main stub """

print(colors.BOLD, 'Loading dataset...', colors.ENDC)

data_set, labels = data.get_data()

print(colors.BOLD, 'Transforming data...', colors.ENDC, end='')

data_set = list(map(partial(transform_input_one_hot, depth=__dim__), data_set))
data_set = tf.stack(data_set)
data_set = tf.map_fn(lambda t: t / tf.reduce_max(t), data_set)

train = data_set[:800]
validation = data_set[800:]

tf.InteractiveSession()

train_np = train.eval()
validation_np = validation.eval()

print(colors.OKGREEN, 'OK', colors.ENDC)

ae = AutoEncoder(data_dimension=__dim__, encoding_dimension=25, verbose=True)

ae.initialize_model()

print(colors.BOLD, 'Fitting autoencoder model...', colors.ENDC)
ae.fit(train_np, validation_np, epochs=300)
print(colors.OKGREEN, 'Autoencoder fitting done.', colors.ENDC)

ae.extend_model(loss_function='binary_crossentropy')

labels = np.array(list(map(transform_labels, labels)))

train_labels = labels[:800]
test_labels = labels[800:]

print(colors.BOLD, 'Fitting full model...', colors.ENDC)
ae.fit_full_model(train_np, train_labels, 
                  test_data=validation_np,
                  test_labels=test_labels,
                  epochs=100)
print(colors.OKGREEN, 'Training complete.', colors.ENDC)

labels = ae.full_model.predict(validation_np[-5:])
demo_labels = test_labels[-5:]

print()
print(colors.OKBLUE, '=======', colors.ENDC)
print(colors.HEADER, 'Training results:', colors.ENDC)
print()

for idx in range(len(demo_labels)):
  print(colors.BOLD, 'Expected:', colors.ENDC, end='')
  print(', '.join(decode_labels(demo_labels[idx])))

  print(colors.BOLD, 'Got:', colors.ENDC, end='')
  print(', '.join(decode_labels(labels[idx])))
  print()