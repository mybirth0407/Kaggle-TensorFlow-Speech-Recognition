# Keras metrics method
from keras import metrics
# Keras perceptron neuron layer implementation.
from keras.layers import Dense
# Keras Dropout layer implementation.
from keras.layers import Dropout
# Keras Activation Function layer implementation.
from keras.layers import Activation
# Keras Batch normalization
from keras.layers.normalization import BatchNormalization
# Keras Model object.
from keras.models import Sequential
# Keras Model load
from keras.models import load_model
# Keras Optimizer for custom user.
from keras import optimizers
# Keras Loss for custom user.
from keras import losses
# Numeric Python Library.
import numpy as np
# Get System Argument.
import sys
# Use plot.
import matplotlib.pyplot as plt
# Use Os Function.
import os
from os import listdir
from os import mkdir
from os.path import isdir
# Using Data Processing for Multiprocessing.
import multiprocessing
from multiprocessing import Pool
# Use HDF File Format.
import h5py

from keras import layers, models
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask

batch_size = 256
frame_size = 51
use_mel = 40
use_mfcc = 39
input_shape = (frame_size, use_mfcc, 1)
n_classes = 12
routings = 3


# mearningful labels
meaningful_label = ['down', 'go', 'left', 'no', 'off',
                    'on', 'right', 'silence', 'stop', 'up', 'yes']
meaningful_label.sort()
# except label
except_label = ['unknown']

meaningful_label_dict = {}

n_except = 0
for (i, label) in zip(range(len(except_label)), except_label):
  meaningful_label_dict[label] = i
  n_except += 1

for (i, label) in zip(range(n_except, len(meaningful_label)), meaningful_label):
  meaningful_label_dict[label] = i


int_dict = dict(zip(
    meaningful_label_dict.values(), meaningful_label_dict.keys()
))

use = sys.argv[2]


def get_feature_file_list(file_list):
  n_processes = multiprocessing.cpu_count()
  pool = Pool(processes=n_processes)
  result = pool.map(get_feature_file, file_list)
  pool.close()
  pool.join()

  return result

def get_feature_file(file):
  # print(file + ' start!')
  h5f = h5py.File(file, 'r')
  if use == 'mel':
    feature = h5f['feature'][:,use_mel - 1:]
  elif use == 'mfcc':
    feature = h5f['feature'][:,:use_mfcc]
  h5f.close()
  # print(file + ' done!')

  return feature

def most_common(l):
    return max(set(l), key=l.count)

def CapsNet(input_shape, n_class, routings):
  """
  A Capsule Network on MNIST.
  :param input_shape: data shape, 3d, [width, height, channels]
  :param n_class: number of classes
  :param routings: number of routing iterations
  :return: Two Keras Models, the first one used for training, and the second one for evaluation.
      `eval_model` can also be used for training.
  """
  x = layers.Input(shape=input_shape)

  # Layer 1: Just a conventional Conv2D layer
  conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

  # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
  primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

  # Layer 3: Capsule layer. Routing algorithm works here.
  digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
               name='digitcaps')(primarycaps)

  # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
  # If using tensorflow, this will not be necessary. :)
  out_caps = Length(name='capsnet')(digitcaps)

  # Decoder network.
  y = layers.Input(shape=(n_class,))
  masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
  masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

  # Shared Decoder model in training and prediction
  decoder = models.Sequential(name='decoder')
  decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
  decoder.add(layers.Dense(1024, activation='relu'))
  decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
  decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

  # Models for training and evaluation (prediction)
  test_model = models.Model(x, [out_caps, decoder(masked)])

  return test_model


def margin_loss(y_true, y_pred):
  """
  Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
  :param y_true: [None, n_classes]
  :param y_pred: [None, num_capsule]
  :return: a scalar loss value.
  """
  L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
    0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

  return K.mean(K.sum(L, 1))


if __name__ == '__main__':
  model = CapsNet(input_shape=input_shape, n_class=n_classes, routings=routings)
  model.summary()
  model.load_weights(sys.argv[1])

  feature_path = '../feature/test/'

  file_list = []

  for file in listdir(feature_path):
    file_list.append(os.path.join(feature_path, file))
  file_list.sort()

  n = 100
  file_list = [file_list[i::n] for i in range(n)]

  dir_name = os.path.dirname(sys.argv[1])
  pred_file = os.path.basename(os.path.normpath(sys.argv[1]))
  pred_file, _ = os.path.splitext(pred_file)
  pred_file = os.path.join(dir_name, pred_file)
  f = open(pred_file + '.csv', 'w')
  f.write('fname,label\n')

  for i in range(len(file_list)):
    print(str(i) + '/' + str(len(file_list)))
    feature_list = get_feature_file_list(file_list[i])
    cnt = 1
    will = len(feature_list)
    labels = []

    for j in range(will):
      feature = feature_list[j]
      # print(feature.shape)
      if use == 'mel':
        feature = feature.reshape(1, frame_size, use_mel, 1)
      elif use == 'mfcc':
        feature = feature.reshape(1, frame_size, use_mfcc, 1)
      # print(str(cnt) + '/' + str(will))
      # axis = 0, column appending
      y_pred, x_recon = model.predict(feature, batch_size=256)
      label_index = list(np.argmax(y_pred, axis=1))
      y = max(set(label_index), key=label_index.count)
      y = int_dict.get(y)
      if y in meaningful_label:
        pass
      else:
        y = except_label[0]
      cnt += 1
      labels.append(y)
    
    for j in range(will):
      file = file_list[i][j]
      fname = os.path.basename(os.path.normpath(file))
      fname, _ = os.path.splitext(fname)
      fname = fname + '.wav'

      f.write(fname + ',' + str(labels[j]) + '\n')

  f.close()