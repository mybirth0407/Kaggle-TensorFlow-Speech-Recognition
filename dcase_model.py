# Numeric Python Library.
import numpy as np
# Keras metrics method
from keras import metrics
# Keras perceptron neuron layer implementation.
from keras.layers import Dense
# Keras Dropout layer implementation.
from keras.layers import Dropout
# Keras Activation Function layer implementation.
from keras.layers import Activation
# Keras Model object.
from keras.models import Sequential
# Keras Optimizer for custom user.
from keras import optimizers
# Keras Loss for custom user.
from keras import losses

from keras.utils import np_utils

# Get System Argument.
import sys
# Use Python Math
import math
# Use K-fold
from sklearn.model_selection import KFold
# Use plot
import matplotlib.pyplot as plt

from os import listdir

import multiprocessing
from multiprocessing import Pool

import h5py

# feature file paths
train_feature_path = './feature/train/train/'
val_feature_path = './feature/train/validation/'
test_feature_path = './feature/train/test/'


# labels
meaningful_label = ['down', 'go', 'left', 'no', 'off',
                    'on', 'right', 'silence', 'stop', 'up', 'yes']

# except label
except_label = ['unknown']


def main(argv):
  meaningful_label.sort()
  label_dict = {}

  epochs = 200
  batch_size = 256

  n_except = 0
  for (i, label) in zip(range(len(except_label)), except_label):
    label_dict[label] = i
    n_except += 1

  for (i, label) in zip(range(n_except, len(meaningful_label)), meaningful_label):
    label_dict[label] = i

  x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(label_dict)
  # x_test, y_test =  split_dataset(label_dict)

  print(x_train.shape)
  print(y_train.shape)
  print(x_val.shape)
  print(y_val.shape)
  print(x_test.shape)
  print(y_test.shape)
  
  print("model constructing!")
  model = Sequential()
  model.add(Dense(x_train, input_dim=x_train[0].shape[1],
      kernel_initializer='uniform', activation='relu'))
  model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(50, kernel_initializer='uniform', activation='relu'))
  model.add(Dropout(0.2))

  model.add(Dense(12, activation='sofmax'))

  adam = optimizers.Adam(
      lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1
  )
  model.compile(
      loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']
  )
  print("model constructing done!")
  print("model fit!")
  model.fit(
    x_train, y_train, epochs=epochs, batch_size=batch_size,
    show_accuracy=True, validation_data=(x_val, y_val), shuffle=True
  )
  print("model fit done!")
  print("model save!")
  model.save('dcase_model.h5')
  print("model save done!")

  score = model.evaluate(x_test, y_test)
  print(score)
  print("\n%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

def split_dataset(label_dict):
  n_processes = multiprocessing.cpu_count()
  pool = Pool(processes=n_processes)

  result = pool.map(get_feature, ['train', 'val', 'test'])

  train_feature_vector = result[0]
  val_feature_vector = result[1]
  test_feature_vector = result[2]

  n_classes = len(label_dict)

  x_train = train_feature_vector[:, :-1]
  y_train = train_feature_vector[:, -1]
  
  for i in range(len(y_train)):
    y = y_train[i].decode('utf-8')
    y_train[i] = label_dict[y]
  y_train = np_utils.to_categorical(y_train, num_classes=n_classes)

  x_val = val_feature_vector[:, :-1]
  y_val = val_feature_vector[:, -1]

  for y in range(len(y_val)):
    y = y_val[i].decode('utf-8')
    y_val[i] = label_dict[y]
  y_val = np_utils.to_categorical(y_val, num_classes=n_classes)

  x_test = test_feature_vector[:, :-1]
  y_test = test_feature_vector[:, -1]

  for y in range(len(y_test)):
    y = y_test[i].decode('utf-8')
    y_test[i] = label_dict[y]
  y_test = np_utils.to_categorical(y_test, num_classes=n_classes)


  return x_train, y_train, x_val, y_val, x_test, y_test
  # return x_test, y_test

def get_feature(arg):
  mode = arg
  if 'train' == mode:
    path = train_feature_path
  elif 'val' == mode:
    path = val_feature_path
  elif 'test' == mode:
    path = test_feature_path
  else:
    print('error!')
    sys.exit(0)

  files = listdir(path)

  feature_test = listdir(path + files[0])[0]

  h5f = h5py.File(path + files[0] + '/' + feature_test)
  feature_vector = np.empty((0, len(h5f['feature'][0])))
  h5f.close()

  print(mode + ' feature extract!')
  cnt = 1
  for label in files:
    for file in listdir(path + label):
      if 0 == cnt % 100:
        print(cnt)
      h5f = h5py.File(path + label + '/' + file, 'r')
      feature = h5f['feature'][:]
      h5f.close()
      feature_vector = np.vstack((feature_vector, feature))

      cnt += 1
  print(mode + ' feature extract done!')

  return feature_vector

if __name__ == '__main__':
  main(sys.argv)

