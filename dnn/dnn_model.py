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
# Using Data Processing for Multiprocessing.
import multiprocessing
from multiprocessing import Pool
# Use List Random Shuffle. 
from random import shuffle
# Use HDF File Format.
import h5py


def main(argv):
###############################################################################

  epochs = 200
  batch_size = 256

###############################################################################

  print('data loading!')
  file_list = listdir(argv[1])
  shuffle(file_list)

  val_list = []
  test_list = []
  train_list = []
  for file in file_list:
    if file.find('val') != -1:
      val_list.append(os.path.join(argv[1], file))
    elif file.find('test') != -1:
      test_list.append(os.path.join(argv[1], file))
    elif file.find('train') != -1:
      train_list.append(os.path.join(argv[1], file))

  # gc plz..
  file_list = None

  x_val, y_val = get_feature_mode('val', val_list)

  print('data loading done!')

###############################################################################

  print(x_val.shape[1])
  print(y_val.shape[1])
  print('model constructing!')
  model = Sequential()
  model.add(Dense(x_val.shape[1], input_dim=x_val.shape[1],
      kernel_initializer='random_uniform', activation='relu'))
  model.add(Dense(100, kernel_initializer='random_uniform'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dense(100, kernel_initializer='random_uniform'))
  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dense(y_val.shape[1], activation='softmax'))

  adam = optimizers.Adam(
      lr=0.0003
  )
  model.compile(
      loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy']
  )
  print('model constructing done!')

###############################################################################

  print('model fit!')

  model.fit_generator(
      generator=generate_file(train_list, batch_size),
      steps_per_epoch=5675,
      validation_data=(x_val, y_val),
      validation_steps=y_val.shape[0] // batch_size,
      epochs=epochs,
      use_multiprocessing=True,
      verbose=1,
      shuffle=True
  )

  # gc
  x_val = None
  y_val = None
  print('model fit done!')
  
###############################################################################

  print('model save!')
  import time
  model.save('dcase_model_' + str(time.time()) + '.h5')
  print('model save done!')

###############################################################################

  print('model evaluate!')
  x_test, y_test = get_feature_mode('test', test_list)
  predict = model.predict(x_test, batch_size=batch_size)
  score = model.evaluate(x_test, y_test, batch_size=batch_size)
  print('model evaluate done!')

###############################################################################

  print(score)
  print('\n%s: %.2f%%' % (model.metrics_names[1], score[1] * 100))


def generate_file(file_list, batch_size):
  while True:
    for file in file_list:
      h5f = h5py.File(file, 'r')
      feature_vector = h5f['feature'][:]
      h5f.close()

      shuffle(feature_vector)
      x = feature_vector[:, :-12]
      y = feature_vector[:, -12:]
      # gc plz..
      feature_vector = None
      i = 0
      for i in range(y.shape[0] // batch_size):
        yield(x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])

        # yield(x, y)
        i += 1

def get_feature_mode(mode, file_list):
  print(mode + ' data loading!')

  feature_vector_list = get_feature_file_list(file_list)
  feature_vector = np.empty((0, feature_vector_list[0][0].shape[0]))

  cnt = 1
  will = len(feature_vector_list)
  for feature in feature_vector_list:
    print(str(cnt) + '/' + str(will))
    # axis = 0, column appending
    feature_vector = np.append(feature_vector, feature, axis=0)
    cnt += 1

  x = feature_vector[:, :-12]
  y = feature_vector[:, -12:]
  feature_vector = None
  print(mode + ' data loading done!')

  return x, y

def get_feature_file_list(file_list):
  n_processes = multiprocessing.cpu_count()
  pool = Pool(processes=n_processes)
  result =  pool.map(get_feature_file, file_list)
  pool.close()
  pool.join()

  return result

def get_feature_file(file):
  print(file + ' start!')
  h5f = h5py.File(file, 'r')
  feature_vector = h5f['feature'][:]
  h5f.close()
  print(file + ' done!')

  return feature_vector

def stop():
  while True: pass

if __name__ == '__main__':
  main(sys.argv)

