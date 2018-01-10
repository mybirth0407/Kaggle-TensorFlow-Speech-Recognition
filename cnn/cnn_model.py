# Keras metrics method
from keras import metrics
# Keras perceptron neuron layer implementation.
from keras.layers import Dense
# Keras Convolution
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
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
from os import mkdir
from os.path import isdir
# Using Data Processing for Multiprocessing.
import multiprocessing
from multiprocessing import Pool
# Use List Random Shuffle. 
from random import shuffle
# Use HDF File Format.
import h5py


def main(argv):
###############################################################################

  epochs = 12
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
    else:
      train_list.append(os.path.join(argv[1], file))

  # gc plz..
  file_list = None

  x_val, y_val = get_feature_mode('val', val_list)
  x_val = x_val.reshape(x_val.shape[0], 51, 39, 1)
  print(x_val.shape)
  print(y_val.shape)

  print('data loading done!')

###############################################################################

  print(x_val.shape[1])
  print(y_val.shape[1])
  input_shape = (51, 39, 1)
  print('model constructing!')
  # model = Sequential()
  # model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape=input_shape))
  # model.add(MaxPooling2D(pool_size=(2, 2)))
  # model.add(Conv2D(32, (3, 3), activation = 'relu'))
  # model.add(MaxPooling2D(pool_size=(2, 2)))
  # model.add(Flatten())
  # model.add(Dense(100))
  # model.add(BatchNormalization())
  # model.add(Activation('relu'))
  # model.add(Dense(100))
  # model.add(BatchNormalization())
  # model.add(Activation('relu'))
  # model.add(Dense(12, activation = 'softmax')) #Last layer with one output per class

  model = Sequential()
  model.add(Conv2D(32, (3, 3), padding='same',
                   input_shape=x_train.shape[1:]))
  model.add(Activation('relu'))
  model.add(Conv2D(32, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64, (3, 3), padding='same'))
  model.add(Activation('relu'))
  model.add(Conv2D(64, (3, 3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(512))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes))
  model.add(Activation('softmax'))
  
  model.compile(
      loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"]
  )
  model.summary()
  print('model constructing done!')

###############################################################################

  print('model fit!')

  model.fit_generator(
      generator=generate_file(train_list, batch_size),
      steps_per_epoch=577,
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
  if not isdir('./model'):
    mkdir('./model')
  model.save('./model/cnn_model_' + str(time.time()) + '.h5')
  print('model save done!')

###############################################################################

  print('model evaluate!')
  x_test, y_test = get_feature_mode('test', test_list)
  x_test = x_test.reshape(x_test.shape[0], 51, 39, 1)
  # predict = model.predict(x_test, batch_size=batch_size)
  score = model.evaluate(x_test, y_test, batch_size=batch_size)
  print('model evaluate done!')

###############################################################################

  print(score)
  print('\n%s: %.2f%%' % (model.metrics_names[1], score[1] * 100))


def generate_file(file_list, batch_size):
  shuffle(file_list)
  while True:
    for file in file_list:
      h5f = h5py.File(file, 'r')
      feature = h5f['feature'][:]
      feature = feature.reshape(feature.shape[0], 51, 39, 1)
      label = h5f['label'][:]
      h5f.close()
      # gc plz..
      i = 0
      for i in range(label.shape[0] // batch_size):
        yield(
          feature[i*batch_size:(i+1)*batch_size],
          label[i*batch_size:(i+1)*batch_size]
        )

        # yield(x, y)
        i += 1

def get_feature_mode(mode, file_list):
  print(mode + ' data loading!')

  feature_vector_list = get_feature_file_list(file_list)
  print(len(feature_vector_list))
  print(feature_vector_list[0][0].shape)
  print(feature_vector_list[0][1].shape)
  # print(feature_vector_list[0][0][0].shape)
  # print(feature_vector_list[0][1][0])

  # feature_vector = np.empty((0, feature_vector_list[0][0][0].shape[0]))
  # cnt = 1
  # will = len(feature_vector_list)
  # for feature in feature_vector_list[0][0]:
  #   print(feature.shape)
  #   stop()
  #   print(str(cnt) + '/' + str(will))
  #   # axis = 0, column appending
  #   feature_vector = np.append(feature_vector[0], feature, axis=0)
  #   cnt += 1

  print(mode + ' data loading done!')

  # return features, labels
  return feature_vector_list[0][0], feature_vector_list[0][1]

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
  feature = h5f['feature'][:]
  label = h5f['label'][:] 
  h5f.close()
  print(file + ' done!')

  return (feature, label)

def stop():
  while True: pass

if __name__ == '__main__':
  main(sys.argv)

