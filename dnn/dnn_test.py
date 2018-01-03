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
# Use List Random Shuffle. 
from random import shuffle
# Use HDF File Format.
import h5py

# labels
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

def gen3digit(x):
  return [f"{i:03}" for i in [x]]

def most_common(l):
    return max(set(l), key=l.count)

if __name__ == '__main__':
  model = load_model(sys.argv[1])
  feature_path = './feature/test/'

  file_list = []

  for file in listdir(feature_path):
    file_list.append(os.path.join(feature_path, file))
  file_list.sort()

  n = 100
  file_list = [file_list[i::n] for i in range(n)]

  if not isdir('./pred'):
    mkdir('./pred')

  import time
  f = open('./pred/pred_' + str(time.time()) +  '.csv', 'w')
  f.write('fname,label\n')

  for i in range(len(file_list)):
    feature_vector_list = get_feature_file_list(file_list[i])

    cnt = 1
    will = len(feature_vector_list)
    for feature in feature_vector_list:
      print(str(cnt) + '/' + str(will))
      # axis = 0, column appending
      predict = model.predict(feature, batch_size=256)
      label_index = list(np.argmax(predict, axis=1))
      y = max(set(label_index), key=label_index.count)
      y = int_dict.get(y)
      cnt += 1
    
    for file in file_list[i]:
      fname = os.path.basename(os.path.normpath(file))
      fname, _ = os.path.splitext(fname)
      fname = fname + '.wav'

      f.write(fname + ',' + str(y) + '\n')

  f.close()