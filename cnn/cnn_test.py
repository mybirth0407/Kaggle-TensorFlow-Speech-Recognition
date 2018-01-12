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

batch_size = 256
frame_size = 51
use_mel = 40
use_mfcc = 39

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


def get_feature_file_list(file_list):
  n_processes = multiprocessing.cpu_count()
  pool = Pool(processes=n_processes)
  result = pool.map(get_feature_file, file_list)
  pool.close()
  pool.join()

  return result

def get_feature_file(file):
  print(file + ' start!')
  h5f = h5py.File(file, 'r')
  feature = h5f['feature'][:, :use_mfcc]
  h5f.close()
  print(file + ' done!')

  return feature

def most_common(l):
    return max(set(l), key=l.count)

if __name__ == '__main__':
  model = load_model(sys.argv[1])
  feature_path = '../feature/test/'

  file_list = []

  for file in listdir(feature_path):
    file_list.append(os.path.join(feature_path, file))
  file_list.sort()

  n = 100
  file_list = [file_list[i::n] for i in range(n)]

  import time
  base_name = os.path.basename(sys.argv[1])
  pred_file = os.path.basename(os.path.normpath(sys.argv[1]))
  pred_file, _ = os.path.splitext(pred_file)
  pred_file = os.path.join(base_name, pred_file)
  f = open(pred_file + '.csv', 'w')
  f.write('fname,label\n')

  for i in range(len(file_list)):
    feature_list = get_feature_file_list(file_list[i])
    cnt = 1
    will = len(feature_list)
    labels = []

    #########
    for j in range(will):
      feature = feature_list[j]
      # print(feature.shape)
      feature = feature.reshape(1, frame_size, use_mfcc, 1)
      print(str(cnt) + '/' + str(will))
      # axis = 0, column appending
      predict = model.predict(feature, batch_size=256)
      label_index = list(np.argmax(predict, axis=1))
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