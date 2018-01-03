import numpy as np
from keras.utils import np_utils
import sys
import math
import os
from os import listdir
from os import mkdir
from os.path import isdir
import multiprocessing
from multiprocessing import Pool
import h5py
import shutil
from random import shuffle


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

def map_func(x):
  if x in meaningful_label_dict.keys():
    return meaningful_label_dict.get(x)
  else:
    # unknown
    return 0

def get_feature_mode(mode, file_list):
  print(mode + ' data loading!')
  # add unknown
  n_meaningful_class = len(meaningful_label) + 1
  shuffle(file_list)
  feature_vector_list = get_feature_file_list(file_list)
  feature_vector = np.empty((0, feature_vector_list[0][0].shape[0]))

  cnt = 1
  will = len(feature_vector_list)
  for feature in feature_vector_list:
    print(str(cnt) + '/' + str(will))
    # axis = 0, column appending
    feature_vector = np.append(feature_vector, feature, axis=0)
    cnt += 1

  shuffle(feature_vector)
  x = feature_vector[:, :-1]
  y = feature_vector[:, -1]
  feature_vector = None

  y = list(map(map_func, [y.decode('utf-8') for y in y]))
  y = np_utils.to_categorical(y, num_classes=n_meaningful_class)
  print(x.shape)
  print(y.shape)
  x = np.hstack([x, y])
  y = None
  print(mode + ' data loading done!')

  return x

def get_feature_file_list(file_list):
  n_processes = multiprocessing.cpu_count()
  pool = Pool(processes=n_processes)
  result = pool.map(get_feature_file, file_list)
  shuffle(result)
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
  path = os.path.join('./feature/train', sys.argv[1])

  labels = listdir(path)
  labels.sort()

  print('data processing!')

  # print(labels)
  file_list = []

  for label in labels:
    base = os.path.join(path, label)
    base_files = listdir(base)
    if sys.argv[1] in ['train']:
      if label in meaningful_label:
        pass
      else:
        shuffle(base_files)
        base_files = base_files[:int(len(base_files) / 2)]

    for file in base_files:
      file_list.append(os.path.join(base, file))

  shuffle(file_list)
  
  n = int(sys.argv[2])
  
  file_list = [file_list[i::n] for i in range(n)]
  shuffle(file_list)

  if not isdir('./hdf'):
    mkdir('./hdf')

  n_sample = 0
  for i in range(len(file_list)):
    file_chunk = file_list[i]
    x = get_feature_mode(str(i) + '\'th', file_chunk)
    n_sample += x.shape[0]
    n_x = x.shape[0]

    h5f = h5py.File(os.path.join('./hdf/', sys.argv[1] + str(i) + '.h5'), 'w')
    # h5f.create_dataset('feature', data=x[i * n_x: (i + 1) * n_x])
    shuffle(x)
    h5f.create_dataset('feature', data=x)
    h5f.close()
    # gc plz..
    file_chunk = None
    file_list[i] = None

  print(n_sample)