import numpy as np
import h5py
import sys
from os import listdir
import os

def cnt(file_list, cnt_list):
  for file in file_list:
    print(file + ' start!')
    h5f = h5py.File(file, 'r')
    feature_vector = h5f['feature'][:]
    h5f.close()
    print(file + ' done!')

    cnt_list.append(feature_vector.shape[0])

  return cnt_list


if __name__ == '__main__':
  file_list = listdir(sys.argv[1])

  val_cnt = []
  test_cnt = []
  train_cnt = []

  val_list = []
  test_list = []
  train_list = []

  for file in file_list:
    if file.find('val') != -1:
      val_list.append(os.path.join(sys.argv[1], file))
    elif file.find('test') != -1:
      test_list.append(os.path.join(sys.argv[1], file))
    else:
      train_list.append(os.path.join(sys.argv[1], file))

  val = cnt(val_list, val_cnt)
  test = cnt(test_list, test_cnt)
  train = cnt(train_list, train_cnt)

  print('val')
  print(sum(val))
  print('test')
  print(sum(test))
  print('train')
  print(sum(train))
