"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 
Usage:
     python capsulenet.py
     python capsulenet.py --epochs 50
     python capsulenet.py --epochs 50 --routings 3
     ... ...
     
Result:
  Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
  About 110 seconds per epoch on a single GTX1070 GPU card
  
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import numpy as np
from keras import layers, models
from keras.optimizers import Adam
from keras import backend as K
from keras.models import load_model
from keras.utils import to_categorical
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

import numpy as np
import sys
import matplotlib.pyplot as plt
import os
from os import listdir
from os import mkdir
from os.path import isdir
import multiprocessing
from multiprocessing import Pool
from random import shuffle
import h5py

epochs = 1
batch_size = 256
frame_size = 51
use_mel = 40
use_mfcc = 39
routings = 3
n_classes = 12
recon = 0.392
lr = 0.001
decay = 0.9

use_mel = 40
use_mfcc = 39

use = sys.argv[3]

def main(argv):
###############################################################################

  print('data loading!')
  file_list = listdir(argv[1])
  shuffle(file_list)

  train_list = []
  
  for file in file_list:
    if file.find('val') != -1:
      train_list.append(os.path.join(argv[1], file))
    elif file.find('test') != -1:
      train_list.append(os.path.join(argv[1], file))

  # gc plz..
  file_list = None

  x_train, y_train = get_feature_mode('val and test', train_list)
  if use == 'mel':
    x_train = x_train.reshape(x_train.shape[0], frame_size, use_mel, 1)
  elif use == 'mfcc':
    x_train = x_train.reshape(x_train.shape[0], frame_size, use_mfcc, 1)
  print(x_train.shape)
  print(y_train.shape)

  print('data loading done!')

###############################################################################

  print('model weight loading!')
  
  print(x_train.shape[1])
  print(y_train.shape[1])
  input_shape = (frame_size, use_mfcc, 1)

  # define model
  model = CapsNet(input_shape=input_shape, n_class=n_classes, routings=routings)
  model.summary()

  model.load_weights(argv[2])
  # compile the model
  model.compile(
      optimizer=Adam(lr=lr),
      loss=[margin_loss, 'mse'],
      loss_weights=[1., recon],
      metrics={'capsnet': 'accuracy'}
  )

  print('model weight loading done!')

###############################################################################

  print('callback define!')
  time_stamp = os.path.dirname(argv[2])
  file_name = os.path.basename(os.path.normpath(argv[2]))
  file_name, _ = os.path.splitext(file_name)
  save_dir = os.path.join(os.getcwd(), time_stamp)
  model_name = file_name + '.{epoch:03d}.h5'

  if not isdir(save_dir):
    mkdir(save_dir)
  filepath = os.path.join(save_dir, model_name)

  # Prepare callbacks for model saving and for learning rate adjustment.
  checkpoint = ModelCheckpoint(filepath=filepath,
                               monitor='capsnet_acc',
                               verbose=1,
                               save_best_only=True)

  # callbacks

  lr_decay = LearningRateScheduler(
      schedule=lambda epoch: lr * (decay ** epoch)
  )

  callbacks = [checkpoint, lr_decay]

  print('callback define done!')

###############################################################################

  print('model fit!')
  
  # gc
  x_train = None
  y_train = None

  model.fit_generator(
      generator=generate_file(train_list, batch_size),
      steps_per_epoch=53,
      epochs=epochs,
      use_multiprocessing=True,
      verbose=1,
      shuffle=True,
      callbacks=callbacks
  )


  print('model fit done!')
  
    
def generate_file(file_list, batch_size):
  shuffle(file_list)
  while True:
    for file in file_list:
      h5f = h5py.File(file, 'r')
      if use == 'mel':
        feature = h5f['feature'][:,:,use_mel - 1:]
        feature = feature.reshape(feature.shape[0], frame_size, use_mel, 1)
      elif use == 'mfcc':
        feature = h5f['feature'][:,:,:use_mfcc]
        feature = feature.reshape(feature.shape[0], frame_size, use_mfcc, 1)

      label = h5f['label'][:]
      h5f.close()
      # gc plz..
      i = 0
      for i in range(label.shape[0] // batch_size):
        yield(
          [[feature[i*batch_size:(i+1)*batch_size],
          label[i*batch_size:(i+1)*batch_size]],

          [label[i*batch_size:(i+1)*batch_size],
          feature[i*batch_size:(i+1)*batch_size]]]
        )

        # yield(x, y)
        i += 1

def get_feature_mode(mode, file_list):
  print(mode + ' data loading!')

  feature_vector_list = get_feature_file_list(file_list)
  print(len(feature_vector_list))
  print(feature_vector_list[0][0].shape)
  print(feature_vector_list[0][1].shape)

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
  if use == 'mel':
    feature = h5f['feature'][:,:,use_mel - 1:]
  elif use =='mfcc':
    feature = h5f['feature'][:,:,:use_mfcc]

  label = h5f['label'][:] 
  h5f.close()
  print(file + ' done!')

  return (feature, label)

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
  train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])

  return train_model


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

if __name__ == "__main__":
  main(sys.argv)
