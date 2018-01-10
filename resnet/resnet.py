from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
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

  n = 3
  depth = n * 9 + 2
  model = resnet_v2(input_shape=input_shape, depth=depth, num_classes=12)
  model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
  model.summary()

###############################################################################

  print('callback define!')
  import time
  now = time.localtime()
  time_stamp = '%02d%02d%02d' % (now.tm_mday, now.tm_hour, now.tm_min)
  save_dir = os.path.join(os.getcwd(), 'saved_models_' + time_stamp)
  model_name = 'resnet_v2_model.{epoch:03d}.h5'

  if not isdir(save_dir):
    mkdir(save_dir)
  filepath = os.path.join(save_dir, model_name)

  # Prepare callbacks for model saving and for learning rate adjustment.
  checkpoint = ModelCheckpoint(filepath=filepath,
                               monitor='val_acc',
                               verbose=1,
                               save_best_only=True)

  lr_scheduler = LearningRateScheduler(lr_schedule)

  lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                 cooldown=0,
                                 patience=5,
                                 min_lr=0.5e-6)

  callbacks = [checkpoint, lr_reducer, lr_scheduler]

  print('callback define done!')

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
      shuffle=True,
      callbacks=callbacks
  )

  # gc
  x_val = None
  y_val = None

  print('model fit done!')
  
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

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)

    return lr

def resnet_block(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            activation-bn-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    x = inputs
    if conv_first:
        x = Conv2D(num_filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation:
            x = Activation(activation)(x)
        return x
    if batch_normalization:
        x = BatchNormalization()(x)
    if activation:
        x = Activation('relu')(x)
    x = Conv2D(num_filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(x)

    return x

def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    Features maps sizes: 16(input), 64(1st sub_block), 128(2nd), 256(3rd)
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    inputs = Input(shape=input_shape)
    num_filters_in = 16
    num_filters_out = 64
    filter_multiplier = 4
    num_sub_blocks = int((depth - 2) / 9)

    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_block(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate convolutional base (stack of blocks).
    activation = None
    batch_normalization = False
    for i in range(3):
        if i > 0:
            filter_multiplier = 2
        num_filters_out = num_filters_in * filter_multiplier

        for j in range(num_sub_blocks):
            strides = 1
            is_first_layer_but_not_first_block = j == 0 and i > 0
            if is_first_layer_but_not_first_block:
                strides = 2
            y = resnet_block(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            activation = 'relu'
            batch_normalization = True
            y = resnet_block(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_block(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if j == 0:
                x = resnet_block(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def stop():
  while True: pass

if __name__ == '__main__':
  main(sys.argv)
