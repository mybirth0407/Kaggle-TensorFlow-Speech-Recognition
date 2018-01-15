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

# Tensorflow's
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import rnn

# Times
import datetime
import time

# Hyperparameter, Some parameters are hard-linked
init_learning_rate = 1e-3
epsilon = 1e-8 # AdamOptimizer epsilon
dropout_rate = 0.2

# Label 
class_num = 12

# Total epochs
total_epochs = 100 #12

# CNN Network Model

class CNNLSTM():
    def __init__(self, x, training):
        self.training = training
        self.model = self.build_net(x)

    def CNN_model(self, x):
        # use a convolutional neural net.
        # Last dimension is for "features" - there are two, melspectrogram and mel-delta spectrogram

        # First convolutional layer - maps two image to 64 feature maps.
        h_conv1 = tf.nn.relu(conv_layer(x, filter=64, kernel=[3,3], stride=1, layer_name='conv1')) 
        h_conv1 = Drop_out(h_conv1, rate=dropout_rate, training=self.training)
        # Pooling layer - downsamples by 2X: to 26*29
        h_pool1 = max_pool_2x2(h_conv1)

        # Second convolutional layer -- maps 64 feature maps to 128.
        h_conv2 = tf.nn.relu(conv_layer(h_pool1, filter=128, kernel=[3,3], stride=1, layer_name='conv2')) 
        h_conv2 = Drop_out(h_conv2, rate=dropout_rate, training=self.training)
        # Second pooling layer: to 13*10
        h_pool2 = max_pool_2x2(h_conv2)

        # Third convolutional layer -- maps 128 feature maps to 256.
        h_conv3 = tf.nn.relu(conv_layer(h_pool2, filter=256, kernel=[3,3], stride=1, layer_name='conv3')) 
        # Third pooling layer: to 13*5
        h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

        # Fourth convolutional layer -- maps 256 feature maps to 512.
        h_conv4 = tf.nn.relu(conv_layer(h_pool3, filter=512, kernel=[3,3], stride=1, layer_name='conv4')) 
        # Second pooling layer: to 13*3
        h_pool4 = tf.nn.max_pool(h_conv4, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

        # Fourth convolutional layer -- maps 256 feature maps to 512.
        h_conv5 = tf.nn.relu(conv_layer(h_pool4, filter=512, kernel=[3,3], stride=1, layer_name='conv5')) 
        # Second pooling layer: to 13*1
        h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1, 3, 1, 1], strides=[1, 3, 1, 1], padding='SAME')

        # features.
        h_out = tf.nn.relu(h_pool5)

        return h_out

    def __map_to_sequence(self, inputdata):
        shape = inputdata.get_shape().as_list()
        assert shape[1] == 1  # H of the feature map must equal to 1
        return tf.squeeze(inputdata, axis=1)

    def __sequence_label(self, inputdata):
        # Parameters hard-linked
        hidden_nums = 256
        class_num = 12

        with tf.variable_scope('LSTMLayers'):

            # unstack to get a list od 'timesteps' tensors
            inputdata = tf.unstack(inputdata, 13, 1)

            # construct stack lstm rcnn layer
            # forward lstm cell
            fw_cell = rnn.BasicLSTMCell(hidden_nums, forget_bias=1.0) 
            # Backward lstm cell
            bw_cell = rnn.BasicLSTMCell(hidden_nums, forget_bias=1.0) 

            lstm_layer, _, _ = rnn.static_bidirectional_rnn(fw_cell, bw_cell, inputdata, dtype=tf.float32)
            lstm_layer = Drop_out(lstm_layer, rate=0.5, training=self.training)

            w = tf.Variable(tf.truncated_normal([2*hidden_nums, class_num], stddev=0.1), name="w")
            b = tf.Variable(tf.truncated_normal([class_num], stddev=0.1), name="b")

            logits = tf.matmul(lstm_layer[-1], w) + b 

        return logits

    def build_net(self, inputdata):

        # first apply the cnn feature extraction stage
        cnn_out = self.CNN_model(x=inputdata)

        # second apply the map to sequence stage
        sequence = self.__map_to_sequence(inputdata=cnn_out)

        # third apply the sequence label stage
        net_out = self.__sequence_label(inputdata=sequence)

        return net_out


def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, 
                                filters=filter, 
                                kernel_size=kernel, 
                                strides=stride, 
                                padding='SAME')
        return network

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')


###############################################################################

def main(argv):

    epochs = 12
    batch_size = 256


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
    x_val = np.transpose(x_val, (0, 2, 1, 3)) # Change shape
    print("x_val.shape: ", x_val.shape)
    print("y_val.shape: ", y_val.shape)

    print('data loading done!')

###############################################################################

    X = tf.placeholder(tf.float32, shape=[None, 39, 51, 1])
    label = tf.placeholder(tf.float32, shape=[None, 12])

    training_flag = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    logits = CNNLSTM(x=X, training=training_flag).model
    prediction = tf.nn.softmax(logits)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
    train = optimizer.minimize(cost)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('loss', cost)
    tf.summary.scalar('accuracy', accuracy)

    saver = tf.train.Saver(tf.global_variables())

###############################################################################

    with tf.Session() as sess:
        if not isdir('./model'):
            mkdir('./model')

        ckpt = tf.train.get_checkpoint_state('./model')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("LOADING\n\n")
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./logs', sess.graph)

        global_step = 0
        epoch_learning_rate = init_learning_rate
        for epoch in range(total_epochs):
            total_steps = 1000
            train_generator = generate_file(train_list, 256)
            for step in range(total_steps):
                batch_x, batch_y = next(train_generator)

                train_feed_dict = {
                    X: batch_x,
                    label: batch_y,
                    learning_rate: epoch_learning_rate,
                    training_flag : True
                }

                _, loss = sess.run([train, cost], feed_dict=train_feed_dict)

                if step % 10 == 0:
                    val_feed_dict = {
                        X: x_val,
                        label: y_val,
                        learning_rate: epoch_learning_rate,
                        training_flag : False
                    }

                    global_step += 10
                    train_summary, train_accuracy = sess.run([merged, accuracy], 
                                                        feed_dict=train_feed_dict)
                    val_loss, val_accuracy = sess.run([cost, accuracy], 
                                                        feed_dict=val_feed_dict)
                    # accuracy.eval(feed_dict=feed_dict)
                    print("Step: %.3d" % step, 
                        "Training Loss: %.5f" % loss, 
                        "Training accuracy: %.5f" % train_accuracy,
                        "Val Loss: %.5f" % val_loss,
                        "Val accuracy: %.5f" % val_accuracy)
                    writer.add_summary(train_summary, global_step=epoch)

                    if val_accuracy > 0.925:
                        print('model save!')
                        timestr = str(datetime.datetime.now())
                        saver.save(sess=sess, save_path='./model/cnnlstm_' 
                                + timestr[:10] + 'T' + timestr[11:19] + '_optimal' +'.ckpt')


        print('model fit done!')
  
###############################################################################

        print('model save!')
        timestr = str(datetime.datetime.now())
        saver.save(sess=sess, save_path='./model/cnnlstm_' 
                                + timestr[:10] + 'T' + timestr[11:19] + '.ckpt')
        print('model save done!')

###############################################################################

        print('model evaluate!')
        x_test, y_test = get_feature_mode('test', test_list)
        x_test = x_test.reshape(x_test.shape[0], 51, 39, 1)
        x_test = np.transpose(x_test, (0, 2, 1, 3))
        test_feed_dict = {
            X: x_test,
            label: y_test,
            learning_rate: epoch_learning_rate,
            training_flag : False
        }
        # predict = model.predict(x_test, batch_size=batch_size)
        score = sess.run(accuracy, feed_dict=test_feed_dict)
        print('model evaluate done!')

###############################################################################

        print("Testing acc: ", score)


def generate_file(file_list, batch_size):
    shuffle(file_list)
    while True:
        for file in file_list:
            h5f = h5py.File(file, 'r')
            feature = h5f['feature'][:]
            feature = feature.reshape(feature.shape[0], 51, 39, 1)
            feature = np.transpose(feature, (0, 2, 1, 3))
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

