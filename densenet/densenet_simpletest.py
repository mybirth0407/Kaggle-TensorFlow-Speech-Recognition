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

import time

import tensorflow as tf

from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

# Hyperparameter
growth_k = 12
nb_block = 2 # how many (dense block + Transition Layer) ?
init_learning_rate = 1e-3
epsilon = 1e-10 # AdamOptimizer epsilon
dropout_rate = 0.2

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4

# Label & batch_size
class_num = 12



# labels
# all_label = listdir('./feature/train/train')
# all_label.sort()
# all_label_dict = {}
# for (i, label) in zip(range(len(all_label)), all_label):
#   all_label_dict[label] = i
#
# int_dict = dict(zip(
#     all_label_dict.values(), all_label_dict.keys()
# ))

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


# Densenet Network Design

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, 
                                filters=filter, 
                                kernel_size=kernel, 
                                strides=stride, 
                                padding='SAME')
        return network

def Global_Average_Pooling(x, stride=1):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride)


#    return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not


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



class DenseNet():
    def __init__(self, x, nb_blocks, filters, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.model = self.Dense_net(x)


    def bottleneck_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            # print(x)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')
        x = Max_Pooling(x, pool_size=[3,3], stride=2)


        for i in range(self.nb_blocks) :
            # 6 -> 12 -> 48
            x = self.dense_block(input_x=x, nb_layers=4, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))

        """
        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')
        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')
        x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')
        """

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

        # 100 Layer
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)


        # x = tf.reshape(x, [-1, 10])
        return x




def get_feature_file_list(file_list):
    n_processes = multiprocessing.cpu_count()
    pool = Pool(processes=n_processes)
    result = pool.map(get_feature_file, file_list)
    pool.close()
    pool.join()

    return result

def get_feature_file(file):
#    print(file + ' start!')
    h5f = h5py.File(file, 'r')
    feature = h5f['feature'][:]
    h5f.close()
#    print(file + ' done!')

    return feature

def most_common(l):
    return max(set(l), key=l.count)



def main(argv):
    feature_path = '../feature/simpletest/'

    file_list = []

    for file in listdir(feature_path):
        file_list.append(os.path.join(feature_path, file))
    file_list.sort()

    n = 100
    file_list = [file_list[i::n] for i in range(n)]

    if not isdir('./pred'):
        mkdir('./pred')

    # f = open('./pred/pred_' + str(time.time()) +  '.csv', 'w')
    pred_file, _ = os.path.splitext(sys.argv[1])
    pred_file = os.path.join('./pred/dense_0111_1030')# pred_file)
    f = open(pred_file + '.csv', 'w')
    f.write('fname,label\n')

    X = tf.placeholder(tf.float32, shape=[None, 51, 39, 1])
    label = tf.placeholder(tf.float32, shape=[None, 12])

    training_flag = tf.placeholder(tf.bool)
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    logits = DenseNet(x=X, nb_blocks=nb_block, filters=growth_k, training=training_flag).model
    prediction = tf.nn.softmax(logits)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

    """
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=nesterov_momentum, use_nesterov=True)
    train = optimizer.minimize(cost + l2_loss * weight_decay)
    In paper, use MomentumOptimizer
    init_learning_rate = 0.1
    """

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
    train = optimizer.minimize(cost)

    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('loss', cost)
    tf.summary.scalar('accuracy', accuracy)

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(sys.argv[1])
        saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(len(file_list)):
            feature_list = get_feature_file_list(file_list[i])
            cnt = 1
            will = len(feature_list)
            labels = []

            #########
            for j in range(will):
                feature = feature_list[j]
                # print(feature.shape)
                feature = feature.reshape(1, 51, 39, 1)
                test_feed_dict = {
                    X: feature,
                    training_flag : False
                }

                predict = sess.run(prediction, feed_dict=test_feed_dict)

                print(str(cnt) + '/' + str(will))
                # axis = 0, column appending
#                predict = model.predict(feature, batch_size=256)
#                print(predict)
                label_index = list(np.argmax(predict, axis=1))
#                print(label_index)
                y = max(set(label_index), key=label_index.count)
#                print(y)
                y = int_dict.get(y)
#                print(y)
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



if __name__ == '__main__':
    main(sys.argv)
