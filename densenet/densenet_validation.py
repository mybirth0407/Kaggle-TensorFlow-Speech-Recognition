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


from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope


import tensorflow as tf
import datetime
import time

# Hyperparameter
growth_k = 12
nb_block = 2 # how many (dense block + Transition Layer) ?
init_learning_rate = 1e-4
epsilon = 1e-10 # AdamOptimizer epsilon
dropout_rate = 0.2

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4

# Label & batch_size
class_num = 12
#batch_size = 100

total_epochs = 3 #12

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
            x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_'+str(i))
            x = self.transition_layer(x, scope='trans_'+str(i))

        """
        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')
        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')
        x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')
        """

        x = self.dense_block(input_x=x, nb_layers=40, layer_name='dense_final')

        # 100 Layer
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)


        # x = tf.reshape(x, [-1, 10])
        return x



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

#    x_val, y_val = get_feature_mode('val', val_list)
#    x_val = x_val.reshape(x_val.shape[0], 51, 39, 1)
    x_test, y_test = get_feature_mode('test', test_list)
    x_test = x_test.reshape(x_test.shape[0], 51, 39, 1)
#    print("x_val.shape: ", x_val.shape)
#    print("y_val.shape: ", y_val.shape)

    print('data loading done!')

###############################################################################

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

###############################################################################

    with tf.Session() as sess:
        if not isdir('./model'):
            mkdir('./model')

        ckpt = tf.train.get_checkpoint_state('./loadval')
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
            total_steps = 10
            train_generator = generate_file(val_list, 256)
            for step in range(total_steps):
                batch_x, batch_y = next(train_generator)

                train_feed_dict = {
                    X: batch_x,
                    label: batch_y,
                    learning_rate: epoch_learning_rate,
                    training_flag : True
                }

                _, loss = sess.run([train, cost], feed_dict=train_feed_dict)

                if step % 1 == 0:
                    val_feed_dict = {
                        X: x_test, #x_val,
                        label: y_test, #y_val,
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

                    if val_accuracy > 0.94:
                        print('model save!')
                        timestr = str(datetime.datetime.now())
                        saver.save(sess=sess, save_path='./loadval/densenet_model_' 
                                + timestr[:10] + 'T' + timestr[11:19] + '_optimal' +'.ckpt')


        print('model fit done!')
  
###############################################################################

        print('model save!')
        timestr = str(datetime.datetime.now())
        saver.save(sess=sess, save_path='./loadval/densenet_model_' 
                                + timestr[:10] + 'T' + timestr[11:19] + '.ckpt')
        print('model save done!')

###############################################################################

        print('model evaluate!')
        x_test, y_test = get_feature_mode('test', test_list)
        x_test = x_test.reshape(x_test.shape[0], 51, 39, 1)
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

