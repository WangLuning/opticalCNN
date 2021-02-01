import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import functools

from datetime import datetime
import argparse
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import tensorflow as tf

import layers.optics as optics
import layers.optics_alt as optics_alt
from layers.utils import *


FUNCTION_MAP = {'relu' : tf.nn.relu,
                'identity' : tf.identity}

# change to your directory
train_data = np.load('assets/quickdraw16_train.npy')
test_data = np.load('assets/quickdraw16_test.npy')


# iteratively read data from the training / test dataset
def get_feed(train, batch_size=50, num_classes=16):
    if train:
        idcs = np.random.randint(0, np.shape(train_data)[0], batch_size)
        x = train_data[idcs, :]
        y = np.zeros((batch_size, num_classes))
        y[np.arange(batch_size), idcs//8000] = 1
            
    else:
        x = test_data
        y = np.zeros((np.shape(test_data)[0], num_classes))
        y[np.arange(np.shape(test_data)[0]), np.arange(np.shape(test_data)[0])//100] = 1                
        
    return x, y


def test_simulation(x, y_, keep_prob, accuracy, train_accuracy, num_iter = 10):
    x_test, y_test = get_feed(train=False)
    test_batches = []
    for i in range(num_iter):
        idx = i*50
        batch_acc = accuracy.eval(feed_dict={x: x_test[idx:idx+50, :], y_: y_test[idx:idx+50, :], keep_prob: 1.0})
        test_batches.append(batch_acc)
    test_acc = np.mean(test_batches)   

    #test_acc = accuracy.eval(feed_dict={x: x_test, y_: y_test, keep_prob: 1.0})
    print('final step %d, train accuracy %g, test accuracy %g' %
          (i, train_accuracy, test_acc))


# test a model with various constraints
def train(params, summary_every=100, print_every=250, save_every=1000, verbose=True):
    # Unpack params
    classes = params.num_classes

    # constraint helpers
    def nonneg(input_tensor):
        return tf.square(input_tensor) if params.isNonNeg else input_tensor

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

    # input placeholders
    with tf.name_scope('input'):
    	# the quickdraw image size is 28 * 28
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.int64, shape=[None, classes])
        keep_prob = tf.placeholder(tf.float32)

        x_image = tf.reshape(x, [-1, 28, 28, 1])
        # in the image dimension give four borders padding size 64
        paddings = tf.constant([[0, 0,], [params.padamt, params.padamt], [params.padamt, params.padamt], [0, 0]])
        x_image = tf.pad(x_image, paddings)
        # x_image = tf.image.resize_nearest_neighbor(x_image, size=(dim, dim))
        tf.summary.image('input', x_image, 3)
        
        # if not isNonNeg and not doNonnegReg:
        #     x_image -= tf.reduce_mean(x_image)

    # nonneg regularizer
    global_step = tf.Variable(0, trainable=False)
    if params.doNonnegReg:
    	# TODO: need to design start and end learning rate to get a valid decaying learning rate
        reg_scale = tf.train.polynomial_decay(0.,
                                              global_step,
                                              decay_steps=8000,
                                              end_learning_rate=10000.)
        psf_reg = optics_alt.nonneg_regularizer(reg_scale)
    else:
        psf_reg = None
    
    # build model 
    # single tiled convolutional layer
    h_conv1 = optics_alt.tiled_conv_layer(x_image, params.tiling_factor, params.tile_size, params.kernel_size, 
                                          name='h_conv1', nonneg=params.isNonNeg, regularizer=psf_reg)

    h_conv2 = optics_alt.tiled_conv_layer(h_conv1, 1, 4, 4, 
                                          name='h_conv2', nonneg=params.isNonNeg, regularizer=psf_reg)

    optics.attach_img("h_conv2", h_conv2)
    # each split is of size (None, 39, 156, 1)
    split_1d = tf.split(h_conv2, num_or_size_splits=4, axis=1)

    # calculating output scores (16, None, 39, 39, 1)
    h_conv_split = tf.concat([tf.split(split_1d[0], num_or_size_splits=4, axis=2),
                              tf.split(split_1d[1], num_or_size_splits=4, axis=2),
                              tf.split(split_1d[2], num_or_size_splits=4, axis=2),
                              tf.split(split_1d[3], num_or_size_splits=4, axis=2)], 0)
    if params.doMean:
        y_out = tf.transpose(tf.reduce_mean(h_conv_split, axis=[2,3,4]))
    else:
        y_out = tf.transpose(tf.reduce_max(h_conv_split, axis=[2,3,4]))

    tf.summary.image('output', tf.reshape(y_out, [-1, 4, 4, 1]), 3)

    # loss, train, acc
    with tf.name_scope('cross_entropy'):
        total_data_loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_out)
        data_loss = tf.reduce_mean(total_data_loss)
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_loss = tf.add(data_loss, reg_loss)
        tf.summary.scalar('data_loss', data_loss)
        tf.summary.scalar('reg_loss', reg_loss)
        tf.summary.scalar('total_loss', total_loss)

    if params.opt_type == 'ADAM':
        train_step = tf.train.AdamOptimizer(params.learning_rate).minimize(total_loss, global_step)
    elif params.opt_type == 'Adadelta':
        train_step = tf.train.AdadeltaOptimizer(params.learning_rate_ad, rho=.9).minimize(total_loss, global_step)
    else:
        train_step = tf.train.MomentumOptimizer(params.learning_rate, momentum=0.5, use_nesterov=True).minimize(total_loss, global_step)
    
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    losses = []

    # tensorboard setup
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(params.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(params.log_dir + '/test')

    tf.global_variables_initializer().run()
    
    # add ops to save and restore all the variables
    saver = tf.train.Saver(max_to_keep=2)
    save_path = os.path.join(params.log_dir, 'model.ckpt')
    
    for i in range(params.num_iters):
        x_train, y_train = get_feed(train=True, num_classes=classes)
        _, loss, reg_loss_graph, train_accuracy, train_summary = sess.run(
                          [train_step, total_loss, reg_loss, accuracy, merged], 
                          feed_dict={x: x_train, y_: y_train, keep_prob: params.dropout})
        losses.append(loss)

        if i % summary_every == 0:
            train_writer.add_summary(train_summary, i)
            
        if i > 0 and i % save_every == 0:
            # print("Saving model...")
            saver.save(sess, save_path, global_step=i)
            
            # validation set
            x_valid, y_valid = get_feed(train=False, batch_size=10)
            test_summary, test_accuracy = sess.run([merged, accuracy],
                                                   feed_dict={x: x_valid, y_: y_valid, keep_prob: 1.0})
            test_writer.add_summary(test_summary, i)
            if verbose:
                print('step %d: validation acc %g' % (i, test_accuracy))
            
        if i % print_every == 0:
            if verbose:
                print('step %d:\t loss %g,\t reg_loss %g,\t train acc %g' %
                      (i, loss, reg_loss_graph, train_accuracy))

    test_simulation(x, y_, keep_prob, accuracy, train_accuracy, num_iter = 10)
    #sess.close()

    train_writer.close()
    test_writer.close()

def main(_):
    if tf.gfile.Exists(params.log_dir):
        tf.gfile.DeleteRecursively(params.log_dir)
    tf.gfile.MakeDirs(params.log_dir)

    # try different constraints
    params.wavelength = 532e-9
    params.isNonNeg = True
    # make sure this node is added into the graph : tf.identity operator
    params.activation = 'identity' # functools.partial(shifted_relu, thresh=10.0)
    params.opt_type = 'ADAM'
    
    params.doMultichannelConv = False
    params.doMean = False
    params.doOpticalConv = False
    
    params.doNonnegReg = False
    
    params.padamt = 64
    params.dim = 40*4
    
    # there are 4 * 4 kernels laterally
    params.tiling_factor = 4
    # leave some protective bands?
    params.tile_size = 40
    params.kernel_size = 32

    train(params, summary_every=10, print_every=10, save_every=10, verbose=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_iters', type=int, default=101,
                      help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                      help='Initial learning rate')
    parser.add_argument('--learning_rate_ad', type=float, default=1,
                      help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')

    # contraints arguments
    parser.add_argument('--wavelength',type=float,  default=532e-9)
    parser.add_argument('--isNonNeg',type=bool, default=False)
    parser.add_argument('--numIters', type=int, default=1000)
    parser.add_argument('--activation', default='relu', choices=FUNCTION_MAP.keys())
    parser.add_argument('--opt_type', type=str, default='ADAM')
    
    # switches
    parser.add_argument('--doMultichannelConv', type=bool, default=False)
    parser.add_argument('--doMean', type=bool, default=False)
    parser.add_argument('--doOpticalConv', type=bool, default=True)
    parser.add_argument('--doAmplitudeMask', type=bool, default=False)
    parser.add_argument('--doZernike', type=bool, default=False)
    parser.add_argument('--doNonnegReg', type=bool, default=False)

    parser.add_argument('--z_modes', type=int, default=1024)
    parser.add_argument('--convdim1', type=int, default=100)
    
    parser.add_argument('--cdim1', type=int, default=16)
    
    parser.add_argument('--padamt', type=int, default=0)
    parser.add_argument('--dim', type=int, default=60) 
    
    parser.add_argument('--tiling_factor', type=int, default=5)
    parser.add_argument('--tile_size', type=int, default=56)
    parser.add_argument('--kernel_size', type=int, default=7)
    parser.add_argument('--num_classes', type=int, default=16)

    # start processing
    now = datetime.now()
    runtime = now.strftime('%Y%m%d-%H%M%S')
    run_id = 'quickdraw_tiled_nonneg/' + runtime + '/'
    parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join('checkpoints/', run_id),
      help='Summaries log directory')
    params, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
