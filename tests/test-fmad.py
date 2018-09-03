import numpy as np
import numpy.random as npr

import tensorflow as tf
import tensorflow.contrib.slim as slim

import time
import fmad

def assign_value(x, value, i, j):
    shape = x[i].shape
    x[i] = np.reshape(x[i],[np.prod(shape)])
    x[i][j] = value
    x[i] = np.reshape(x[i], shape)
    return x

DEVICE       = '/gpu:0'
batch_size   = 15
DROPOUT      = False
COMPARE_TIME = True

tf.set_random_seed(0)
npr.seed(0)

with tf.device(DEVICE):
    x = tf.placeholder(tf.float32, [batch_size, 10, 10, 3])
    x_val = npr.randn(batch_size,10,10,3)

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=None):
        y = slim.conv2d(x, 10, [3, 3], scope='conv1')
        y = slim.batch_norm(y,scale=True)
        y = tf.nn.relu(y)
        y = slim.max_pool2d(y, [2, 2], scope='pool1')
        y = slim.conv2d(y, 15, [3, 3], scope='conv2')
        if DROPOUT:
            y = tf.layers.dropout(y, rate=.5, training=True)
        y = slim.batch_norm(y,scale=True)
        y = tf.reshape(y,[batch_size,15*5*5])
        y = slim.fully_connected(y, 1, scope='fully_c1')
        y = tf.square(y)
        y = tf.reduce_mean(y)

    w   = tf.trainable_variables()
    u   = [tf.placeholder(w[i].dtype, shape=w[i].get_shape()) for i in range(len(w))]
    jvp = fmad.fmad_prod(y, w, u)

    # Compute gradient w.r.t the parameters
    g = tf.gradients(y, w)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        grad_bw = sess.run(g, feed_dict={x:x_val})

        # Now compute forward gradients
        var_pl   = [x] + u
        u_val    = [np.zeros([3,3,3,10]), np.zeros([10]), np.zeros([10]), np.zeros([10]), np.zeros([3,3,10,15]), np.zeros([15]), np.zeros([15]), np.zeros([15]), np.zeros([15*5*5,1]), np.zeros([1])]
        grad_fwd = [np.zeros([3,3,3,10]), np.zeros([10]), np.zeros([10]), np.zeros([10]), np.zeros([3,3,10,15]), np.zeros([15]), np.zeros([15]), np.zeros([15]), np.zeros([15*5*5,1]), np.zeros([1])]

        for i in range(len(u_val)):
            for j in range(np.prod(u_val[i].shape)):
                u_val        = assign_value(u_val, 1, i, j)
                var_val      = [x_val] + u_val
                grad, grad_b = sess.run([jvp, g],  feed_dict={i:d for i,d in zip(var_pl,var_val)})
                u_val        = assign_value(u_val, 0, i, j)
                grad_fwd     = assign_value(grad_fwd, np.sum(grad), i, j)
                if DROPOUT:
                    grad_b       = np.reshape(grad_b[i], np.prod(grad_b[i].shape))
                    grad_bw  = assign_value(grad_bw, grad_b[j], i, j)

        for i in range(len(u_val)):
            print(np.sum(np.abs(grad_fwd[i]-grad_bw[i])))

        if COMPARE_TIME:
            bwt, fwt = [], []
            for i in range(100):
                start   = time.time()
                sess.run([jvp],  feed_dict={i:d for i,d in zip(var_pl,var_val)})
                t1      = time.time()
                fwt.append(t1-start)
                sess.run(g, feed_dict={x:x_val})
                t2      = time.time()
                bwt.append(t2-t1)
            print("Backward Time: %f"%(sum(bwt)))
            print("Forward Auto-diff Time: %f"%(sum(fwt)))

