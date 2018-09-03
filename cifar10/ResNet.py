import re
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cifar10_input

FLAGS = tf.app.flags.FLAGS
NUM_CLASSES = cifar10_input.NUM_CLASSES

def ConvBlock(input_tensor, channels, ker, stride_c, num_conv, activation=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=None,
                        biases_initializer=None,
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        trainable=True):
      x = slim.conv2d(input_tensor, channels, ker, stride=stride_c, scope='conv%s'%(num_conv))
      x = slim.batch_norm(x,scale=True)
      if activation:
        x = tf.nn.relu(x)
      return x

def ResNet(images, device):
  """Build the CIFAR-10 ResNet-18 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
    device: Device where we will put the variables.
  Returns:
    Logits.
  """
  blocksPerSection = [2, 2, 2, 2]
  channelsPerSection = [64, 128, 256, 512]
  channelsPerBlock = [1, 1]
  downsampleSection = [0, 1, 1, 1]


  x = images
  channelsOut = 64

  with tf.device(device):

    x = ConvBlock(x, 64, [7,7], 2, '_init')
    x = slim.max_pool2d(x, [3, 3], stride=2, scope='pool_1')

    for s in range(len(blocksPerSection)):
      for l in range(blocksPerSection[s]):

        # Stride at the beginning of each block
        stride = 1
        if l == 0 and downsampleSection[s]:
          stride = 2

        sumInput = x

        # 2 conv only
        x = ConvBlock(x, channelsPerSection[s]*channelsPerBlock[1], [3, 3], stride, '%d_1_%d'%(s,l))
        x = ConvBlock(x, channelsPerSection[s]*channelsPerBlock[1], [3, 3], 1, '%d_2_%d'%(s,l), False)

        if l == 0 and channelsOut != channelsPerSection[s]*channelsPerBlock[1]:
          sumInput = ConvBlock(sumInput, channelsPerSection[s]*channelsPerBlock[1], [1,1], stride, '_sum%d'%(s), False)

        channelsOut = channelsPerSection[s]*channelsPerBlock[1]
        x = sumInput + x
        x = tf.nn.relu(x)

    with slim.arg_scope([slim.fully_connected],
                        activation_fn=None,
                        weights_regularizer=slim.l2_regularizer(0.0005),
                        biases_regularizer=slim.l2_regularizer(0.0005),
                        trainable=True):
      x              = tf.reduce_mean(x, [1,2])
      softmax_linear = slim.fully_connected(x, NUM_CLASSES, scope='fc_1')

  return softmax_linear

