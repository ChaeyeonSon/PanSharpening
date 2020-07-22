import numpy as np
import tensorflow as tf
from utils import *
import os
from data_process import *
import tensorflow.contrib.slim as slim

# Refer to https://github.com/trevor-m/tensorflow-SRGAN/blob/master/srgan.py
# Refer to https://github.com/tegg89/SRCNN-Tensorflow/blob/master/model.py 

# VGG19 net
def vgg_19(inputs,
           num_classes=1000,
           is_training=False,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           reuse = False,
           fc_conv_padding='VALID'):
  """Oxford Net VGG 19-Layers version E Example.
  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output. Otherwise,
      the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.
  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, 3, scope='conv1', reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, 3, scope='conv2',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 4, slim.conv2d, 256, 3, scope='conv3', reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv4',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 4, slim.conv2d, 512, 3, scope='conv5',reuse=reuse)
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)

      return net, end_points
vgg_19.default_image_size = 128

def VGG19_slim(input, type, reuse, scope):
    # Define the feature to extract according to the type of perceptual
    if type == 'VGG54':
       # target_layer = scope + 'vgg_19/conv5/conv5_4'
        target_layer = 'vgg_19/conv5/conv5_4'
    elif type == 'VGG22':
        target_layer = scope + 'vgg_19/conv2/conv2_2'
    else:
        raise NotImplementedError('Unknown perceptual type')
    _, output = vgg_19(input, is_training=False, reuse=reuse)
    print(output)
    output = output[target_layer]

    return output

class SRGenerator:
  def __init__(self, content_loss='MSE', use_gan=False, learning_rate=1e-4, num_blocks=12, num_upsamples=2):
      self.learning_rate = learning_rate
      self.num_blocks = num_blocks
      self.num_upsamples = num_upsamples
      self.use_gan = use_gan
      self.reuse_vgg = False
      if content_loss not in ['MSE', 'VGG22', 'VGG54']:
          print('Invalid content loss function. Must be \'mse\', \'vgg22\', or \'vgg54\'.')
          exit()
      self.content_loss = content_loss
      #self.device = "/device:"+device+":"+str(device_num)
  
  def _residual_block(self, x, kernel_size, filters, strides=1, training=False):
    #x = tf.nn.relu(x)
    skip = x
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same', use_bias=False)
    x = tf.layers.batch_normalization(x, training=training)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same', use_bias=False)
    x = tf.layers.batch_normalization(x, training=training)
    x = x + skip
    return x

  def _Upsample2xBlock(self, x, kernel_size, filters, strides=1):
    """Upsample 2x via SubpixelConv"""
    x = tf.layers.conv2d(x, kernel_size=kernel_size, filters=filters, strides=strides, padding='same', use_bias=False)
    x = tf.depth_to_space(x, 2)
    x = tf.nn.relu(x)
    return x

  def forward(self, x, is_train, reuse,global_connection=True):
    """Builds the forward pass network graph"""
#    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
    with tf.variable_scope('generator', reuse=reuse) as scope:
      x = tf.layers.conv2d(x, kernel_size=9, filters=64, strides=1, padding='same')
      x = tf.nn.relu(x)
      #x = tf.contrib.keras.layers.PReLU(shared_axes=[1,2])(x)
      skip = x

      # B x ResidualBlocks
      for i in range(self.num_blocks):
        with tf.name_scope("ResBlock_"+str(i)):
          x = self._residual_block(x, kernel_size=3, filters=64, strides=1, training=is_train)

      x = tf.nn.relu(x)
      x = tf.layers.conv2d(x, kernel_size=3, filters=64, strides=1, padding='same', use_bias=False)
      x = tf.layers.batch_normalization(x, training=is_train)
      if global_connection:
        x = x + skip

      # Upsample blocks
      for i in range(self.num_upsamples-1):
        with tf.name_scope("Upsample_"+str(i)):
          x = self._Upsample2xBlock(x, kernel_size=3, filters=256)
      
      x = tf.layers.conv2d(x, kernel_size=9, filters=3, strides=1, padding='same', name='forward')

    return x
  
  def loss_function(self, y, y_pred):
    """Loss function"""
    # if self.use_gan:
    #   # Weighted sum of content loss and adversarial loss
    #   return self._content_loss(y, y_pred) + 1e-3*self._adversarial_loss(y_pred)
    # Content loss only
    # return self._content_loss(y, y_pred)
    if self.content_loss == 'VGG54':
        with tf.name_scope('vgg_19_1') as scope:
            extracted_feature_gen = VGG19_slim(y_pred, 'VGG54', reuse=tf.AUTO_REUSE, scope=scope)
        with tf.name_scope('vgg_19_2') as scope:
            extracted_feature_target = VGG19_slim(y, 'VGG54', reuse=True, scope=scope)

    # Use the VGG22 feature
    elif self.content_loss == 'VGG22':
        with tf.name_scope('vgg_19_1') as scope:
            extracted_feature_gen = VGG19_slim(y_pred, 'VGG22', reuse=False, scope=scope)
        with tf.name_scope('vgg_19_2') as scope:
            extracted_feature_target = VGG19_slim(y, 'VGG22', reuse=True, scope=scope)

    # Use MSE loss directly
    elif self.content_loss == 'MSE':
        extracted_feature_gen = y_pred
        extracted_feature_target = y

    diff = extracted_feature_gen - extracted_feature_target
    if self.content_loss == 'MSE':
        content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))
    else:
        content_loss = 0.0061*tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))
    return content_loss
  
  def optimize(self, loss):
    #tf.control_dependencies([discrim_train
    # update_ops needs to be here for batch normalization to work
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
    with tf.control_dependencies(update_ops):
      return tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

  def save(self, sess, saver, checkpoint_dir, step):
    model_name = "SRResNet"
    #model_dir = "%s_%s" % ("srresnet", "valid20")
    #checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

  def load(self, sess, saver, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    #model_dir = "%s_%s" % ("srresnet", "valid20")
    #checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("Restored %s "%ckpt_name)
        
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        return True, int(ckpt_name.split('-')[-1])+1
    else:
        return False, 0
