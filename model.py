import numpy as np
import tensorflow as tf
from utils import *
import os
from data_process import *
# import tensorflow.contrib.slim as slim
from tqdm import tqdm
import time

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

class UPSNet:
  def __init__(self, content_loss='MSE', use_gan=False, learning_rate=1e-4, num_blocks=12, num_upsamples=2):
      self.learning_rate = learning_rate
      self.num_blocks = num_blocks
      self.num_upsamples = num_upsamples
      self.use_gan = use_gan
      self.reuse_vgg = False
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
  def forward(self, x, is_train, reuse,global_connection=True):
    """Builds the forward pass network graph"""
  # with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
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
  
  def loss_function(self, ps, pan, x):
    """Loss function"""
    # x is aligned ms, 
    ps_gray = tf.image.rgb_to_grayscale(ps)
    # detail loss
    content_loss = tf.reduce_sum(tf.abs(grad_ex(pan)-grad_ex(ps_gray))) #mean? sum?
    # dual-gradient loss
    r_loss = tf.minimum(tf.abs(grad_ex(pan)-grad_ex(ps[0])),tf.abs(-grad_ex(pan)-grad_ex(ps[0])))
    g_loss = tf.minimum(tf.abs(grad_ex(pan)-grad_ex(ps[1])),tf.abs(-grad_ex(pan)-grad_ex(ps[1])))
    b_loss = tf.minimum(tf.abs(grad_ex(pan)-grad_ex(ps[2])),tf.abs(-grad_ex(pan)-grad_ex(ps[2])))
    content_loss += tf.reduce_sum(r_loss+g_loss+b_loss)
    # color loss
    content_loss += tf.reduce_sum

    return content_loss
  
  def optimize(self, loss):
    #tf.control_dependencies([discrim_train
    # update_ops needs to be here for batch normalization to work
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
    with tf.control_dependencies(update_ops):
      return tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

  def psnr(self, x, y):
    return compute_psnr_tf(x, y)

  def make_summary(self, scope, x, y, pred, loss, psnr):
    with tf.name_scope(scope):
      ts1= tf.summary.image('input', x)
      ts2 = tf.summary.image('target', y)
      ts3 = tf.summary.image('outputs', tf.clip_by_value(pred,0,1))
      ts_loss = tf.summary.scalar('loss', loss)
      ts_psnr = tf.summary.scalar('psnr', psnr)

    img_merged = tf.summary.merge([ts1, ts2, ts3])
    scalar_merged = tf.summary.merge([ts_loss,ts_psnr])
    return img_merged, scalar_merged


  def training(self, sess, FLAGS, train_x, train_y, valid_x, valid_y):
    train_pred = self.forward(train_x, True, reuse=False)
    train_loss = self.loss_function(train_y, train_pred)
    train_psnr = self.psnr(train_pred, train_y)
    train_op = self.optimize(train_loss)
    train_loss_avg = tf.placeholder(tf.float32)
    train_psnr_avg = tf.placeholder(tf.float32)
    train_img, train_scalar = self.make_summary("train_summary", train_x, train_y, train_pred, train_loss_avg, train_psnr_avg)

    valid_pred = self.forward(valid_x, False, reuse=True)
    valid_loss = self.loss_function(valid_y, valid_pred) 
    valid_psnr = self.psnr(valid_pred, valid_y)
    valid_loss_avg = tf.placeholder(tf.float32)
    valid_psnr_avg = tf.placeholder(tf.float32)
    valid_img, valid_scalar = self.make_summary("valid_summary", valid_x, valid_y, valid_pred, valid_loss_avg, valid_psnr_avg)

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    saver = tf.train.Saver(var_list)

    writer = tf.summary.FileWriter('./board/graph/'+FLAGS.model_name, sess.graph)
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())
    loaded, start_epoch = self.load(sess, saver, os.path.join(FLAGS.checkpoint_dir,FLAGS.model_name))
    if loaded:
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    for epoch in tqdm(range(start_epoch,FLAGS.epochs)):
      start_time = time.time()
      sess.run(train_it.initializer)
      t_loss, t_psnr, v_loss, v_psnr = 0.0, 0.0, 0.0, 0.0
      count = 0
      try:
          while True:
              _, loss, psnr, train_img_summary = sess.run([train_op, train_loss, train_psnr, train_img])
              t_loss += loss
              t_psnr += psnr
              count += 1
      except tf.errors.OutOfRangeError:
          pass
      t_loss /= count
      t_psnr /= count

      sess.run(valid_it.initializer)
      count = 0
      while True:
          try:
              loss, psnr, valid_img_summary = sess.run([valid_loss, valid_psnr,valid_img])
              v_loss += loss
              v_psnr += psnr
              count += 1 
          except tf.errors.OutOfRangeError:
              break
      v_loss /= count
      v_psnr /= count
      print("Epoch: [%2d], time: [%4.4f], train_loss: [%.8f], train_psnr: [%.4f], valid_loss: [%.8f], valid_psnr: [%.4f]"% ((epoch+1), time.time()-start_time, t_loss, t_psnr, v_loss, v_psnr))
      
      self.save(sess, saver, os.path.join(FLAGS.checkpoint_dir,FLAGS.model_name), epoch)
      
      writer.add_summary(train_img_summary,epoch)
      writer.add_summary(valid_img_summary,epoch)
      
      train_summary = sess.run(train_scalar, feed_dict={train_loss_avg: t_loss, train_psnr_avg: t_psnr})
      valid_summary = sess.run(valid_scalar, feed_dict={valid_loss_avg: v_loss, valid_psnr_avg: v_psnr})
      writer.add_summary(train_summary,epoch)
      writer.add_summary(valid_summary,epoch)

  def save(self, sess, saver, checkpoint_dir, step):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=step)

  def load(self, sess, saver, checkpoint_dir):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("Restored %s "%ckpt_name)
        
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        return True, int(ckpt_name.split('-')[-1])+1
    else:
        return False, 0


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
    with tf.variable_scope('generator', reuse=reuse):
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
      for i in range(self.num_upsamples//2):
        with tf.name_scope("Upsample_"+str(i)):
          x = self._Upsample2xBlock(x, kernel_size=3, filters=256)
      
      x = tf.layers.conv2d(x, kernel_size=9, filters=3, strides=1, padding='same', name='forward')

    return x
  
  def psnr(self, x, y):
    return compute_psnr_tf(x, y)

  def ssim(self, x,y):
    return compute_ssim_tf(x,y)

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
        content_loss = tf.reduce_mean(tf.square(diff))
    else:
        content_loss = 0.0061*tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))
    return content_loss
  
  def optimize(self, loss):
    #tf.control_dependencies([discrim_train
    # update_ops needs to be here for batch normalization to work
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
    with tf.control_dependencies(update_ops):
      return tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

  def make_summary(self, scope, x, y, pred, loss, psnr,ssim):
    with tf.name_scope(scope):
      ts1= tf.summary.image('input', x)
      ts2 = tf.summary.image('target', y)
      ts3 = tf.summary.image('outputs', tf.clip_by_value(pred,0,1))
      ts_loss = tf.summary.scalar('loss', loss)
      ts_psnr = tf.summary.scalar('psnr', psnr)
      ts_ssim = tf.summary.scalar('ssim', ssim)

    img_merged = tf.summary.merge([ts1, ts2, ts3])
    scalar_merged = tf.summary.merge([ts_loss,ts_psnr, ts_ssim])
    return img_merged, scalar_merged

  def make_it(self, is_train, train_img_dir, train_label_dir, valid_img_dir, valid_label_dir):
    if is_train:
      train_dataset = make_dataset_ps(train_img_dir, train_label_dir, train=True)
      valid_dataset = make_dataset_ps(valid_img_dir, valid_label_dir, train=False) 
      return train_dataset.make_initializable_iterator(), valid_dataset.make_initializable_iterator()
    else:
      valid_dataset =  make_dataset_ps(valid_img_dir, valid_label_dir, train=False) 
      return valid_dataset.make_initializable_iterator()

  def training(self, sess, FLAGS):
    if FLAGS.is_train:
      train_it, valid_it = self.make_it(True,FLAGS.train_img_dir, FLAGS.train_img_dir, \
                                        FLAGS.valid_img_dir, FLAGS.valid_img_dir)
      train_x, train_y = train_it.get_next()

      train_pred = self.forward(train_x, True, reuse=False)
      train_loss = self.loss_function(train_y, train_pred)
      train_psnr = self.psnr(train_pred, train_y)
      train_ssim = self.ssim(train_pred, train_y)
      train_op = self.optimize(train_loss)

      train_loss_avg = tf.placeholder(tf.float32)
      train_psnr_avg = tf.placeholder(tf.float32)
      train_ssim_avg = tf.placeholder(tf.float32)
      train_img, train_scalar = self.make_summary("train_summary", train_x, train_y, train_pred, train_loss_avg, train_psnr_avg, train_ssim_avg)

    else:
      valid_it = self.make_it(False, None, None, FLAGS.test_img_dir, FLAGS.test_img_dir)
    valid_x, valid_y = valid_it.get_next()
    
    valid_pred = self.forward(valid_x, False, reuse=True)
    valid_loss = self.loss_function(valid_y, valid_pred) 
    valid_psnr = self.psnr(valid_pred, valid_y)
    valid_ssim = self.ssim(valid_pred, valid_y)
    if FLAGS.is_train:
      valid_loss_avg = tf.placeholder(tf.float32)
      valid_psnr_avg = tf.placeholder(tf.float32)
      valid_ssim_avg = tf.placeholder(tf.float32)
      valid_img, valid_scalar = self.make_summary("valid_summary", valid_x, valid_y, valid_pred, valid_loss_avg, valid_psnr_avg, valid_ssim_avg)
      writer = tf.summary.FileWriter('./board/graph/'+FLAGS.model_name, sess.graph)
      writer.add_graph(sess.graph)

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    saver = tf.train.Saver(var_list)

    sess.run(tf.global_variables_initializer())
    loaded, start_epoch = self.load(sess, saver, os.path.join(FLAGS.checkpoint_dir,FLAGS.model_name))
    if loaded:
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

    for epoch in tqdm(range(start_epoch,FLAGS.epochs)):
      start_time = time.time()
      t_loss, t_psnr, t_ssim, v_loss, v_psnr, v_ssim = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
      if FLAGS.is_train:
        sess.run(train_it.initializer)      
        count = 0
        try:
            while True:
                _, loss, psnr, ssim, train_img_summary = sess.run([train_op, train_loss, train_psnr, train_ssim, train_img])
                t_loss += loss
                t_psnr += np.mean(psnr)
                t_ssim += np.mean(ssim)
                count += 1
        except tf.errors.OutOfRangeError:
            pass
        t_loss /= count
        t_psnr /= count
        t_ssim /= count
      else:
        f = open(FLAGS.sample_dir+"/result.txt","w")
      sess.run(valid_it.initializer)
      count = 0
      while True:
          try:
              if FLAGS.is_train:
                loss, psnr, ssim, valid_img_summary = sess.run([valid_loss, valid_psnr, valid_ssim, valid_img])
              else:
                loss, psnr, ssim, x, y, pred = sess.run([valid_loss, valid_psnr, valid_ssim, valid_x, valid_y, valid_pred])
              v_loss += loss
              v_psnr += np.mean(psnr)
              v_ssim += np.mean(ssim)
              count += 1 
              if not FLAGS.is_train:
                save_image(FLAGS.sample_dir+"/LR_"+str(count)+".jpg", x[0])
                save_image(FLAGS.sample_dir+"/HR_"+str(count)+".jpg", y[0])
                save_image(FLAGS.sample_dir+"/pred_"+str(count)+".jpg", pred[0])
                f.write("%dth img => time: [%4.4f], loss: [%.8f], psnr: [%.4f], bicubic_psnr: [%.4f], ssim: [%.4f], bicubic_ssim: [%.4f]\n"% ((count), time.time()-start_time, loss, psnr, bic_psnr, ssim, bic_ssim))
          except tf.errors.OutOfRangeError:
              break
      v_loss /= count
      v_psnr /= count
      v_ssim /= count
      if FLAGS.is_train:
        print("Epoch: [%2d], time: [%4.4f], train_loss: [%.8f], train_psnr: [%.4f],train_ssim: [%.4f], valid_loss: [%.8f], valid_psnr: [%.4f], valid_ssim: [%.4f]"% ((epoch+1), time.time()-start_time, t_loss, t_psnr, t_ssim, v_loss, v_psnr, v_ssim))
      else:
        f.write("Avg. Loss : %.8f, Avg. PSNR : %.4f, Avg. SSIM : %.4f"%(v_loss/count, v_psnr/count, v_ssim/count))
        f.close()
        break    
      self.save(sess, saver, os.path.join(FLAGS.checkpoint_dir,FLAGS.model_name), epoch)
      
      writer.add_summary(train_img_summary,epoch)
      writer.add_summary(valid_img_summary,epoch)
      
      train_summary = sess.run(train_scalar, feed_dict={train_loss_avg: t_loss, train_psnr_avg: t_psnr, train_ssim_avg: t_ssim})
      valid_summary = sess.run(valid_scalar, feed_dict={valid_loss_avg: v_loss, valid_psnr_avg: v_psnr, valid_ssim_avg: v_ssim})
      writer.add_summary(train_summary,epoch)
      writer.add_summary(valid_summary,epoch)

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
