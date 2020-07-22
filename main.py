import tensorflow as tf 
import numpy as np
import argparse
from data_process import *
from model import *

import os
import glob
import pprint
from tqdm import tqdm
import time
import cv2

import utils

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS=False

flags = tf.app.flags

flags.DEFINE_integer("batch_size", 8, "The size of batch images [128]")
#flags.DEFINE_integer("image_size", 33, "The size of image to use [33]")
#flags.DEFINE_integer("label_size", 21, "The size of label to produce [21]")
flags.DEFINE_integer("epochs", 3200, "The number of epochs of learning")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of gradient descent algorithm [1e-4]")
#flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [1]")
#flags.DEFINE_integer("scale", 2, "The size of scale factor for preprocessing input image [3]")
#flags.DEFINE_integer("stride", 14, "The size of stride to apply input image [14]")
# The content loss parameter
flags.DEFINE_string('perceptual_mode', 'MSE', 'The type of feature used in perceptual loss')
#flags.DEFINE_string('perceptual_mode', 'VGG54', 'The type of feature used in perceptual loss')
flags.DEFINE_float('EPS', 1e-12, 'The eps added to prevent nan')
flags.DEFINE_float('ratio', 0.001, 'The ratio between content loss and adversarial loss')
flags.DEFINE_float('vgg_scaling', 0.0061, 'The scaling factor for the perceptual loss if using vgg perceptual loss')
flags.DEFINE_string('vgg_ckpt', './vgg19/vgg_19.ckpt', 'path to checkpoint file for the vgg19')

flags.DEFINE_string("checkpoint_dir", "./checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
#flags.DEFINE_string("train_img_dir", "../train_data/bicubic_upscaled/*.png", "Name of train img directory [train_img]")
flags.DEFINE_string("train_img_dir", "../train_data/DIV2K_train_LR_bicubic/*.png", "Name of train img directory [train_img]")
flags.DEFINE_string("train_label_dir", "../train_data/DIV2K_train_HR/*.png", "Name of train label directory [train_label]")
#flags.DEFINE_string("valid_img_dir", "../valid_data/bicubic_upscaled/*.png", "Name of valid img directory [valid_img]")
flags.DEFINE_string("valid_img_dir", "../valid_data/DIV2K_valid_LR_bicubic/*.png", "Name of valid img directory [valid_img]")
flags.DEFINE_string("valid_label_dir", '../valid_data/DIV2K_valid_HR/*.png', "Name of svalid label directory [valid_label]")
#flags.DEFINE_string("test_img_dir", "../test_data/bicubic_upscaled/*.png", "Name of valid img directory [valid_img]")
flags.DEFINE_string("test_img_dir", "../test_data/DIV2K_valid_LR_bicubic/*.png", "Name of valid img directory [valid_img]")
flags.DEFINE_string("test_label_dir", '../test_data/DIV2K_valid_HR/*.png', "Name of svalid label directory [valid_label]")

flags.DEFINE_string("model_name", 'srresnet_12_blk_vgg54', "Name of Experiment")

flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
flags.DEFINE_string("device", "GPU", "Which device to use")
flags.DEFINE_integer("device_num", 0, "Which device number to use")

flags.DEFINE_boolean("placeholder", False, "True for using placeholder in data input")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()


def main():
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists('./board'):
        os.makedirs('./board')
    os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.device_num)
    tf.config.set_soft_device_placement(True)
    model = SRGenerator(content_loss=FLAGS.perceptual_mode) 
    
    if FLAGS.is_train:
        train_dataset = make_dataset(FLAGS.train_img_dir, FLAGS.train_label_dir, train=True,batch_size=FLAGS.batch_size)
        train_it = train_dataset.make_initializable_iterator()
        train_x, train_y = train_it.get_next()
        if FLAGS.placeholder:
            valid_data= np.array([np.array(cv2.cvtColor(cv2.imread(fname),cv2.COLOR_BGR2RGB)/255.0*2.0-1.0) for fname in glob.glob(FLAGS.valid_img_dir)], dtype=np.object)
            valid_label = np.array([(np.array(cv2.cvtColor(cv2.imread(fname),cv2.COLOR_BGR2RGB))/255.0*2.0-1.0) for fname in glob.glob(FLAGS.valid_label_dir)], dtype=np.object)
            valid_x=tf.placeholder(tf.float32,[1,None,None,3])
            valid_y=tf.placeholder(tf.float32,[1,None,None,3])
        else:
            valid_dataset = make_dataset(FLAGS.valid_img_dir, FLAGS.valid_label_dir, train=False, batch_size=1) 
            valid_it = valid_dataset.make_initializable_iterator()
            valid_x, valid_y = valid_it.get_next()

        train_pred = model.forward(train_x, True, reuse=False)
        train_loss = model.loss_function(train_y, train_pred)
        train_psnr = utils.compute_psnr_tf(train_pred, train_y)
        train_op = model.optimize(train_loss)

        valid_pred = model.forward(valid_x, False, reuse=True)
        valid_loss = model.loss_function(valid_y, valid_pred) 
        valid_psnr = utils.compute_psnr_tf(valid_pred, valid_y)
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        saver = tf.train.Saver(var_list)

        if not FLAGS.perceptual_mode == 'MSE':
            vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
            vgg_restore = tf.train.Saver(vgg_var_list)


        with tf.name_scope('train_summary'):
            ts1= tf.summary.image('input_summary', train_x)
            ts2 = tf.summary.image('target_summary', train_y)
            ts3 = tf.summary.image('outputs_summary', tf.clip_by_value(train_pred,0,1))
        train_img_merged = tf.summary.merge([ts1, ts2, ts3])
        with tf.name_scope('valid_summary'):
            vs1= tf.summary.image('input_summary', valid_x)
            vs2 = tf.summary.image('target_summary', valid_y)
            vs3 = tf.summary.image('outputs_summary', tf.clip_by_value(valid_pred,0,1))
        valid_img_merged = tf.summary.merge([vs1, vs2, vs3])

        train_loss_avg = tf.placeholder(tf.float32)
        valid_loss_avg = tf.placeholder(tf.float32)
        ts_loss = tf.summary.scalar('train_loss', train_loss_avg)
        vs_loss = tf.summary.scalar('valid_loss', valid_loss_avg) 
        train_psnr_avg = tf.placeholder(tf.float32)
        valid_psnr_avg = tf.placeholder(tf.float32)
        ts_psnr = tf.summary.scalar('train_psnr', train_psnr_avg)
        vs_psnr = tf.summary.scalar('valid_psnr', valid_psnr_avg) 
        scalar_merged= tf.summary.merge([ts_loss,vs_loss,ts_psnr,vs_psnr])
        

        with tf.Session() as sess:

            writer = tf.summary.FileWriter('./board/graph/'+FLAGS.model_name, sess.graph)
            writer.add_graph(sess.graph)

            sess.run(tf.global_variables_initializer())
            loaded, start_epoch = model.load(sess, saver, os.path.join(FLAGS.checkpoint_dir,FLAGS.model_name))
            if loaded:
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            if not FLAGS.perceptual_mode == 'MSE':
                vgg_restore.restore(sess, FLAGS.vgg_ckpt)
                print('VGG19 restored successfully!!')
            
            for epoch in tqdm(range(start_epoch,FLAGS.epochs)):
                start_time = time.time()
                sess.run(train_it.initializer)
                t_loss = 0.0
                t_psnr = 0.0
                count = 0
                try:
                    while True:
                        # loss = 0
                        # _ = sess.run(train_x)
                        _, loss, psnr, train_img_summary = sess.run([train_op, train_loss, train_psnr, train_img_merged])
                        t_loss += loss
                        t_psnr += psnr
                        count += 1
                except tf.errors.OutOfRangeError:
                    pass
                t_loss /= count
                t_psnr /= count

                v_loss = 0.0
                v_psnr = 0.0

                if FLAGS.placeholder:
                    for vd, vl in zip(valid_data,valid_label):
                        loss, psnr, valid_img_summary = sess.run([valid_loss, valid_psnr,valid_img_merged], feed_dict={valid_x: vd[np.newaxis,:], valid_y:vl[np.newaxis,:]})
                        v_loss += loss
                        v_psnr += psnr
                    count = len(valid_data)
                else:
                    sess.run(valid_it.initializer)
                    count = 0
                    while True:
                        try:
                        # _ = sess.run(valid_x)
                        # v_loss = 0
                            loss, psnr, valid_img_summary = sess.run([valid_loss, valid_psnr,valid_img_merged])
                            v_loss += loss
                            v_psnr += psnr
                            count += 1 
                        except tf.errors.OutOfRangeError:
                            break
                v_loss /= count
                v_psnr /= count
                print("Epoch: [%2d], time: [%4.4f], train_loss: [%.8f], train_psnr: [%.4f], valid_loss: [%.8f], valid_psnr: [%.4f]"% ((epoch+1), time.time()-start_time, t_loss, t_psnr, v_loss, v_psnr))
                model.save(sess, saver, os.path.join(FLAGS.checkpoint_dir,FLAGS.model_name), epoch)
                
                summary = sess.run(scalar_merged, feed_dict={train_loss_avg: t_loss, valid_loss_avg: v_loss, train_psnr_avg: t_psnr, valid_psnr_avg: v_psnr})
            
                writer.add_summary(train_img_summary, epoch)
                writer.add_summary(valid_img_summary, epoch)
                writer.add_summary(summary, epoch)
        
    else:
        valid_dataset = make_dataset(FLAGS.test_img_dir, FLAGS.test_label_dir, train=False, batch_size=1)
        valid_it = valid_dataset.make_initializable_iterator()

        valid_x, valid_y = valid_it.get_next()
        #valid_bicubic = bicubic_upsample_x2_tf(valid_x)
        valid_pred = model.forward(valid_x, is_train=False,reuse=tf.AUTO_REUSE)

        valid_loss = model.loss_function(valid_y, valid_pred) 
        #valid_bic_psnr = compute_psnr_tf(valid_y, valid_bicubic)
        valid_psnr = compute_psnr_tf(valid_y, valid_pred)
        #valid_bic_ssim = compute_ssim_tf(valid_y, valid_bicubic)
        valid_ssim = compute_ssim_tf(valid_y, valid_pred)

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
        saver = tf.train.Saver(var_list)
        with tf.Session() as sess:
            loaded, _ = model.load(sess, saver, os.path.join(FLAGS.checkpoint_dir,FLAGS.model_name))
            if loaded:
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
            sess.run(valid_it.initializer)
            v_loss = 0.0
            v_psnr = 0.0
            v_ssim = 0.0
            v_bic_psnr = 0.0
            v_bic_ssim = 0.0
            count = 0
            f = open(FLAGS.sample_dir+"/result.txt","w")
            try:
                while True:
                    start_time = time.time()
                    loss, psnr, ssim, x, y, pred = sess.run([valid_loss, valid_psnr, valid_ssim, valid_x, valid_y, valid_pred])
                    #bic_x = x[0] # when bicubic upsampled
                    bic_x = bicubic_upsample_x2_np(x[0])
                    bic_psnr = compute_psnr_np(y[0], bic_x)
                    bic_ssim = compute_ssim_np(y[0], bic_x)
                    # loss, psnr, bic_psnr, ssim, bic_ssim, bic_x, x, y, pred = sess.run([valid_loss, valid_psnr, valid_bic_psnr, valid_ssim, valid_bic_ssim, valid_bicubic, valid_x, valid_y, valid_pred])
                    if count< 10:
                        save_image(FLAGS.sample_dir+"/LR_"+str(count)+".jpg", x[0])
                        save_image(FLAGS.sample_dir+"/HR_"+str(count)+".jpg", y[0])
                        save_image(FLAGS.sample_dir+"/bicubic_"+str(count)+".jpg", bic_x)
                        #save_image(FLAGS.sample_dir+"/bicubic_"+str(count)+".jpg", bic_x[0])
                        save_image(FLAGS.sample_dir+"/pred_"+str(count)+".jpg", pred[0])
                    v_loss += loss
                    v_psnr += psnr
                    v_ssim += ssim
                    v_bic_psnr += bic_psnr
                    v_bic_ssim += bic_ssim
                    f.write("%dth img => time: [%4.4f], loss: [%.8f], psnr: [%.4f], bicubic_psnr: [%.4f], ssim: [%.4f], bicubic_ssim: [%.4f]\n"% ((count), time.time()-start_time, loss, psnr, bic_psnr, ssim, bic_ssim))
                    count += 1
            except tf.errors.OutOfRangeError:
                pass
            #v_loss /= count
            f.write("Avg. Loss : %.8f, Avg. PSNR : %.4f, Avg. SSIM : %.4f, Avg. BICUBIC_PSNR : %.4f, Avg. BICUBIC_SSIM : %.4f"%(v_loss/count, v_psnr/count, v_ssim/count, v_bic_psnr/count, v_bic_ssim/count))
            f.close()

if __name__=='__main__':
    main()