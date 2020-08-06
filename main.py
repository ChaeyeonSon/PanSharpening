import tensorflow as tf 
import numpy as np
import argparse
# from data_process import *
from model import *

import os
import glob
import pprint
from tqdm import tqdm
import time
import cv2

import utils

import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

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
flags.DEFINE_string("train_img_dir", "/media/kaist/SSD_1T/cyson/PSdata/RGB", "Name of train img directory [train_img]")
flags.DEFINE_string("train_label_dir", "../sr/train_data/DIV2K_train_HR/*.png", "Name of train label directory [train_label]")

flags.DEFINE_string("valid_img_dir", "/media/kaist/SSD_1T/cyson/PSdata/RGB_test", "Name of valid img directory [valid_img]")
flags.DEFINE_string("valid_label_dir", '../sr/valid_data/DIV2K_valid_HR/*.png', "Name of svalid label directory [valid_label]")

flags.DEFINE_string("test_img_dir", "/media/kaist/SSD_1T/cyson/PSdata/RGB_test", "Name of valid img directory [valid_img]")
flags.DEFINE_string("test_label_dir", '../test_data/DIV2K_valid_HR/*.png', "Name of svalid label directory [valid_label]")

flags.DEFINE_string("model_name", 'PS_supervised', "Name of Experiment")

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
    model = SRGenerator(content_loss=FLAGS.perceptual_mode, num_upsamples=4) 
    
    model.training(tf.Session(),FLAGS)

if __name__=='__main__':
    main()
