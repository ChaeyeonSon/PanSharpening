import tensorflow as tf
import numpy as np
import os 
import glob 



train_img_data_dir = '../train_data/DIV2K_train_LR_bicubic/*.png'
train_label_data_dir = '../train_data/DIV2K_train_HR/*.png'
valid_img_data_dir = '../DIV2K_valid_LR_bicubic/*.png'
valid_label_data_dir = '../DIV2K_valid_HR/*.png'

#train_img_data_list = os.listdir(train_img_data_dir)
#train_label_data_list = os.listdir(train_label_data_dir)
#print(train_img_data_list)
#print(train_label_data_list)

def decode_img(img, neg):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_png(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32) 
  # convert range into [-1, 1]
  if neg:
    img = img*2. - 1.
  return img
  
def process_path(img_file_path, label_file_path):
  #img_file_path, label_file_path = file_path
  label = tf.io.read_file(label_file_path)
  label = decode_img(label, neg=False)
  # load the raw data from the file as a string
  img = tf.io.read_file(img_file_path)
  img = decode_img(img, False)
  return img, label
import random
def randomCrop(img, mask, width, height,bicubic_upsampled=False):
    #assert img.shape[0] >= height
    #assert img.shape[1] >= width
    #assert img.shape[0]*2 == mask.shape[0]
    #assert img.shape[1]*2 == mask.shape[1]
    
    if bicubic_upsampled:  
        x=tf.random.uniform([], minval = 0, maxval=tf.shape(img)[1] - width*2, dtype=tf.int32)
        y=tf.random.uniform([], minval = 0, maxval=tf.shape(img)[0] - height*2, dtype=tf.int32)
        img = tf.image.crop_to_bounding_box(img,y, x, height*2, width*2)              
        mask = tf.image.crop_to_bounding_box(mask,y, x, height*2, width*2)
    else:
        x=tf.random.uniform([], minval = 0, maxval=tf.shape(img)[1] - width, dtype=tf.int32)    
        y=tf.random.uniform([], minval = 0, maxval=tf.shape(img)[0] - height, dtype=tf.int32)
        img = tf.image.crop_to_bounding_box(img,y, x,height,width)
        mask = tf.image.crop_to_bounding_box(mask,y*2, x*2, height*2, width*2)
    return img, mask

def random_crop_size64(images, labels):
  
  return randomCrop(images,labels,64,64)

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.

def random_crop_size96(images, labels):

  return randomCrop(images,labels,48,48)


def make_dataset(img_file_path, label_file_path, train=True, batch_size=128, feed_dict=False):
  ds1 = tf.data.Dataset.list_files(img_file_path, shuffle=False)
  ds2 = tf.data.Dataset.list_files(label_file_path, shuffle=False)
  ds = tf.data.Dataset.zip((ds1, ds2))
  if train:
    train_labeled_ds = ds.shuffle(800).map(process_path, num_parallel_calls=10).map(random_crop_size64).batch(batch_size).prefetch(10)

#    train_labeled_ds = ds.shuffle(800).map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE).map(random_crop_size64).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return train_labeled_ds

  else:
    valid_labeled_ds = ds.map(process_path, num_parallel_calls=1).batch(1).prefetch(1)

#    valid_labeled_ds = ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE)
    return valid_labeled_ds