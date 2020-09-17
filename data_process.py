import tensorflow as tf
import numpy as np
import os 
import glob 
from PIL import Image
import tifffile
import cv2
from tqdm import tqdm
import random
from utils import *

N_CPU = 4
NUM_SUB_PER_IM=30

def decode_img(img, neg):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_png(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32) 
  # convert range into [-1, 1]
  if neg:
    img = img*2. - 1.
  return img
  

def normalize_ms_pan(down_ms, down_pan, ms):
  return normalize(down_ms, 3), normalize(down_pan, 1), normalize(ms, 3)

import random

def randomCrop(img, mask, width, height, factor=2):

  x=tf.random.uniform([], minval = 0, maxval=tf.shape(img)[1] - width, dtype=tf.int32)    
  y=tf.random.uniform([], minval = 0, maxval=tf.shape(img)[0] - height, dtype=tf.int32)
  img = tf.image.crop_to_bounding_box(img,y, x,height,width)
  mask = tf.image.crop_to_bounding_box(mask,y*factor, x*factor, height*factor, width*factor)
  return img, mask
    
def randomCrop_np(imgs, width, height, factors):
  x = np.random.randint(0, imgs[0].shape[0] - width)
  y = np.random.randint(0, imgs[0].shape[1] - height)
  cropped=[0]*len(imgs)
  for i, img in enumerate(imgs):
    cropped[i]=img[x*factors[i]: (x+width)*factors[i], y*factors[i]:(y+height)*factors[i], :]
  return cropped

def random_crop32_pan_np(ms, pan, label):
  return randomCrop_np([ms,pan,label],32,32,[1,4,4])

def random_crop32_PAN(ms, pan):
  return randomCrop(ms,pan, 32, 32, 4)


def random_crop_size64(images, labels):
  
  return randomCrop(images,labels,64,64)

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.

def random_crop_size96(images, labels):

  return randomCrop(images,labels,48,48)

def pannet_input_process(ms, pan):

  hp_pan= hp_filter(pan)
  hp_ms = bicubic_upsample_np(hp_filter(ms),4)
  spectral = bicubic_upsample_np(ms,4)
  # print(pan.shape,hp_pan.shape, hp_ms.shape, spectral.shape, "WHATR")
  return np.concatenate((hp_pan, hp_ms, spectral), axis=-1)


def process_path(pan_file_path, ms_file_path):
  label = tf.io.read_file(ms_file_path)
  #img_file_path, label_file_path = file_path
  # 저장은 나중에 
  # if not os.path.exists('./minmax.txt'):
  label = decode_img(label, neg=False)
  # load the raw data from the file as a string
  img = tf.io.read_file(pan_file_path)
  img = decode_img(img, False)
  img = tf.nn.space_to_depth(img, 4)
  return img, label

def process_path_tiff_w_pan(pan_file_path, ms_file_path):
  #img_file_path, label_file_path = file_path
  # 저장은 나중에 
  # if not os.path.exists('./minmax.txt'):
  #   f = open("./minmax.txt","w")
  data_list = [os.path.basename(x) for x in glob.glob(pan_file_path+'/*.tif')]
  data = []
  for data in data_list:
    pan = np.array(tifffile.imread(pan_file_path+os.path.sep+data))
    p_max = np.max(pan)
    p_min = np.min(pan)
    pan = (pan-p_min)/(p_max-p_min)
    #pan = space_to_depth(pan,4)
    ms = np.array(Image.open(ms_file_path+os.path.sep+data))
    m_max=np.max(np.max(ms,axis=2), axis=1)
    m_min=np.min(np.min(ms, axis=2), axis=1)
    ms = (ms-m_min)/(m_max-m_min)
    data.append(np.concatenate((pan,ms),axis=2)) 
  img = np.array(data)
  return img


def generator_tiff_w_pan_supervised(ms_file_path, pan_file_path):
  ## wrong!! crop -> normalize
  data_list = [os.path.basename(x) for x in glob.glob(ms_file_path+b'/*.tif')]#*NUM_SUB_PER_IM#???
  random.shuffle(data_list)
  # indrand = np.random.randint(len(data_list))
  i = 0
  for data in tqdm(data_list):
    ms_data = ms_file_path+b'/'+data
    ms = tifffile.imread(ms_data.decode("utf-8"))
    # ms_data = ms_file_path+b'/'+data
    pan = tifffile.imread((pan_file_path+b'/'+data).decode("utf-8"))
    pan = pan[:,:,np.newaxis]
    if ms.shape[0] %4 !=0:
      ms = ms[:ms.shape[0]//4*4,:ms.shape[1]//4*4,:]
      pan = pan[:pan.shape[0]//16*16, :pan.shape[1]//16*16, :]
    down_ms = downsampling(ms,4)
    down_pan = downsampling(pan, 4)
    crop_ms_lr, crop_pan_lr, crop_ms_hr =random_crop32_pan_np(down_ms, down_pan, ms)
    norm_ms_lr, norm_pan_lr, norm_ms_hr = normalize_ms_pan(crop_ms_lr, crop_pan_lr, crop_ms_hr)
    input_ = pannet_input_process(norm_ms_lr, norm_pan_lr)
    yield input_, norm_ms_hr
    #yield down_ms, down_pan, ms


def make_dataset(img_file_path, label_file_path, train=True, batch_size=128, feed_dict=False):
  ds1 = tf.data.Dataset.list_files(img_file_path, shuffle=False)
  ds2 = tf.data.Dataset.list_files(label_file_path, shuffle=False)
  ds = tf.data.Dataset.zip((ds1, ds2))
  if train:
    train_labeled_ds = ds.shuffle(800).map(process_path, num_parallel_calls=10).map(random_crop_size64).batch(batch_size).prefetch(10)
    #train_labeled_ds = ds.shuffle(10).map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefech(10)
    return train_labeled_ds

  else:
    valid_labeled_ds = ds.map(process_path, num_parallel_calls=1).batch(1).prefetch(1)
    # valid_labeled_ds = ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE)
    return valid_labeled_ds

def tf_ds_gen(pan_file_path, ms_file_path):
  dataset0 = tf.data.Dataset.from_generator(generator_tiff_w_pan_supervised,args=[pan_file_path, ms_file_path], \
                    output_types=(tf.float32, tf.float32), output_shapes=(tf.TensorShape([None, None, 7]), tf.TensorShape([None, None, 3])))#.map(random_crop32_PAN).map(normalize_ms_pan)
  return dataset0

def make_dataset_ps(ms_file_path, pan_file_path, train=True, batch_size=1):
  if train:  
    # ts = process_tiff_wo_pan_supervised(None, ms_file_path)
    # tss = tf.placeholder(tf.float32, shape=[None])
    # ds = tf.data.Dataset.from_tensor_slices(tss)
    dataset = (tf.data.Dataset.range(100).repeat()
               .apply(
        tf.data.experimental.parallel_interleave(lambda filename: tf_ds_gen(ms_file_path, pan_file_path), cycle_length=int(N_CPU), sloppy=False))
               #.apply(tf.data.experimental.unbatch())
               #.shuffle(buffer_size=int(NUM_SUB_PER_IM * 10))
               .batch(batch_size)
               )
    # ds = tf.data.Dataset.from_generator(generator_tiff_wo_pan_supervised,args=[pan_file_path, ms_file_path],\
    #   output_types=(tf.float32, tf.float32,tf.float32, tf.float32),output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])))
    #return ds.map(random_crop32_PAN).shuffle(1000).batch(4).repeat().prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
  else:
    # ts = process_tiff_wo_pan_supervised(None, ms_file_path)
    # ds = tf.data.Dataset.from_tensor_slices(ts)
    ds = tf.data.Dataset.from_generator(generator_tiff_w_pan_supervised,args=[ms_file_path, pan_file_path],\
      output_types=(tf.float32, tf.float32),output_shapes=(tf.TensorShape([None, None, 7]), tf.TensorShape([None, None, 3])))
    return ds.batch(1).prefetch(tf.data.experimental.AUTOTUNE)

def make_dataset_unsupervised(pan_file_path, ms_file_path, train=True, batch_size=1):
  ds1 = tf.data.Dataset.list_files(pan_file_path, shuffle=False)
  ds2 = tf.data.Dataset.list_files(ms_file_path, shuffle=False)
  
  ts = process_path_glob(pan_file_path, ms_file_path)
  ds = tf.data.Dataset.from_tensor_slices(ts)
  return ds.shuffle(100).map(random_crop_size32)
  

