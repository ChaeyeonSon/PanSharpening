import tensorflow as tf
import numpy as np
import os 
import glob 
from PIL import Image
import tifffile
import cv2
from tqdm import tqdm


def decode_img(img, neg):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_png(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32) 
  # convert range into [-1, 1]
  if neg:
    img = img*2. - 1.
  return img
  
def downsampling(img, factor):
  width, height, depth = img.shape
  gb = cv2.GaussianBlur(img,(5,5),1/factor)
  result = np.zeros((width//factor, height//factor, depth))
  #print(gb.shape)
  for i in range(width//factor):
    for j in range(height//factor):
      result[i,j,:] = np.mean(np.mean(gb[i*factor:(i+1)*factor, j*factor:(j+1)*factor,:], axis=0), axis=0)
  return result
  
# path = "../PSdata/RGB/*.tif"
# data = tifffile.imread(glob.glob(path)[0])
# print(np.max(data))
# data = data/np.max(data) "../PSdata/RGB/*.tif"
# data = tifffile.imread(glob.glob(path)[0])
# print(np.max(data))
# data = data/np.max(data)
# cv2.imshow("a",data)
# cv2.waitKey(1000)
# s = downsampling(data, 4)
# cv2.imshow("name",s)
# cv2.waitKey(1000)

# cv2.imshow("a",data)
# cv2.waitKey(1000)
# s = downsampling(data, 4)
# cv2.imshow("name",s)
# cv2.waitKey(1000)

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

def process_path_tiff_w_pan_supervised(pan_file_path, ms_file_path):
  #img_file_path, label_file_path = file_path
  # 저장은 나중에 
  # if not os.path.exists('./minmax.txt'):
  #   f = open("./minmax.txt","w")
  data_list = [os.path.basename(x) for x in glob.glob(pan_file_path+'/*.tif')]
  imgs = []
  labels=[]
  for data in data_list:
    pan = np.array(tifffile.imread(pan_file_path+os.path.sep+data))
    p_mean = np.mean(pan)
    p_std = np.std(pan)
    pan = (pan-p_mean)/p_std
    ms = np.array(Image.open(ms_file_path+os.path.sep+data))
    m_std_r=np.std(ms[:,:,0]) 
    m_std_g=np.std(ms[:,:,1]) 
    m_std_b=np.std(ms[:,:,2]) 
    m_std = np.array([m_std_r, m_std_g, m_std_b])
    m_mean=np.mean(np.mean(ms, axis=0), axis=0)
    ms = (ms-m_mean)/m_std
    down_ms = downsampling(ms,4)
    imgs.append(down_ms)
    labels.append(ms) 
  img = np.array(imgs)
  label= np.array(labels)
  return img, label


def generator_tiff_wo_pan_supervised(pan_file_path, ms_file_path):
  #img_file_path, label_file_path = file_path
  # 저장은 나중에 
  # if not os.path.exists('./minmax.txt'):
  #   f = open("./minmax.txt","w")
  data_list = [os.path.basename(x) for x in glob.glob(ms_file_path+b'/*.tif')]
  # imgs = []
  # labels=[]
  for data in tqdm(data_list):
    ms_data = ms_file_path+b'/'+data
    ms = tifffile.imread(ms_data.decode("utf-8"))
    if ms.shape[0] %4 !=0:
      ms = ms[:ms.shape[0]//4*4,:ms.shape[1]//4*4,:]
    m_std_r=np.std(ms[:,:,0]) 
    m_std_g=np.std(ms[:,:,1]) 
    m_std_b=np.std(ms[:,:,2]) 
    m_std = np.array([m_std_r, m_std_g, m_std_b]) 
    m_mean=np.mean(np.mean(ms, axis=0), axis=0)
    ms = (ms-m_mean)/m_std
    down_ms = downsampling(ms,4)
    yield down_ms, ms
    # imgs.append(down_ms)
    # labels.append(ms) 
    # if i > 100:
    #   break
  # img = np.array(imgs,axis=0)
  # label= np.stack(labels,axis=0)
  # print(img.shape, label.shape)
  # return img, label

import random

def randomCrop(img, mask, width, height, factor=2):
    #assert img.shape[0] >= height
    #assert img.shape[1] >= width
    #assert img.shape[0]*2 == mask.shape[0]2
    #assert img.shape[1]*2 == mask.shape[1]
    
    x=tf.random.uniform([], minval = 0, maxval=tf.shape(img)[1] - width, dtype=tf.int32)    
    y=tf.random.uniform([], minval = 0, maxval=tf.shape(img)[0] - height, dtype=tf.int32)
    img = tf.image.crop_to_bounding_box(img,y, x,height,width)
    mask = tf.image.crop_to_bounding_box(mask,y*factor, x*factor, height*factor, width*factor)
    return img, mask

def random_crop32_PAN(ms, pan):
  return randomCrop(ms,pan, 32, 32, 4)


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
    #train_labeled_ds = ds.shuffle(10).map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).prefech(10)
    return train_labeled_ds

  else:
    valid_labeled_ds = ds.map(process_path, num_parallel_calls=1).batch(1).prefetch(1)
    # valid_labeled_ds = ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE)
    return valid_labeled_ds

def make_dataset_ps(pan_file_path, ms_file_path, train=True, batch_size=1):
  if train:  
    # ts = process_path_tiff_wo_pan_supervised(pan_file_path, ms_file_path)
    # ds = tf.data.Dataset.from_tensor_slices(ts)
    ds = tf.data.Dataset.from_generator(generator_tiff_wo_pan_supervised,args=[pan_file_path, ms_file_path],\
      output_types=(tf.float32, tf.float32),output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])))
    return ds.map(random_crop32_PAN).shuffle(100).batch(4).prefetch(tf.data.experimental.AUTOTUNE)
  else:
    # ts = process_path_tiff_wo_pan_supervised(pan_file_path, ms_file_path)
    # ds = tf.data.Dataset.from_tensor_slices(ts)
    ds = tf.data.Dataset.from_generator(generator_tiff_wo_pan_supervised,args=[pan_file_path, ms_file_path],\
      output_types=(tf.float32, tf.float32),output_shapes=(tf.TensorShape([None, None, 3]), tf.TensorShape([None, None, 3])))
    return ds.batch(1).prefetch(tf.data.experimental.AUTOTUNE)

def make_dataset_unsupervised(pan_file_path, ms_file_path, train=True, batch_size=1):
  ds1 = tf.data.Dataset.list_files(pan_file_path, shuffle=False)
  ds2 = tf.data.Dataset.list_files(ms_file_path, shuffle=False)
  
  ts = process_path_glob(pan_file_path, ms_file_path)
  ds = tf.data.Dataset.from_tensor_slices(ts)
  return ds.shuffle(100).map(random_crop_size32)
  