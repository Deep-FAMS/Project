#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system(' nvidia-smi')


# In[ ]:


# ! wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
# ! tar -xvzf 102flowers.tgz
# ! mv 102flowers 102flowers_dataset_raw


# ### Use tensorflow_version 2.x

# In[ ]:


import os
from PIL import Image
from glob import glob
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf    # tensorflow_version 2.x
print(tf.__version__)


# In[ ]:


def resize_imgs(imgs_dir, output_dir, img_dim=(256, 256)):
    
    raw_imgs_ls = glob(f'{imgs_dir}/*.jpg')
    
    for x in tqdm(raw_imgs_ls):
        img = Image.open(x)
        img_name = Path(img.filename).stem
        
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        resized_array = tf.keras.preprocessing.image.smart_resize(img_array, img_dim, interpolation='bilinear')
        resized_img = tf.keras.preprocessing.image.array_to_img(resized_array)
        
        Path(output_dir).mkdir(exist_ok=True)
        resized_img.save(f'{output_dir}/{img_name}.jpg')


# In[ ]:


resize_imgs(imgs_dir='102flowers_dataset_raw', output_dir='102flowers_resized')


# ### Switch to tensorflow_version 1.x

# In[ ]:


import os
from glob import glob
import numpy as np
import PIL.Image

import tensorflow as tf    # tensorflow_version 1.x
print(tf.__version__)


# In[ ]:


class TFRecordExporter:
    def __init__(self, tfrecord_dir, expected_images, print_progress=True, progress_interval=10, tfr_prefix=None):
        self.tfrecord_dir       = tfrecord_dir
        if tfr_prefix is None:
            self.tfr_prefix = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        else:
            self.tfr_prefix = os.path.join(self.tfrecord_dir, tfr_prefix)
        self.expected_images    = expected_images
        self.cur_images         = 0
        self.shape              = None
        self.resolution_log2    = None
        self.tfr_writers        = []
        self.print_progress     = print_progress
        self.progress_interval  = progress_interval

        if self.print_progress:
            name = '' if tfr_prefix is None else f' ({tfr_prefix})'
            print(f'Creating dataset "{tfrecord_dir}"{name}')
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert os.path.isdir(self.tfrecord_dir)

    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    def choose_shuffled_order(self): # Note: Images and labels must be added in shuffled order.
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    def add_image(self, img):
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.shape = img.shape
            self.resolution_log2 = int(np.log2(self.shape[1]))
            assert self.shape[0] in [1, 3]
            assert self.shape[1] == self.shape[2]
            assert self.shape[1] == 2**self.resolution_log2
            tfr_opt = tf.io.TFRecordOptions(tf.compat.v1.io.TFRecordCompressionType.NONE)
            for lod in range(self.resolution_log2 - 1):
                tfr_file = self.tfr_prefix + '-r%02d.tfrecords' % (self.resolution_log2 - lod)
                self.tfr_writers.append(tf.io.TFRecordWriter(tfr_file, tfr_opt))
        assert img.shape == self.shape
        for lod, tfr_writer in enumerate(self.tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_labels(self, labels):
        if self.print_progress:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        assert labels.shape[0] == self.cur_images
        with open(self.tfr_prefix + '-rxx.labels', 'wb') as f:
            np.save(f, labels.astype(np.float32))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# In[ ]:


def error(msg):
    print('Error: ' + msg)
    exit(1)

def create_from_images(tfrecord_dir, image_dir, shuffle):
    print('Loading images from "%s"' % image_dir)
    image_filenames = sorted(glob(os.path.join(image_dir, '*')))
    if len(image_filenames) == 0:
        error('No input images found')

    img = np.asarray(PIL.Image.open(image_filenames[0]))
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1
    if img.shape[1] != resolution:
        error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        error('Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')

    with TFRecordExporter(tfrecord_dir, len(image_filenames)) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(len(image_filenames))
        for idx in range(order.size):
            img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
            if channels == 1:
                img = img[np.newaxis, :, :] # HW => CHW
            else:
                img = img.transpose([2, 0, 1]) # HWC => CHW
            tfr.add_image(img)


# In[ ]:


WORK = os.environ["WORK"]


# In[ ]:


create_from_images(tfrecord_dir=f'{WORK}/datasets/102flowers_custom',
                   image_dir=f'{WORK}/102flowers_resized', shuffle=1)


# In[ ]:


get_ipython().run_line_magic('alias', 'python_tf1 "$WORK/.conda/envs/ada-env/bin/python"')


# In[ ]:


# ! git clone https://github.com/Deep-FAMS/stylegan2-ada.git
get_ipython().system(' cd "stylegan2-ada" && git pull')


# In[ ]:


get_ipython().system(' $WORK/.conda/envs/ada-env/bin/python stylegan2-ada/train.py stylegan2-ada/train.py     --outdir=$WORK/102flowers_training-runs     --gpus=1     --data=$WORK/datasets/102flowers_custom     --dry-run')


# In[1]:


from glob import glob
import os

def last_snap(num):
    files = glob(f'{sorted(glob(os.environ["WORK"] + "/102flowers_training-runs/*"))[num]}/*')
    files = [x for x in files if 'network-snapshot' in x]
    return files

for n in range(-1, -10, -1):
    files = last_snap(n)
    if files != []:
        break

file = sorted(files)[-1]
print(file)


# In[ ]:


get_ipython().system(' module load anaconda &&     module load compiler/gcc/4.7 &&     module load cuda &&     $WORK/.conda/envs/ada-env/bin/python stylegan2-ada/train.py     --outdir=$WORK/102flowers_training-runs     --gpus=2     --data=$WORK/datasets/102flowers_custom     --snap=1     --resume="/work/chaselab/malyetama/102flowers_training-runs/00022-102flowers_custom-auto1-resumecustom/network-snapshot-000161.pkl"')


# In[ ]:




