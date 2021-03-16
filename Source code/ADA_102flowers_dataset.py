#!/usr/bin/env python
# coding: utf-8

# ## Download stylegan2-ada repo

# In[ ]:


# ! git clone https://github.com/Deep-FAMS/stylegan2-ada.git
# ! cd "stylegan2-ada" && git pull


# In[2]:


get_ipython().run_line_magic('cd', 'stylegan2-ada')


# ## Download 102flowers raw data

# In[ ]:


# ! wget https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
# ! tar -xvzf 102flowers.tgz
# ! mv 102flowers 102flowers_dataset_raw


# ### Use tensorflow_version 2.x

# In[ ]:


import tensorflow as tf    # tensorflow_version 2.x
print(tf.__version__)


# ## Resize images to unify the size (256x256)

# In[ ]:


import os
from PIL import Image
from glob import glob
from pathlib import Path
from tqdm import tqdm


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


import tensorflow as tf    # tensorflow_version 1.x
print(tf.__version__)


# ## Convert dataset to `TFRecordExporter` format

# In[ ]:


import os
from glob import glob
import numpy as np
import PIL.Image


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
create_from_images(tfrecord_dir=f'{WORK}/datasets/102flowers_custom',
                   image_dir=f'{WORK}/102flowers_resized', shuffle=1)


# ## Set enviroment variables and prepend/append paths of `Anaconda`, `compiler/gcc/4.7` and `Cuda`

# In[3]:


import os
import sys

prepend_paths = []

def prepend_path(x, y):
    prepend_paths.append(y)

prepend_path("PATH","/util/opt/cuda/10.2/bin")
prepend_path("LD_LIBRARY_PATH","/util/opt/cuda/10.2/lib64")
prepend_path("LIBRARY_PATH","/util/opt/cuda/10.2/lib64")
prepend_path("MANPATH","/util/opt/cuda/10.2/doc/man")
prepend_path("CPATH","/util/opt/cuda/10.2/include")
# prepend_path("CONDA_ENVS_PATH","~/.conda/envs")

# sys.path.append("/util/opt/anaconda3/2.0/envs")
# sys.path.append("/util/opt/anaconda/2.2/envs")
# sys.path.append("/util/opt/anaconda/4.3/envs")

# prepend_path("PATH","/util/opt/anaconda/4.8.2/condabin")
# prepend_path("PATH","/util/opt/anaconda/4.8/condabin")

prepend_path("PATH","/util/comp/gcc/4.7/bin")
prepend_path("LD_LIBRARY_PATH","/util/comp/gcc/4.7/lib")
prepend_path("LD_LIBRARY_PATH","/util/comp/gcc/4.7/lib64")
prepend_path("LIBRARY_PATH","/util/comp/gcc/4.7/lib")
prepend_path("LIBRARY_PATH","/util/comp/gcc/4.7/lib64")
prepend_path("MANPATH","/util/comp/gcc/4.7/share/man")
prepend_path("INCLUDE","/util/comp/gcc/4.7/include")
prepend_path("PKG_CONFIG_PATH","/util/comp/gcc/4.7/lib/pkgconfig")
prepend_path("MODULEPATH","/util/opt/modulefiles/Compiler/gcc/4.7")

os.environ["GCC_LIB"] = "/util/comp/gcc/4.7/lib64"
os.environ["CC"] = "gcc"
os.environ["FC"] = "gfortran"
os.environ["F90"] = "gfortran"
os.environ["F77"] = "gfortran"
os.environ["CXX"] = "g++"
os.environ["CFLAGS"] = "-march=corei7-avx"
os.environ["FFLAGS"] = "-march=corei7-avx"
os.environ["CXXFLAGS"] = "-march=corei7-avx"
os.environ["F90FLAGS"] = "-march=corei7-avx"
os.environ["FCFLAGS"] = "-march=corei7-avx"
    
os.environ["CUDA_HOME"] = "/util/opt/cuda/10.2"
os.environ["CUDA_PATH"] = "/util/opt/cuda/10.2"


for item in prepend_paths:
    sys.path.insert(0, item)


# ## Setup training options

# In[4]:


import os
import argparse
import json
import re
import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib

from training import training_loop
from training import dataset
from metrics import metric_defaults

#----------------------------------------------------------------------------

class UserError(Exception):
    pass

#----------------------------------------------------------------------------

def setup_training_options(
    # General options (not included in desc).
    gpus       = None, # Number of GPUs: <int>, default = 1 gpu
    snap       = None, # Snapshot interval: <int>, default = 50 ticks

    # Training dataset.
    data       = None, # Training dataset (required): <path>
    res        = None, # Override dataset resolution: <int>, default = highest available
    mirror     = None, # Augment dataset with x-flips: <bool>, default = False

    # Metrics (not included in desc).
    metrics    = None, # List of metric names: [], ['fid50k_full'] (default), ...
    metricdata = None, # Metric dataset (optional): <path>

    # Base config.
    cfg        = None, # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar', 'cifarbaseline'
    gamma      = None, # Override R1 gamma: <float>, default = depends on cfg
    kimg       = None, # Override training duration: <int>, default = depends on cfg

    # Discriminator augmentation.
    aug        = None, # Augmentation mode: 'ada' (default), 'noaug', 'fixed', 'adarv'
    p          = None, # Specify p for 'fixed' (required): <float>
    target     = None, # Override ADA target for 'ada' and 'adarv': <float>, default = depends on aug
    augpipe    = None, # Augmentation pipeline: 'blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc' (default), ..., 'bgcfnc'

    # Comparison methods.
    cmethod    = None, # Comparison method: 'nocmethod' (default), 'bcr', 'zcr', 'pagan', 'wgangp', 'auxrot', 'spectralnorm', 'shallowmap', 'adropout'
    dcap       = None, # Multiplier for discriminator capacity: <float>, default = 1

    # Transfer learning.
    resume     = None, # Load previous network: 'noresume' (default), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>
    freezed    = None, # Freeze-D: <int>, default = 0 discriminator layers
):
    # Initialize dicts.
    args = dnnlib.EasyDict()
    args.G_args = dnnlib.EasyDict(func_name='training.networks.G_main')
    args.D_args = dnnlib.EasyDict(func_name='training.networks.D_main')
    args.G_opt_args = dnnlib.EasyDict(beta1=0.0, beta2=0.99)
    args.D_opt_args = dnnlib.EasyDict(beta1=0.0, beta2=0.99)
    args.loss_args = dnnlib.EasyDict(func_name='training.loss.stylegan2')
    args.augment_args = dnnlib.EasyDict(class_name='training.augment.AdaptiveAugment')

    # ---------------------------
    # General options: gpus, snap
    # ---------------------------

    if gpus is None:
        gpus = 1
    assert isinstance(gpus, int)
    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    if snap is None:
        snap = 50
    assert isinstance(snap, int)
    if snap < 1:
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = snap
    args.network_snapshot_ticks = snap

    # -----------------------------------
    # Training dataset: data, res, mirror
    # -----------------------------------

    assert data is not None
    assert isinstance(data, str)
    data_name = os.path.basename(os.path.abspath(data))
    if not os.path.isdir(data) or len(data_name) == 0:
        raise UserError('--data must point to a directory containing *.tfrecords')
    desc = data_name

    with tf.Graph().as_default(), tflib.create_session().as_default(): # pylint: disable=not-context-manager
        args.train_dataset_args = dnnlib.EasyDict(path=data, max_label_size='full')
        dataset_obj = dataset.load_dataset(**args.train_dataset_args) # try to load the data and see what comes out
        args.train_dataset_args.resolution = dataset_obj.shape[-1] # be explicit about resolution
        args.train_dataset_args.max_label_size = dataset_obj.label_size # be explicit about label size
        validation_set_available = dataset_obj.has_validation_set
        dataset_obj.close()
        dataset_obj = None

    if res is None:
        res = args.train_dataset_args.resolution
    else:
        assert isinstance(res, int)
        if not (res >= 4 and res & (res - 1) == 0):
            raise UserError('--res must be a power of two and at least 4')
        if res > args.train_dataset_args.resolution:
            raise UserError(f'--res cannot exceed maximum available resolution in the dataset ({args.train_dataset_args.resolution})')
        desc += f'-res{res:d}'
    args.train_dataset_args.resolution = res

    if mirror is None:
        mirror = False
    else:
        assert isinstance(mirror, bool)
        if mirror:
            desc += '-mirror'
    args.train_dataset_args.mirror_augment = mirror

    # ----------------------------
    # Metrics: metrics, metricdata
    # ----------------------------

    if metrics is None:
        metrics = ['fid50k_full']
    assert isinstance(metrics, list)
    assert all(isinstance(metric, str) for metric in metrics)

    args.metric_arg_list = []
    for metric in metrics:
        if metric not in metric_defaults.metric_defaults:
            raise UserError('\n'.join(['--metrics can only contain the following values:', 'none'] + list(metric_defaults.metric_defaults.keys())))
        args.metric_arg_list.append(metric_defaults.metric_defaults[metric])

    args.metric_dataset_args = dnnlib.EasyDict(args.train_dataset_args)
    if metricdata is not None:
        assert isinstance(metricdata, str)
        if not os.path.isdir(metricdata):
            raise UserError('--metricdata must point to a directory containing *.tfrecords')
        args.metric_dataset_args.path = metricdata

    # -----------------------------
    # Base config: cfg, gamma, kimg
    # -----------------------------

    if cfg is None:
        cfg = 'auto'
    assert isinstance(cfg, str)
    desc += f'-{cfg}'

    cfg_specs = {
        'auto':          dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # populated dynamically based on 'gpus' and 'res'
        'stylegan2':     dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # uses mixed-precision, unlike original StyleGAN2
        'paper256':      dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        'paper512':      dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper1024':     dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
        'cifar':         dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=0.5, lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
        'cifarbaseline': dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=0.5, lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=8),
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        desc += f'{gpus:d}'
        spec.ref_gpus = gpus
        spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
        spec.mbstd = min(spec.mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32

    args.total_kimg = spec.kimg
    args.minibatch_size = spec.mb
    args.minibatch_gpu = spec.mb // spec.ref_gpus
    args.D_args.mbstd_group_size = spec.mbstd
    args.G_args.fmap_base = args.D_args.fmap_base = int(spec.fmaps * 16384)
    args.G_args.fmap_max = args.D_args.fmap_max = 512
    args.G_opt_args.learning_rate = args.D_opt_args.learning_rate = spec.lrate
    args.loss_args.r1_gamma = spec.gamma
    args.G_smoothing_kimg = spec.ema
    args.G_smoothing_rampup = spec.ramp
    args.G_args.mapping_layers = spec.map
    args.G_args.num_fp16_res = args.D_args.num_fp16_res = 4 # enable mixed-precision training
    args.G_args.conv_clamp = args.D_args.conv_clamp = 256 # clamp activations to avoid float16 overflow

    if cfg == 'cifar':
        args.loss_args.pl_weight = 0 # disable path length regularization
        args.G_args.style_mixing_prob = None # disable style mixing
        args.D_args.architecture = 'orig' # disable residual skip connections

    if gamma is not None:
        assert isinstance(gamma, float)
        if not gamma >= 0:
            raise UserError('--gamma must be non-negative')
        desc += f'-gamma{gamma:g}'
        args.loss_args.r1_gamma = gamma

    if kimg is not None:
        assert isinstance(kimg, int)
        if not kimg >= 1:
            raise UserError('--kimg must be at least 1')
        desc += f'-kimg{kimg:d}'
        args.total_kimg = kimg

    # ---------------------------------------------------
    # Discriminator augmentation: aug, p, target, augpipe
    # ---------------------------------------------------

    if aug is None:
        aug = 'ada'
    else:
        assert isinstance(aug, str)
        desc += f'-{aug}'

    if aug == 'ada':
        args.augment_args.tune_heuristic = 'rt'
        args.augment_args.tune_target = 0.6

    elif aug == 'noaug':
        pass

    elif aug == 'fixed':
        if p is None:
            raise UserError(f'--aug={aug} requires specifying --p')

    elif aug == 'adarv':
        if not validation_set_available:
            raise UserError(f'--aug={aug} requires separate validation set; please see "python dataset_tool.py pack -h"')
        args.augment_args.tune_heuristic = 'rv'
        args.augment_args.tune_target = 0.5

    else:
        raise UserError(f'--aug={aug} not supported')

    if p is not None:
        assert isinstance(p, float)
        if aug != 'fixed':
            raise UserError('--p can only be specified with --aug=fixed')
        if not 0 <= p <= 1:
            raise UserError('--p must be between 0 and 1')
        desc += f'-p{p:g}'
        args.augment_args.initial_strength = p

    if target is not None:
        assert isinstance(target, float)
        if aug not in ['ada', 'adarv']:
            raise UserError('--target can only be specified with --aug=ada or --aug=adarv')
        if not 0 <= target <= 1:
            raise UserError('--target must be between 0 and 1')
        desc += f'-target{target:g}'
        args.augment_args.tune_target = target

    assert augpipe is None or isinstance(augpipe, str)
    if augpipe is None:
        augpipe = 'bgc'
    else:
        if aug == 'noaug':
            raise UserError('--augpipe cannot be specified with --aug=noaug')
        desc += f'-{augpipe}'

    augpipe_specs = {
        'blit':     dict(xflip=1, rotate90=1, xint=1),
        'geom':     dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':    dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter':   dict(imgfilter=1),
        'noise':    dict(noise=1),
        'cutout':   dict(cutout=1),
        'bg':       dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':      dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }

    assert augpipe in augpipe_specs
    if aug != 'noaug':
        args.augment_args.apply_func = 'training.augment.augment_pipeline'
        args.augment_args.apply_args = augpipe_specs[augpipe]

    # ---------------------------------
    # Comparison methods: cmethod, dcap
    # ---------------------------------

    assert cmethod is None or isinstance(cmethod, str)
    if cmethod is None:
        cmethod = 'nocmethod'
    else:
        desc += f'-{cmethod}'

    if cmethod == 'nocmethod':
        pass

    elif cmethod == 'bcr':
        args.loss_args.func_name = 'training.loss.cmethods'
        args.loss_args.bcr_real_weight = 10
        args.loss_args.bcr_fake_weight = 10
        args.loss_args.bcr_augment = dnnlib.EasyDict(func_name='training.augment.augment_pipeline', xint=1, xint_max=1/32)

    elif cmethod == 'zcr':
        args.loss_args.func_name = 'training.loss.cmethods'
        args.loss_args.zcr_gen_weight = 0.02
        args.loss_args.zcr_dis_weight = 0.2
        args.G_args.num_fp16_res = args.D_args.num_fp16_res = 0 # disable mixed-precision training
        args.G_args.conv_clamp = args.D_args.conv_clamp = None

    elif cmethod == 'pagan':
        if aug != 'noaug':
            raise UserError(f'--cmethod={cmethod} is not compatible with discriminator augmentation; please specify --aug=noaug')
        args.D_args.use_pagan = True
        args.augment_args.tune_heuristic = 'rt' # enable ada heuristic
        args.augment_args.pop('apply_func', None) # disable discriminator augmentation
        args.augment_args.pop('apply_args', None)
        args.augment_args.tune_target = 0.95

    elif cmethod == 'wgangp':
        if aug != 'noaug':
            raise UserError(f'--cmethod={cmethod} is not compatible with discriminator augmentation; please specify --aug=noaug')
        if gamma is not None:
            raise UserError(f'--cmethod={cmethod} is not compatible with --gamma')
        args.loss_args = dnnlib.EasyDict(func_name='training.loss.wgangp')
        args.G_opt_args.learning_rate = args.D_opt_args.learning_rate = 0.001
        args.G_args.num_fp16_res = args.D_args.num_fp16_res = 0 # disable mixed-precision training
        args.G_args.conv_clamp = args.D_args.conv_clamp = None
        args.lazy_regularization = False

    elif cmethod == 'auxrot':
        if args.train_dataset_args.max_label_size > 0:
            raise UserError(f'--cmethod={cmethod} is not compatible with label conditioning; please specify a dataset without labels')
        args.loss_args.func_name = 'training.loss.cmethods'
        args.loss_args.auxrot_alpha = 10
        args.loss_args.auxrot_beta = 5
        args.D_args.score_max = 5 # prepare D to output 5 scalars per image instead of just 1

    elif cmethod == 'spectralnorm':
        args.D_args.use_spectral_norm = True

    elif cmethod == 'shallowmap':
        if args.G_args.mapping_layers == 2:
            raise UserError(f'--cmethod={cmethod} is a no-op for --cfg={cfg}')
        args.G_args.mapping_layers = 2

    elif cmethod == 'adropout':
        if aug != 'noaug':
            raise UserError(f'--cmethod={cmethod} is not compatible with discriminator augmentation; please specify --aug=noaug')
        args.D_args.adaptive_dropout = 1
        args.augment_args.tune_heuristic = 'rt' # enable ada heuristic
        args.augment_args.pop('apply_func', None) # disable discriminator augmentation
        args.augment_args.pop('apply_args', None)
        args.augment_args.tune_target = 0.6

    else:
        raise UserError(f'--cmethod={cmethod} not supported')

    if dcap is not None:
        assert isinstance(dcap, float)
        if not dcap > 0:
            raise UserError('--dcap must be positive')
        desc += f'-dcap{dcap:g}'
        args.D_args.fmap_base = max(int(args.D_args.fmap_base * dcap), 1)
        args.D_args.fmap_max = max(int(args.D_args.fmap_max * dcap), 1)

    # ----------------------------------
    # Transfer learning: resume, freezed
    # ----------------------------------

    resume_specs = {
        'ffhq256':      'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
        'ffhq512':      'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
        'ffhq1024':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
        'celebahq256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
        'lsundog256':   'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
    }

    assert resume is None or isinstance(resume, str)
    if resume is None:
        resume = 'noresume'
    elif resume == 'noresume':
        desc += '-noresume'
    elif resume in resume_specs:
        desc += f'-resume{resume}'
        args.resume_pkl = resume_specs[resume] # predefined url
    else:
        desc += '-resumecustom'
        args.resume_pkl = resume # custom path or url

    if resume != 'noresume':
        args.augment_args.tune_kimg = 100 # make ADA react faster at the beginning
        args.G_smoothing_rampup = None # disable EMA rampup

    if freezed is not None:
        assert isinstance(freezed, int)
        if not freezed >= 0:
            raise UserError('--freezed must be non-negative')
        desc += f'-freezed{freezed:d}'
        args.D_args.freeze_layers = freezed

    return desc, args


# ### Retrive the pickle file path of the lastest snap

# In[5]:


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


# In[6]:


run_desc, training_options = setup_training_options(
    gpus       = 2,
    snap       = 1,
    data       = '/work/chaselab/malyetama/datasets/102flowers_custom',
    resume     = "/work/chaselab/malyetama/102flowers_training-runs/00026-102flowers_custom-auto2-resumecustom/network-snapshot-000061.pkl"
)


# ## Run training

# In[7]:


#----------------------------------------------------------------------------

def run_training(outdir, seed, dry_run, run_desc, training_options):
    # Setup training options.
    tflib.init_tf({'rnd.np_random_seed': seed})
#     run_desc, training_options = setup_training_options(**hyperparam_options)

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    training_options.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(training_options.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(training_options, indent=2))
    print()
    print(f'Output directory:  {training_options.run_dir}')
    print(f'Training data:     {training_options.train_dataset_args.path}')
    print(f'Training length:   {training_options.total_kimg} kimg')
    print(f'Resolution:        {training_options.train_dataset_args.resolution}')
    print(f'Number of GPUs:    {training_options.num_gpus}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Kick off training.
    print('Creating output directory...')
    os.makedirs(training_options.run_dir)
    with open(os.path.join(training_options.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(training_options, f, indent=2)
    with dnnlib.util.Logger(os.path.join(training_options.run_dir, 'log.txt')):
        training_loop.training_loop(**training_options)


# In[ ]:


# ! nvidia-smi


# In[ ]:


# dry-run (no training)
run_training(outdir="/work/chaselab/malyetama/102flowers_training-runs", seed=1000,
             dry_run=True, run_desc=run_desc, training_options=training_options)


# In[ ]:


run_training(outdir="/work/chaselab/malyetama/102flowers_training-runs", seed=1000,
             dry_run=False, run_desc=run_desc, training_options=training_options)


# In[ ]:




