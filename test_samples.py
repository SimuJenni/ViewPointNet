import tensorflow as tf
from Preprocessor import Preprocessor
from scipy import misc
import numpy as np
from utils import montage
from datasets.ObjectNet3D import ObjectNet3D

slim = tf.contrib.slim

imgs_tr = [None for i in range(16)]

data = ObjectNet3D('car')
preprocessor = Preprocessor(target_shape=[96, 96, 3])

with tf.Session() as sess:
    for i in range(16):
        provider = slim.dataset_data_provider.DatasetDataProvider(data.get_trainset(), num_readers=1)
        [img, im_size, box, vp] = provider.get(['image', 'im_size', 'bbox', 'viewpoint'])

        # Preprocess data
        img = preprocessor.process_train(img, box, im_size)
        imgs_tr[i] = img.eval()

mont = montage(imgs_tr)
misc.toimage(mont).save('test.png')