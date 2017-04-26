import tensorflow as tf
import os
from Preprocessor import Preprocessor
from utils import montage_tf
from datasets.ObjectNet3D import ObjectNet3D
from constants import LOG_DIR

slim = tf.contrib.slim

bs = 100
data = ObjectNet3D('car')
preprocessor = Preprocessor(target_shape=[96, 96, 3])
LOG_PATH = os.path.join(LOG_DIR, 'test_preprocess/')


with tf.Session() as sess:
    global_step = slim.create_global_step()

    provider = slim.dataset_data_provider.DatasetDataProvider(data.get_trainset(), num_readers=1)
    [img, im_size, box, vp] = provider.get(['image', 'im_size', 'bbox', 'viewpoint'])

    # Preprocess data
    img_processed = preprocessor.process_train(img, box, im_size)
    imgs, imgs_processed = tf.train.batch([img, img_processed], batch_size=bs, num_threads=1)

    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'MSE-dummy': slim.metrics.streaming_mean_squared_error(imgs_processed, imgs_processed),
    })

    summary_ops = []
    summary_ops.append(tf.image_summary('images/orig', montage_tf(imgs, 10, 10), max_images=1))
    summary_ops.append(tf.image_summary('images/processed', montage_tf(imgs_processed, 10, 10), max_images=1))

    slim.evaluation.evaluation_loop('', '', LOG_PATH,
                                                num_evals=1,
                                                max_number_of_evaluations=1,
                                                eval_op=names_to_updates.values(),
                                                summary_op=tf.merge_summary(summary_ops))