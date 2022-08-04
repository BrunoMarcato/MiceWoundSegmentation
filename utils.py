'''
    Some utils function to handle tfrecords objects
'''

import numpy as np
import tensorflow as tf
from PIL import Image

def get_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

def get_label(path):
    label = Image.open(path)
    return np.array(label)[..., None]


'''Produces a tf example from an image and a label'''
def record_example(img, label):
    feature = {
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[img.shape[0]])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[img.shape[1]])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tobytes()])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()]))
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

def parse_decode_record(record):
    features = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string)
    }

    parsed_record = tf.io.parse_single_example(record, features)

    height = parsed_record['height']
    width = parsed_record['width']

    image = tf.io.decode_raw(parsed_record['image'], out_type=tf.uint8)
    label = tf.io.decode_raw(parsed_record['label'], out_type=tf.uint8)

    image = np.reshape(image, (height, width, 3))
    label = np.reshape(label, (height, width, 1))

    return image, label
