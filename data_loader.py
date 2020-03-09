#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import tensorflow as tf

import cyclegan_datasets
import model


def _load_samples(csv_name, image_type):
    filename_queue = tf.train.string_input_producer(
        [csv_name])

    reader = tf.TextLineReader()
    _, csv_filename = reader.read(filename_queue)

    record_defaults = [tf.constant([], dtype=tf.string),
                       tf.constant([], dtype=tf.string),
                       tf.constant([], dtype=tf.string)]

    filename_i, filename_j, filename_k = tf.decode_csv(
        csv_filename, record_defaults=record_defaults)

    file_contents_i = tf.read_file(filename_i)
    file_contents_j = tf.read_file(filename_j)
    file_contents_k = tf.read_file(filename_k)
    if image_type == '.jpg':
        image_decoded_A = tf.image.decode_jpeg(
            file_contents_i, channels=model.IMG_CHANNELS)
        image_decoded_B = tf.image.decode_jpeg(
            file_contents_j, channels=model.IMG_CHANNELS)
        image_decoded_ref = tf.image.decode_jpeg(
            file_contents_k, channels=model.IMG_CHANNELS)
    elif image_type == '.png':
        image_decoded_A = tf.image.decode_png(
            file_contents_i, channels=model.IMG_CHANNELS, dtype=tf.uint8)
        image_decoded_B = tf.image.decode_png(
            file_contents_j, channels=model.IMG_CHANNELS, dtype=tf.uint8)
        image_decoded_ref = tf.image.decode_png(
            file_contents_k, channels=model.IMG_CHANNELS, dtype=tf.uint8)

    return image_decoded_A, image_decoded_B, image_decoded_ref, filename_i, filename_j, filename_k


def load_data(dataset_name, image_size_before_crop, do_flipping=False):
    """

    :param dataset_name: The name of the dataset.
    :param image_size_before_crop: Resize to this size before random cropping.
    :param do_shuffle: Shuffle switch.
    :param do_flipping: Flip switch.
    :return:
    """
    if dataset_name not in cyclegan_datasets.DATASET_TO_SIZES:
        raise ValueError('split name %s was not recognized.'
                         % dataset_name)

    csv_name = cyclegan_datasets.PATH_TO_CSV[dataset_name]

    image_i, image_j, image_k, path_i, path_j, path_k = _load_samples(
        csv_name, cyclegan_datasets.DATASET_TO_IMAGETYPE[dataset_name])

    # Preprocessing:
    image_i = tf.image.resize_images(
        image_i, [model.IMG_HEIGHT, model.IMG_WIDTH])
    image_j = tf.image.resize_images(
        image_j, [model.IMG_HEIGHT, model.IMG_WIDTH])
    image_k = tf.image.resize_images(
        image_k, [model.IMG_HEIGHT, model.IMG_WIDTH])

    if do_flipping is True:
        image_i = tf.image.random_flip_left_right(image_i)
        image_j = tf.image.random_flip_left_right(image_j)
        image_k = tf.image.random_flip_left_right(image_k)

    
    image_i = tf.random_crop(
        image_i, [model.IMG_HEIGHT, model.IMG_WIDTH, 3])
    image_j = tf.random_crop(
        image_j, [model.IMG_HEIGHT, model.IMG_WIDTH, 3])
    image_k = tf.random_crop(
        image_k, [model.IMG_HEIGHT, model.IMG_WIDTH, 3])

    image_i = tf.subtract(tf.div(image_i, 127.5), 1)
    image_j = tf.subtract(tf.div(image_j, 127.5), 1)
    image_k = tf.subtract(tf.div(image_k, 127.5), 1)
    
    
    # Batch
    images_i, images_j, images_k = tf.train.batch([image_i, image_j, image_k], 1)
    
    inputs = {
        'images_i': image_i,
        'images_j': image_j,
        'images_k': image_k
    }

    paths = {
	'filename_i':path_i,
	'filename_j':path_j,
	'filename_k':path_k
    }        
    return inputs, paths

