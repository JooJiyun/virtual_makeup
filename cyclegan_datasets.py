#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""Contains the standard train/test splits for the cyclegan data."""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    'lipstick_data': 630,
    'lipstick_data_test': 630
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'lipstick_data': '.png',
    'lipstick_data_test': '.png',
}

"""The path to the input csv file."""
PATH_TO_CSV = {
    'lipstick_data': './CycleGAN_TensorFlow/input/train.csv',
    'lipstick_data_test': './CycleGAN_TensorFlow/input/test.csv',
}

