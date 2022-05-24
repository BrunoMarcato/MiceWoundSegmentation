import os
import numpy as np
import tensorflow as tf
from shutil import copy2

def split_dataset(BASE_PATH='split_data', DATASET_PATH='rats_data', len_train=50, len_val=26):
    os.makedirs(BASE_PATH, exist_ok=True)

    #creating train/val folders
    train_dir = os.path.join(BASE_PATH, 'train')

    val_dir = os.path.join(BASE_PATH, 'validation')

    dirs = ['images', 'labels']
    for dir in dirs:
        os.makedirs(os.path.join(train_dir, dir))
        os.makedirs(os.path.join(val_dir, dir))
    del dirs

    #copying files from original folder to dataset folder
    images = []
    labels = []

    treatments = os.listdir(DATASET_PATH)
    for treatment in treatments:
        if treatment == 'CIC' or treatment == 'PDX':
            days = os.listdir(os.path.join(DATASET_PATH, treatment))
            for day in days:
                animals = os.listdir(os.path.join(DATASET_PATH, treatment, day))
                for animal in animals:
                    path = os.path.join(DATASET_PATH, treatment, day, animal)
                    images.append(os.path.join(path, animal+'.png'))
                    labels.append(os.path.join(path, animal+'_label.png'))
        else:
            doses = os.listdir(os.path.join(DATASET_PATH, treatment))
            for dose in doses:
                days = os.listdir(os.path.join(DATASET_PATH, treatment, dose))
                for day in days:
                    animals = os.listdir(os.path.join(DATASET_PATH, treatment, dose, day))
                    for animal in animals:
                        path = os.path.join(DATASET_PATH, treatment, dose, day, animal)
                        images.append(os.path.join(path, animal+'.png'))
                        labels.append(os.path.join(path, animal+'_label.png'))

    if len(images) != len_train + len_val:
        size = len_train + len_val
        len_train = size*3//4
        len_val = size//4
        print(f'Invalid size for train/val. New train: {len_train}. New Val: {len_val}')
        del size


    #distributing images and labels at train/val datasets
    for image, label in zip(images[:len_train], labels[:len_train]):
        copy2(image, os.path.join(BASE_PATH, 'train', 'images'))
        copy2(label, os.path.join(BASE_PATH, 'train', 'labels'))

    for image, label in zip(images[len_train:len_train+len_val], labels[len_train:len_train+len_val]):
        copy2(image, os.path.join(BASE_PATH, 'validation', 'images'))
        copy2(label, os.path.join(BASE_PATH, 'validation', 'labels'))