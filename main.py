#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

#importing built-in python packages
import cv2
import matplotlib.pyplot as plt
from sklearn import model_selection
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model

# importing local files=
from data import split_dataset
from augmentation import augment_data

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

def load_data(DATASET_PATH, augment=True):
    print("Loading data")
    X_train, X_test, y_train, y_test = split_dataset(DATASET_PATH, holdout=0.8)

    X_train_images = []
    for image in X_train:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_train_images.append(img)

    X_test_images = []
    for image in X_test:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        X_test_images.append(img)

    y_train_images = []
    for image in y_train:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y_train_images.append(img)

    y_test_images = []
    for image in y_test:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        y_test_images.append(img)

    if augment:
        print(f"Original Images: {len(X_train_images)} - Original Masks: {len(y_train_images)}")
        augment_data(X_train, y_train, X_train_images, y_train_images, save_path='rats_data')
        print(f"\nAugmented Images: {len(X_train_images)} - Augmented Masks: {len(y_train_images)}")

        print(f"\n\nOriginal Images: {len(X_test_images)} - Original Masks: {len(y_test_images)}")
        augment_data(X_test, y_test, X_test_images, y_test_images, save_path='rats_data')
        print(f"\nAugmented Images: {len(X_test_images)} - Augmented Masks: {len(y_test_images)}")

    return ([X_train_images, y_train_images, X_test_images, y_test_images])

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    (X_train, y_train, X_test, y_test) = load_data('rats_data')