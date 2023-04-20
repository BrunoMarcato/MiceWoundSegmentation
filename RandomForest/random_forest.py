"""# Import dependencies and useful functions"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from utils.utils import split_dataset

from keras.models import Model

import os

# -----------------------------------------------------------------------------------------------

ROOT_DIR = "data/all_data"
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
TEST_IMG_DIR = "data/test_images/"
TEST_MASK_DIR = "data/test_masks/"

IMG_HEIGHT = 256
IMG_WIDTH = 256

FEATURE_MODEL = f"RandomForest/models/vgg16_{IMG_HEIGHT}x{IMG_WIDTH}.sav"

TEST_SIZE = 0.3
VAL_SIZE = 0.2
NUM_JOBS = 7
LOAD_MODEL = False
NUM_RUNS = 5
SEEDS = range(NUM_RUNS)

if __name__ == '__main__':

  SEEDS = iter(SEEDS)

  df = pd.DataFrame()
  for run in range(NUM_RUNS):
    print(f'RUN {run+1}\n\n')

    split_dataset(ROOT_DIR, 
      TRAIN_IMG_DIR,
      TRAIN_MASK_DIR,
      VAL_IMG_DIR,
      VAL_MASK_DIR,
      TEST_IMG_DIR,
      TEST_MASK_DIR,
      test_size = TEST_SIZE,
      val_size = VAL_SIZE,
      seed = next(SEEDS)
    )

    """# Pre-processing and Feature Extraction"""

    print('PRE-PROCESSING AND FEATURE EXTRACTION')

    images_names = list(os.listdir(TRAIN_IMG_DIR))
    images_names.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

    train_images = []
    for image in images_names:
      print(image)
      img = cv2.imread(os.path.join(TRAIN_IMG_DIR, image), cv2.IMREAD_COLOR)
      img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

      train_images.append(img)

    train_images = np.array(train_images)

    masks_names = list(os.listdir(TRAIN_MASK_DIR))
    masks_names.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

    train_masks = []
    for mask in masks_names:
      print(mask)
      mask = cv2.imread(os.path.join(TRAIN_MASK_DIR, mask), 0)
      mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
      mask[mask != 0.0] = 1.0
      train_masks.append(mask)

    train_masks = np.array(train_masks)
    train_masks = np.expand_dims(train_masks, axis=3)

    VGG16_model = pickle.load(open(FEATURE_MODEL, 'rb'))

    feature_extractor_model = Model(inputs=VGG16_model.input, outputs=VGG16_model.get_layer('block1_conv2').output)

    X = feature_extractor_model.predict(train_images)

    #visualize features
    # square = 8
    # ix = 1
    # for _ in range(square):
    #   for _ in range(square):
    #     ax = plt.subplot(square, square, ix)
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     plt.imshow(X[1, :, :, ix-1], cmap='gray')
    #     ix += 1
    # plt.show()

    X = X.reshape(-1, X.shape[3])

    Y = train_masks.reshape(-1)

    dataset = pd.DataFrame(X)

    dataset['Label'] = Y

    X_train = dataset.drop(labels = ['Label'], axis = 1)
    y_train = dataset['Label']

    """# Train RF"""

    filename = f'RandomForest/models/RF_run{run+1}.sav'

    if LOAD_MODEL:
      print('\n\nLOADING MODEL')
      model = pickle.load(open(filename, 'rb'))
    else:
      print('\n\nTRAINING RF')

      model = RandomForestClassifier(n_estimators = 50, criterion = "gini", random_state = 42, n_jobs = NUM_JOBS)

      model.fit(X_train, y_train)

      #saving the model

      pickle.dump(model, open(filename, 'wb'))

    """# Testing"""

    print('TESTING')

    f1_scores = []
    filenames = []
    for file in os.listdir(TEST_IMG_DIR):
      test_img = cv2.imread(os.path.join(TEST_IMG_DIR, file), cv2.IMREAD_COLOR)
      test_img = cv2.resize(test_img, (IMG_HEIGHT, IMG_WIDTH))
      test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
      test_img = np.expand_dims(test_img, axis=0)

      X_feature = feature_extractor_model.predict(test_img)
      X_feature = X_feature.reshape(-1, X_feature.shape[3])

      prediction = model.predict(X_feature)
      prediction[prediction != 0.0] = 1.0

      prediction_image = prediction.reshape(mask.shape)

      mask = cv2.imread(os.path.join(TEST_MASK_DIR, file.split('.')[0] + '_mask.png'), 0)
      mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
      mask[mask != 0.0] = 1.0

      plt.imsave(f'RandomForest/prediction_images/RF_run{run+1}_{file}', prediction_image, cmap='gray')

      f1 = f1_score(mask.reshape(-1), prediction)
      print(f1)
      f1_scores.append(f1)
      filenames.append(file)

    f1_scores = np.array(f1_scores)
    filenames = np.array(filenames)

    df[f'RF_run{run+1}'] = f1_scores
    df[f'filename_run{run+1}'] = filenames

  df.to_csv('boxplots/RandomForest_f1_scores.csv', index=False, encoding='utf-8')