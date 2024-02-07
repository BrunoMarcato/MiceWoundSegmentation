"""# Import dependencies and useful functions"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, jaccard_score
from sklearn.model_selection import KFold

from keras.models import Model

import os

# -----------------------------------------------------------------------------------------------

IMG_DIR = "data/images"
MASK_DIR = "data/masks"
FOLDS = 5
NUM_TREES = 500
LOAD_MODEL = False

IMG_HEIGHT = 256
IMG_WIDTH = 256

FEATURE_MODEL = f"RandomForest/models/vgg16_{IMG_HEIGHT}x{IMG_WIDTH}.sav"

# -----------------------------------------------------------------------------------------------

# Load images and masks

images_names = list(os.listdir(IMG_DIR))
images_names.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

images = []
for image in images_names:
    print(image)
    img = cv2.imread(os.path.join(IMG_DIR, image), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    images.append(img / 255.0)

images = np.array(images)

masks_names = list(os.listdir(MASK_DIR))
masks_names.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

masks = []
for mask in masks_names:
    print(mask)
    mask = cv2.imread(os.path.join(MASK_DIR, mask), 0)
    mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
    mask[mask != 0.0] = 1.0
    masks.append(mask)

masks = np.array(masks)
masks = np.expand_dims(masks, axis=3)


# Load vgg model (feature extractor)

VGG16_model = pickle.load(open(FEATURE_MODEL, 'rb'))
feature_extractor_model = Model(inputs=VGG16_model.input, outputs=VGG16_model.get_layer('block1_conv2').output)

# To save f1 and jaccard scores from each fold
scores = []

# K Fold cross validation
kf = KFold(n_splits=FOLDS, shuffle=True)
for fold, (train_idx, test_idx) in enumerate(kf.split(images)):
    print(f"Fold {fold + 1}")
    print("-------")

    #extract features
    features_train = feature_extractor_model.predict(images[train_idx])
    features_train = features_train.reshape(-1, features_train.shape[3])

    # create dataset for training
    X_train = pd.DataFrame(features_train)
    y_train = masks[train_idx].reshape(-1)

    """# Train RF"""

    filename = f'RandomForest/models/RF_run{fold+1}.sav'

    if LOAD_MODEL:
        print('\n\nLOADING MODEL')
        model = pickle.load(open(filename, 'rb'))
    else:
        print('\nTRAINING')

        model = RandomForestClassifier(n_estimators = NUM_TREES, random_state = 42, n_jobs = -1)

        model.fit(X_train, y_train)

        #saving the model

        pickle.dump(model, open(filename, 'wb'))

    """# Testing"""

    print('TESTING')

    scores_run = []
    filenames = np.array(images_names)[test_idx]
    labels = masks[test_idx]
    for i, test_img in enumerate(images[test_idx]):
        test_img = np.expand_dims(test_img, axis=0)

        X_feature = feature_extractor_model.predict(test_img)
        X_feature = X_feature.reshape(-1, X_feature.shape[3])

        prediction = model.predict(X_feature)
        prediction[prediction != 0.0] = 1.0

        prediction_image = prediction.reshape(labels[i].shape).squeeze()

        plt.imsave(f'RandomForest/prediction_images500/RF_run{fold+1}_{filenames[i]}', prediction_image, cmap='gray')

        f1score = f1_score(labels[i].reshape(-1), prediction)
        jaccardscore = jaccard_score(labels[i].reshape(-1), prediction)
        scores_run.append((f1score, jaccardscore, int(filenames[i].split('.')[0])))

    scores += scores_run
    scores_run = np.array(scores_run)

    print(scores_run)

    print(f'\nF1 SCORE\nMédia: {scores_run[:,0].mean()} / Desvio padrao: {scores_run[:,0].std()} \
        \nJACCARD SCORE\nMédia: {scores_run[:,1].mean()} / Desvio padrao: {scores_run[:,1].std()} \
        \nIMAGES: {scores_run[:,2]}')
    
#print results
scores = np.array(scores)
print(f'\n\nF1 SCORE\nMédia: {scores[:,0].mean()} / Desvio padrao: {scores[:,0].std()} \
      \n\nJACCARD SCORE\nMédia: {scores[:,1].mean()} / Desvio padrao: {scores[:,1].std()}')

#create results dataframe
results = pd.DataFrame(scores, columns=['F1 Score', 'Jaccard Score', 'Image Number']).astype({'Image Number': 'int32'})

#sorting by Image Number
results = results.sort_values('Image Number')

#save dataframe as csv file
results.to_csv(f'RandomForest/results/results.csv', index=False)

