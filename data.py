import os
import numpy as np
from sklearn.model_selection import train_test_split

#Split dataset using sklearn.model_selection.train_test_split

def split_dataset(DATASET_PATH='rats_data', holdout=0.8):
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

    return train_test_split(images, labels, test_size=1-holdout, random_state=np.random.randint(0,1000000))

#Split dataset without using sklearn.model_selection.train_test_split

def split_dataset2(DATASET_PATH='rats_data', holdout=80):
    pair = []

    treatments = os.listdir(DATASET_PATH)
    for treatment in treatments:
        if treatment == 'CIC' or treatment == 'PDX':
            days = os.listdir(os.path.join(DATASET_PATH, treatment))
            for day in days:
                animals = os.listdir(os.path.join(DATASET_PATH, treatment, day))
                for animal in animals:
                    path = os.path.join(DATASET_PATH, treatment, day, animal)
                    tupla = os.path.join(path, animal+'.png'), os.path.join(path, animal+'_label.png')
                    pair.append(tupla)

        else:
            doses = os.listdir(os.path.join(DATASET_PATH, treatment))
            for dose in doses:
                days = os.listdir(os.path.join(DATASET_PATH, treatment, dose))
                for day in days:
                    animals = os.listdir(os.path.join(DATASET_PATH, treatment, dose, day))
                    for animal in animals:
                        path = os.path.join(DATASET_PATH, treatment, dose, day, animal)
                        tupla = os.path.join(path, animal+'.png'), os.path.join(path, animal+'_label.png')
                        pair.append(tupla)

    #shuffle for cross validation
    np.random.seed(np.random.randint(0, 1000000))
    np.random.shuffle(pair)
    
    train_dataset = []
    test_dataset = []

    for item in range(len(pair)):
        if item < len(pair)*holdout//100:
            train_dataset.append(pair[item])
        else:
            test_dataset.append(pair[item])

    del pair

    # #distributing images and labels at train/val datasets
    X_train, X_test, y_train, y_test = [], [], [], []
    for i in range(len(train_dataset)):
        X_train.append(train_dataset[i][0])
        y_train.append(train_dataset[i][1])

    for i in range(len(test_dataset)):
        X_test.append(test_dataset[i][0])
        y_test.append(test_dataset[i][1])

    del train_dataset, test_dataset

    return X_train, X_test, y_train, y_test