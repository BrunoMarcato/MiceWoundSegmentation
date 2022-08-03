import os
from sklearn.model_selection import train_test_split

def split_dataset(DATASET_PATH, holdout=0.8):
    images = []
    labels = []

    for dirpath, dirnames, filenames in os.walk(DATASET_PATH):
            for filename in filenames:
                dir = os.path.join(dirpath, filename).split('\\')[1]
                if dir == 'images':
                    images.append(os.path.join(dirpath, filename))
                else:
                    labels.append(os.path.join(dirpath, filename))

    return train_test_split(images, labels, test_size=1-holdout)