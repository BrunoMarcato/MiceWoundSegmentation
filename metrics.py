import numpy as np
import cv2
from sklearn.metrics import jaccard_score, f1_score
import pandas as pd

import os

NUM_RUNS = 5
algs = ['RandomForest', 'UNet']

for alg in algs:
    predictions = os.listdir(f'{alg}/prediction_images')

    df_f1_score = pd.DataFrame()
    df_jaccard_score = pd.DataFrame()

    for current_run in range(1, NUM_RUNS+1):

        f1_scores = []
        jaccard_scores = []
        img_names = []
        for pred in predictions:
            img = pred.split('.')[0]
            _, run, num_img = img.split('_')

            if int(run[3]) == current_run:
                print(run, end='/')
                print(pred)
                mask = cv2.imread(f'data/all_data/{num_img}_mask.png', 0)
                pred = cv2.imread(f'{alg}/prediction_images/{pred}', 0)

                pred = cv2.resize(pred, mask.shape[::-1])

                pred = np.array(pred)
                mask = np.array(mask)

                mask[mask != 0.0] = 1.0
                pred[pred != 0.0] = 1.0

                pred = pred.reshape(-1)
                mask = mask.reshape(-1)

                f1 = f1_score(mask, pred)
                jaccard = jaccard_score(mask, pred)

                f1_scores.append(f1)
                jaccard_scores.append(jaccard)

                print(f'f1 = {f1}')
                print(f'jaccard = {jaccard}')

                img_names.append(num_img)


        f1_scores = np.array(f1_scores)
        jaccard_scores = np.array(jaccard_scores)
        img_names = np.array(img_names)

        df_f1_score[f'score_run{current_run}'] = pd.Series(f1_scores)
        df_f1_score[f'image_run{current_run}'] = pd.Series(img_names)

        df_jaccard_score[f'score_run{current_run}'] = pd.Series(jaccard_scores)
        df_jaccard_score[f'image_run{current_run}'] = pd.Series(img_names)

    df_f1_score.to_csv(f'results/{alg}_f1_scores.csv', index=False, encoding='utf-8')
    df_jaccard_score.to_csv(f'results/{alg}_jaccard_scores.csv', index=False, encoding='utf-8')