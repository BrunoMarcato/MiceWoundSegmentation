'''Configure hyper params and train the model'''

from sklearn.metrics import f1_score, jaccard_score
from sklearn.model_selection import KFold

import numpy as np
import albumentations as A
from tqdm import tqdm
import pandas as pd

from model import UNet
from dataset import RatsDataset
from augment import offline_augment

from utils import save_model

import torchvision
from torchvision import transforms
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

# -----------------------------------------------------------------------------------------------

# Hyper params and others

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
FOLDS = 5
NUM_EPOCHS = 50
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMG_DIR = "data/images"
MASK_DIR = "data/masks"

# -----------------------------------------------------------------------------------------------


def train(model, device, train_loader, optimizer, epoch, scaler):
    '''
    Function to train the model.

    Parameters:
    - model (torch.nn.Module): The neural network model to be trained.
    - device (torch.device): The device where the data will be loaded.
    - train_loader (DataLoader): The DataLoader for training data.
    - optimizer (torch.optim.Optimizer): The optimizer used for updating the model parameters.
    - epoch (int): The current epoch number.
    - scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed precision training.

    Returns:
    - float: The total training loss for the epoch.

    This function trains the specified neural network model for one epoch using the given training DataLoader,
    optimizer, and gradient scaler. It computes and updates the model parameters based on the provided data,
    calculates the training loss, and returns the total training loss for the epoch.
    '''
        
    loop = tqdm(train_loader)
    train_loss = 0.0

    model.train()
    for _, (data, target, _) in enumerate(loop):
        data, target = data.to(device), target.float().unsqueeze(1).to(device)

        #forward pass
        with torch.cuda.amp.autocast():
            output = model(data)
            loss = loss_func(output, target)

        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        loop.set_postfix(loss = loss.item())
       
    tb.add_scalar('Loss', train_loss, epoch)
    return train_loss

# -----------------------------------------------------------------------------------------------

dataset = RatsDataset(img_dir = IMG_DIR, mask_dir = MASK_DIR)

transforms = [
    A.Rotate(limit=45, p=1.0),
    A.HorizontalFlip(p=1.0),
    A.RandomBrightnessContrast(p=1.0),
    A.ShiftScaleRotate(p=1.0),
]

# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the k-fold cross validation
kf = KFold(n_splits=FOLDS, shuffle=True)

# To save f1 and jaccard scores from each fold
scores = []

# Loop through each fold
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}")
    print("-------")

    tb = SummaryWriter()

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    train_dataset = offline_augment(train_dataset, transforms=transforms)

    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    test_dataset = offline_augment(test_dataset, transforms=transforms, test=True)

    # Define the data loaders for the current fold
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
   
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
    )

    # Initialize the model and optimizer
    model = UNet(in_channels = 3, out_channels = 1).to(device)
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model on the current fold
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        print(f'EPOCH {epoch+1}')
        train_loss = train(model, device, train_loader, optimizer, epoch, scaler)

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        scores_run = []
        for x, y, fname in test_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            #saving prediction as image
            torchvision.utils.save_image(preds, f"{'UNet/pred_train_remake'}/{model.name}_run{fold+1}_{''.join(fname)}")

            f1score = f1_score(y.cpu().numpy().flatten(), preds.cpu().numpy().flatten())
            jaccardscore = jaccard_score(y.cpu().numpy().flatten(), preds.cpu().numpy().flatten())

            scores_run.append((f1score, jaccardscore, int(fname[0].split('.')[0])))
   
    scores += scores_run

    scores_run = np.array(scores_run)

    print(f'\nF1 SCORE\nMédia: {scores_run[:,0].mean()} / Desvio padrao: {scores_run[:,0].std()} \
        \nJACCARD SCORE\nMédia: {scores_run[:,1].mean()} / Desvio padrao: {scores_run[:,1].std()} \
        \nIMAGES: {scores_run[:,2]}')

    save_model(model, optimizer, NUM_EPOCHS, train_loss, filename = f'UNet/models/{model.name}_run{fold+1}.pt')

    tb.close()

#print results
scores = np.array(scores)
print(f'\n\nF1 SCORE\nMédia: {scores[:,0].mean()} / Desvio padrao: {scores[:,0].std()} \
      \n\nJACCARD SCORE\nMédia: {scores[:,1].mean()} / Desvio padrao: {scores[:,1].std()}')

#create results dataframe
results = pd.DataFrame(scores, columns=['F1 Score', 'Jaccard Score', 'Image Number']).astype({'Image Number': 'int32'}) #pylint: disable='line-too-long'

#sorting by Image Number
results = results.sort_values('Image Number')

#save dataframe as csv file
results.to_csv('UNet/results/results.csv', index=False)
