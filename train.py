import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
from utils import (
   load_checkpoint,
   save_checkpoint,
   get_loaders,
   metrics,
   save_preds,
   plot_loss_curve,
   split_dataset
)

# -----------------------------------------------------------------------------------------------

# Hyper params

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
BATCH_SIZE = 2
NUM_EPOCHS = 100
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
HOLDOUT = 0.8 # Percentage of training images
NUM_WORKERS = 2
LOAD_MODEL = False
PIN_MEMORY = True
ROOT_DIR = "data/total_data"
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

# -----------------------------------------------------------------------------------------------

# List to store the losses from each epoch

TRAIN_LOSSES = []

# -----------------------------------------------------------------------------------------------

def train(loader, model, opt, loss_function, scaler):
  '''Train function. 

    Args:
      loader: The data loader
      model: Model to be trained 
      opt: The optimizer that will be used to train
      loss_function: The loss function that will be used to train 
      scaler: To do mixed precision training

    '''

  loop = tqdm(loader)

  train_loss = 0.0
  for _, (data, targets) in enumerate(loop):
    data = data.to(device = DEVICE)
    targets = targets.float().unsqueeze(1).to(device = DEVICE)

    # forward pass
    with torch.cuda.amp.autocast():
      preds = model(data)
      loss = loss_function(preds, targets)

    # backward pass
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    train_loss += loss.item()

    # update tqdm loop
    loop.set_postfix(loss = loss.item())

  TRAIN_LOSSES.append(train_loss)

# -----------------------------------------------------------------------------------------------

def main():
  train_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )
  
  val_transforms = A.Compose(
      [
          A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
          A.Normalize(
              mean=[0.0, 0.0, 0.0],
              std=[1.0, 1.0, 1.0],
              max_pixel_value=255.0,
          ),
          ToTensorV2(),
      ]
  )

  model = UNet(in_channels = 3, out_channels = 1).to(DEVICE)
  loss_function = nn.BCEWithLogitsLoss() #binary cross entropy with logits
  optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

  split_dataset(ROOT_DIR, 
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    VAL_IMG_DIR,
    VAL_MASK_DIR,
    holdout= HOLDOUT,
    seed = np.random.randint(1000000)
  )

  train_loader, val_loader = get_loaders(
      TRAIN_IMG_DIR,
      TRAIN_MASK_DIR,
      VAL_IMG_DIR,
      VAL_MASK_DIR,
      BATCH_SIZE,
      train_transforms,
      val_transforms,
      NUM_WORKERS,
      PIN_MEMORY
  )

  # if LOAD_MODEL:
  #   load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

  #   metrics(val_loader, model, device = DEVICE)

  scaler = torch.cuda.amp.GradScaler()
  for _ in range(NUM_EPOCHS):
    #perform the train steps (forward, backward)
    train(train_loader, model, optimizer, loss_function, scaler)

    # save the model
    checkpoint = {"state": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)

    # check some metrics (accuracy, dice score)
    metrics(val_loader, model, device = DEVICE)

    # print predictions in a folder
    save_preds(val_loader, model, folder = "saved_images/", device = DEVICE)

    plot_loss_curve(TRAIN_LOSSES, 'plots/train_loss.png')

# -----------------------------------------------------------------------------------------------

if __name__ == '__main__':
  main()