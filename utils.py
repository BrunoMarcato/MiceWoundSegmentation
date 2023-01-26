import torch
import torchvision
from dataset import RatsDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------------------------

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
  print(" ... Saving checkpoint ...")
  torch.save(state, filename)

# -----------------------------------------------------------------------------------------------

def load_checkpoint(checkpoint, model):
  print("... Loading checkpoint ...")
  model.load_state_dict(checkpoint["state_dict"])

# -----------------------------------------------------------------------------------------------

def get_loaders(
    train_dir,
    train_mask_dir,
    val_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers = 4,
    pin_memory = True
):
  train_dataset = RatsDataset(
      img_dir = train_dir,
      mask_dir = train_mask_dir,
      transform = train_transform
  )

  train_loader = DataLoader(
      train_dataset,
      batch_size = batch_size,
      num_workers = num_workers,
      pin_memory = pin_memory,
      shuffle = True
  )

  val_dataset = RatsDataset(
      img_dir = val_dir,
      mask_dir = val_mask_dir,
      transform = val_transform
  )

  val_loader = DataLoader(
      val_dataset,
      batch_size = batch_size,
      num_workers = num_workers,
      pin_memory = pin_memory,
      shuffle = False
  )

  return train_loader, val_loader

# -----------------------------------------------------------------------------------------------

def metrics(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum()) + 1e-8
            
    print(f"--> Correct pixels: {num_correct}; Total pixels: {num_pixels}")
    print(f"--> Accuracy: {num_correct/num_pixels*100:.2f}")
    print(f"--> Dice score: {dice_score/len(loader)}")


    model.train()

# -----------------------------------------------------------------------------------------------

def save_preds(loader, model, folder="saved_images/", device="cuda"):

    model.eval()
    
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()

# -----------------------------------------------------------------------------------------------

def plot_loss_curve(train_loss, path):
    #plt.legend('Train/Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    #plt.yticks(np.arange(0, 1, 0.1))

    plt.plot(train_loss, color='blue')
    #plt.plot(val_loss, color='red')

    plt.savefig(path)

# -----------------------------------------------------------------------------------------------