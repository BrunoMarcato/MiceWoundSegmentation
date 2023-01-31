import os
import torch
import torchvision
from sklearn.model_selection import train_test_split
from dataset import RatsDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from shutil import copy2

# -----------------------------------------------------------------------------------------------

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
  print("... Saving checkpoint ...")
  torch.save(state, filename)

# -----------------------------------------------------------------------------------------------

def load_checkpoint(checkpoint, model):
  print("... Loading checkpoint ...")
  model.load_state_dict(checkpoint["state_dict"])

# -----------------------------------------------------------------------------------------------

def get_loaders(
    train_image_dir,
    train_mask_dir,
    val_image_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers = 4,
    pin_memory = True
):
  train_dataset = RatsDataset(
      img_dir = train_image_dir,
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
      img_dir = val_image_dir,
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

def remove_folder_contents(path):
  files = os.listdir(path)
  if len(files) == 0:
    return
  else:
    for filename in files:
      try:
        file = os.path.join(path, filename)
        os.remove(file)
      except OSError as e:
        print(f'Failed to delete {file}; Reason: {e}')

# -----------------------------------------------------------------------------------------------

def split_dataset(data_path, 
                  train_images_path, train_masks_path, 
                  val_images_path, val_masks_path, 
                  holdout=0.8, seed=42):

  files = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i))]

  images = []
  masks = []
  for f in files:
      if f.endswith('_mask.png'):
          masks.append(f)
      else:
          images.append(f)

  images.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))
  masks.sort(key = lambda f: int(''.join(filter(str.isdigit, f))))

  X_train, X_val, y_train, y_val = train_test_split(images, masks, random_state = seed, test_size = 1-holdout)


  remove_folder_contents(train_images_path)
  remove_folder_contents(train_masks_path)
  remove_folder_contents(val_images_path)
  remove_folder_contents(val_masks_path)

  for f in X_train:
      copy2(os.path.join(data_path, f), os.path.join(train_images_path, f))

  for f in y_train:
      copy2(os.path.join(data_path, f), os.path.join(train_masks_path, f))

  for f in X_val:
      copy2(os.path.join(data_path, f), os.path.join(val_images_path, f))

  for f in y_val:
      copy2(os.path.join(data_path, f), os.path.join(val_masks_path, f))

# -----------------------------------------------------------------------------------------------