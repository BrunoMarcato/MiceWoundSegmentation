import os
import torch
import torchvision
import numpy as np
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
    test_image_dir,
    test_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    test_transform,
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

  test_dataset = RatsDataset(
      img_dir = test_image_dir,
      mask_dir = test_mask_dir,
      transform = test_transform
  )

  test_loader = DataLoader(
      test_dataset,
      batch_size = 1,
      num_workers = 0,
      pin_memory = pin_memory,
      shuffle = True
  )

  return train_loader, val_loader, test_loader

# -----------------------------------------------------------------------------------------------

def metrics(loader, model, mode = 'val', device="cuda"):
    dice_score = 0

    model.eval()

    if mode == 'val':
        num_correct = 0
        num_pixels = 0

        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                dice_score += (2. * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
                
        print(f"--> Correct pixels: {num_correct}; Total pixels: {num_pixels}")
        print(f"--> Accuracy: {num_correct/num_pixels*100:.2f}")
        print(f"--> Dice score: {dice_score/len(loader)}")
    
    elif mode == 'test':
        dice_scores = []
        with torch.no_grad():
            for x, y, fname in loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()

                dice_score = (2. * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
                dice_score = np.array(dice_score)
                dice_scores.append((dice_score, ''.join(fname)))
        
        model.train()

        return dice_scores

    else:
        raise ValueError(f'{mode} is an invalid mode. Allowed values: [\'val\',  \'test\']')


    model.train()

# -----------------------------------------------------------------------------------------------

def save_preds(loader, model, num_exec, folder="test_images_pred/", device="cuda"):

    model.eval()
    
    for x, y, fname in loader:
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        #{nome do modelo}_{numero da execução}_{nome da imagem}.png
        torchvision.utils.save_image(preds, f"{folder}/{model.name}_{num_exec}_{''.join(fname)}")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{''.join(fname)}")

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
                  test_images_path, test_masks_path, 
                  test_size=0.2, val_size=0.2, seed=42):

    #tip: val_size refers to the percentage over the train subset

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

    X_train, X_test, y_train, y_test = train_test_split(images, masks, random_state = seed, test_size = test_size)


    remove_folder_contents(train_images_path)
    remove_folder_contents(train_masks_path)
    remove_folder_contents(val_images_path)
    remove_folder_contents(val_masks_path)
    remove_folder_contents(test_images_path)
    remove_folder_contents(test_masks_path)

    counter = 0
    for f in X_train:
        copy2(os.path.join(data_path, f), os.path.join(train_images_path, f))

        if counter < int(val_size*len(X_train)):
            copy2(os.path.join(data_path, f), os.path.join(val_images_path, f))

        counter += 1

    counter = 0
    for f in y_train:
        copy2(os.path.join(data_path, f), os.path.join(train_masks_path, f))

        if counter < int(val_size*len(X_train)):
            copy2(os.path.join(data_path, f), os.path.join(val_masks_path, f))
        
        counter += 1

    for f in X_test:
        copy2(os.path.join(data_path, f), os.path.join(test_images_path, f))

    for f in y_test:
        copy2(os.path.join(data_path, f), os.path.join(test_masks_path, f))

# -----------------------------------------------------------------------------------------------