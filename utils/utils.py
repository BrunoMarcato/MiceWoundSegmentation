import os
import torch
import torchvision
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from UNet.dataset import RatsDataset
from torch.utils.data import DataLoader
from shutil import copy2
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# -----------------------------------------------------------------------------------------------

def save_model(model, filename = "/model"):
  print("... Saving model ...")
  with open(filename, 'wb') as file:
    pickle.dump(model, file)

# -----------------------------------------------------------------------------------------------

def load_model(filename):
    print("... Loading model ...")
    with open(filename, 'rb') as file:
        model = pickle.load(file)

    return model

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

def get_fnames_from_loader(loader):
    names = []
    for _, _, fname in loader:
        names.append(''.join(fname))

    return names

# -----------------------------------------------------------------------------------------------

#metrics for unet
def metrics_unet(loader, model, summary_writer, epoch = None, mode = 'val', device="cuda"):

    model.eval()

    if mode == 'val':
        num_correct = 0
        num_pixels = 0
        score = 0
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
                num_correct += (preds == y).sum()
                num_pixels += torch.numel(preds)
                score += f1_score(y.cpu().numpy().flatten(), preds.cpu().numpy().flatten())
                #(2. * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

        summary_writer.add_scalar("Val_Dice_Score", score/len(loader), epoch)

        print(f"--> Correct pixels: {num_correct}; Total pixels: {num_pixels}")
        print(f"--> Accuracy: {num_correct/num_pixels*100:.2f}")
        print(f"--> Dice score: {(score/len(loader)):.4f}")
    
    elif mode == 'test':
        f1scores = []
        with torch.no_grad():
            for num, (x, y, _) in enumerate(loader):
                x = x.to(device)
                y = y.to(device).unsqueeze(1)
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()

                score = f1_score(y.cpu().numpy().flatten(), preds.cpu().numpy().flatten())

                summary_writer.add_scalar("Test_Dice_Score", score, num)

                f1scores.append(score)
        
        f1scores = np.array(f1scores)

        model.train()

        return f1scores

    else:
        raise ValueError(f'{mode} is an invalid mode. Allowed values: [\'val\',  \'test\']')


    model.train()

# -----------------------------------------------------------------------------------------------

#Save predictions from UNet
def save_preds(loader, model, num_run, folder="UNet/test_images_pred", device="cuda"):

    model.eval()
    
    for x, _, fname in loader:
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(preds, f"{folder}/{model.name}_run{num_run}_{''.join(fname)}")

    model.train()

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