import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# -----------------------------------------------------------------------------------------------

class RatsDataset(Dataset):
  def __init__(self, img_dir, mask_dir, transform=None):
    self.img_dir = img_dir
    self.mask_dir = mask_dir
    self.transform = transform
    self.images = os.listdir(img_dir)
  
  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, idx):
    fname = self.images[idx]
    img_path = os.path.join(self.img_dir, fname)
    mask_path = os.path.join(self.mask_dir, fname.replace(".png", "_mask.png"))

    image = np.array(Image.open(img_path).convert('RGB'))
    mask = np.array(Image.open(mask_path).convert('L'), dtype = np.float32)
    mask[mask == 38.0] = 1.0

    if self.transform != None:
      augmentations = self.transform(image = image, mask = mask) #Albumentation transforms
      image = augmentations['image']
      mask = augmentations['mask']

      #image = self.transform(image) #torch transforms
      #mask = self.transform(mask)

    return image, mask, fname

# -----------------------------------------------------------------------------------------------