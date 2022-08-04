import os
import cv2
from tqdm import tqdm #to create a task bar
from glob import glob
from albumentations import CenterCrop, RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------

def augment_data(images, masks, save_images, save_masks, H=1024, W=768, save_path=None):

    #TODO: Document the parameters

    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = x.split("/")[-1].split(".")
        """ Extracting the name and extension of the image and the mask. """
        image_name = name[0]
        image_extn = name[1]

        name = y.split("/")[-1].split(".")
        mask_name = name[0]
        mask_extn = name[1]

        """ Reading image and mask. """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        """ Augmentation """
        aug = CenterCrop(600, 600, p=1.0)
        augmented = aug(image=x, mask=y)
        x1 = augmented["image"]
        y1 = augmented["mask"]

        aug = RandomRotate90(p=1.0)
        augmented = aug(image=x, mask=y)
        x2 = augmented['image']
        y2 = augmented['mask']

        aug = GridDistortion(p=1.0)
        augmented = aug(image=x, mask=y)
        x3 = augmented['image']
        y3 = augmented['mask']

        aug = HorizontalFlip(p=1.0)
        augmented = aug(image=x, mask=y)
        x4 = augmented['image']
        y4 = augmented['mask']

        aug = VerticalFlip(p=1.0)
        augmented = aug(image=x, mask=y)
        x5 = augmented['image']
        y5 = augmented['mask']

        save_images = [x1, x2, x3, x4, x5]
        save_masks =  [y1, y2, y3, y4, y5]

        """ Saving the image and mask. """
        idx = 0
        for i, m in zip(save_images, save_masks):
            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W, H))

            if save_path is not None:
                if len(images) == 1:
                    tmp_img_name = f"{image_name}.{image_extn}"
                    tmp_mask_name = f"{mask_name}.{mask_extn}"

                else:
                    tmp_img_name = f"{image_name}_{idx}.{image_extn}"
                    tmp_mask_name = f"{mask_name}_{idx}.{mask_extn}"

                image_path = os.path.join(save_path, "images", tmp_img_name)
                mask_path = os.path.join(save_path, "labels", tmp_mask_name)

                cv2.imwrite(image_path, i)
                cv2.imwrite(mask_path, m)

                idx += 1

            save_images.append(i)
            save_masks.append(m)

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------