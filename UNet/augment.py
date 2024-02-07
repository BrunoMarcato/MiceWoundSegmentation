import albumentations as A
from albumentations.pytorch import ToTensorV2

def offline_augment(dataset, transforms, test=False):
    augmented_dataset = []

    all_img_transforms = A.Compose([
        A.Resize(height = 256, width = 256),
        A.Normalize(mean=0.0, std=1.0, max_pixel_value=255.0),
        ToTensorV2()
    ])

    # adds the old dataset, with transformations, to the new one
    for image, mask, fname in dataset:
        augmentations = all_img_transforms(image = image, mask = mask)
        transformed_image = augmentations['image']
        transformed_mask = augmentations['mask']

        augmented_dataset.append((transformed_image, transformed_mask, fname))

    if not test:
        for transform in transforms:
            composed_transform = A.Compose([
                transform,
                all_img_transforms,
            ])

            for image, mask, fname in dataset:

                # Apply the transformation to the image
                augmentations = composed_transform(image=image, mask=mask)

                transformed_image = augmentations['image']

                transformed_mask = augmentations['mask']

                # Adds the transformed image to the augmented dataset
                augmented_dataset.append((transformed_image, transformed_mask, fname))

    return augmented_dataset
