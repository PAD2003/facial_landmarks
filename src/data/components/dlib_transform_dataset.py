from torch.utils.data import Dataset
from .dlib_dataset import DlibDataset
from typing import Optional
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import torch
import torchvision
from PIL import Image

class DlibTransformDataset(Dataset):
    def __init__(self, 
                 dataset: DlibDataset, 
                 transform: Optional[A.Compose] = None):
        # set dataset
        self.dataset = dataset

        # set transform
        if transform is None:
            self.transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # get sample from DlibDataset (landmarks: pixels coordinates)
        cropped_image, landmarks = self.dataset[index]

        # turn cropped_image from PIL format into np.array
        cropped_image = np.array(cropped_image)

        # transform
        transformed = self.transform(image=cropped_image, keypoints=landmarks)
        transformed_cropped_image = transformed["image"]
        transformed_landmarks = transformed["keypoints"]

        # normalize transformed_landmarks
        color_channels, height, width = transformed_cropped_image.shape
        normalized_transformed_landmarks = transformed_landmarks / np.array([width, height]) - 0.5

        # landmarks: centered and normalized
        return transformed_cropped_image, normalized_transformed_landmarks.astype(np.float32)
    
    @staticmethod
    def annotate_tensor(batch_transformed_cropped_image: torch.Tensor, batch_normalized_transformed_landmarks: np.array):
        # define mean, std
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # reverse COLOR transform
        def denormalize(batch_transformed_cropped_image, mean=mean, std=std) -> torch.Tensor:
            # clone: make a copy
            # permute: [batch, 3, H, W] -> [3, H, W, batch]
            tmp = batch_transformed_cropped_image.clone().permute(1, 2, 3, 0)

            # denormalize
            for t, m, s in zip(tmp, mean, std):
                t.mul_(s).add_(m)

            # clamp: limit value to [0, 1]
            # permute: [3, H, W, batch] -> [batch, 3, H, W]
            return torch.clamp(tmp, 0, 1).permute(3, 0, 1, 2)
        
        batch_cropped_image = denormalize(batch_transformed_cropped_image)

        # set an empty list
        images_to_save = []

        # loop through each sample in batch
        for cropped_image, normalized_transformed_landmarks in zip(batch_cropped_image, batch_normalized_transformed_landmarks):
            # get size of cropped_image
            cropped_image = cropped_image.permute(1, 2, 0).numpy() * 255
            height, width, color_channels = cropped_image.shape

            # denormalize landmarks -> pixel coordinates
            landmarks = (normalized_transformed_landmarks + 0.5) * np.array( [width, height] )

            # draw landmarks on cropped image
            annotated_cropped_image = DlibDataset.annotate_image(Image.fromarray(cropped_image.astype(np.uint8)), landmarks)

            # save drawed cropped image
            images_to_save.append( torchvision.transforms.ToTensor()(annotated_cropped_image) )

        return torch.stack(images_to_save)