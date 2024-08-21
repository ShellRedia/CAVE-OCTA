from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import albumentations as alb

probability = 0.2
transform = alb.Compose([
    # level 1
    alb.RandomBrightnessContrast(p=probability),
    alb.CLAHE(p=probability), 
    # level 2
    alb.Rotate(limit=15, p=probability),
    alb.VerticalFlip(p=probability),
    alb.HorizontalFlip(p=probability),
    # level 3
    alb.AdvancedBlur(p=probability),
    alb.PiecewiseAffine(p=probability),
    alb.CoarseDropout(40,10,10,p=probability),
])
resize = alb.Compose([alb.Resize(height=512, width=512, always_apply=True, p=1)])

class OCTA_Dataset(Dataset):
    def __init__(self, dataset_name="ROSE", is_training=True):
        self.is_training = is_training
        sub_dir = {True:"train", False:"test"}[is_training]
        
        data_dir = "datasets/{}/{}/images".format(dataset_name, sub_dir)
        label_dir = "datasets/{}/{}/labels".format(dataset_name, sub_dir)

        sample_paths = sorted(["{}/{}".format(data_dir, x) for x in os.listdir(data_dir)])
        label_paths = sorted(["{}/{}".format(label_dir, x) for x in os.listdir(label_dir)])

        self.samples = [cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in sample_paths]
        self.samples = [np.array(x) for x in self.samples]
        self.labels = [cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in label_paths]
        self.labels = [np.array(x) for x in self.labels]
        

            
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample, label = self.samples[index], self.labels[index]
        process = lambda x : np.array([x], dtype=np.float32) / 255

        if self.is_training:
            transformed = transform(**{"image": sample, "mask": label})
            sample, label = transformed["image"], transformed["mask"]
    
        transformed = resize(**{"image": sample, "mask": label})
        sample, label = transformed["image"], transformed["mask"]
            
        return process(sample), process(label)