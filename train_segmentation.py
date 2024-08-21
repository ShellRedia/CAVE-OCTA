import torch
import torch.nn as nn
import torch.optim as optim
import time

import os
import random
import numpy as np
from tqdm import tqdm

from monai.networks.nets import *
from octa_datasets import OCTA_Dataset
from torch.utils.data import DataLoader

import pandas as pd
from collections import defaultdict


class ModifiedModel(nn.Module):
    def __init__(self, original_model):
        super(ModifiedModel, self).__init__()
        self.original_model = original_model
        self.new_layer = nn.Sigmoid()

    def forward(self, x):
        x = self.original_model(x)
        x = self.new_layer(x)
        return x

class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum()
        dice_score = (2. * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1. - dice_score
        return dice_loss
    
class TrainManager:
    def __init__(self, model_name="UNet", dataset_name="ROSE"):
        seed = 42

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = model_name
        self.dataset_name = dataset_name

        self.epochs = 200
        self.lr = 1e-4
        self.epsilon = 1e-8

        if model_name == "UNet":
            N, B = 3, 4
            channels = [2 ** x for x in range(B, B+N)]
            strides = [2] * (len(channels) - 1)
            self.model = UNet(in_channels=1, out_channels=1, spatial_dims=2, channels=channels,strides=strides)

        elif model_name == "SwinUNETR":
            self.model = SwinUNETR(img_size=(512,512), in_channels=1, out_channels=1, feature_size=24, spatial_dims=2)

        elif model_name == "SegResNet":
            self.model = SegResNet(in_channels=1, out_channels=1, spatial_dims=2)

        elif model_name == "FlexUNet":
            self.model = FlexUNet(in_channels=1, out_channels=1, spatial_dims=2, backbone="efficientnet-b4")

        elif model_name == "DiNTS":
            dints_space = TopologyInstance(spatial_dims=2, device="cuda")
            self.model = DiNTS(dints_space=dints_space, in_channels=1, num_classes=1, spatial_dims=2)

        self.model = ModifiedModel(self.model).to(device)
    
        pg = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(pg, lr=1, weight_decay=1e-4)
        epoch_p = self.epochs // 5
        lr_lambda = lambda x: max(1e-4, self.lr * x / epoch_p if x <= epoch_p else self.lr * 0.97 ** (x - epoch_p))
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        dataset_train = OCTA_Dataset(dataset_name, True)
        dataset_test = OCTA_Dataset(dataset_name, False)

        self.train_loader = DataLoader(dataset_train, batch_size=1)
        self.test_loader = DataLoader(dataset_test, batch_size=1)

        self.inputs_process = lambda x : x.to(device)
        self.loss_func = DiceLoss()

    def cal_jaccard_index(self, pred, label):
        intersection = (pred & label).sum().item()
        union = (pred | label).sum().item()
        jaccard_index = intersection / (union + self.epsilon)
        return jaccard_index

    def cal_dice(self, pred, label):
        intersection = (pred & label).sum().item()
        union = pred.sum().item() + label.sum().item()
        dice = 2 * intersection / (union + self.epsilon)
        return dice
        
    
    def train(self):
        metrics = defaultdict(list)

        to_cpu = lambda x:x[0][0].cpu().detach().int()
                    
        progress_bar = tqdm(
            range(self.epochs),
            desc="{}-{}".format(self.model_name, self.dataset_name)
        )
        # training loop:
        for epoch in range(1, self.epochs+1):
            for samples, labels in self.train_loader:
                samples, labels = map(self.inputs_process, (samples, labels))
                self.optimizer.zero_grad()
                preds = self.model(samples)
                self.loss_func(preds, labels).backward()
                self.optimizer.step()
            self.scheduler.step()

            dice_lst = []
            jac_lst = []
            
            for samples, labels in self.test_loader:
                samples, labels = map(self.inputs_process, (samples, labels))
                preds = self.model(samples)
                preds = torch.gt(preds, 0.8).int()
                labels = torch.gt(labels, 0.8).int()
                labels, preds = to_cpu(labels), to_cpu(preds)
                dice_lst.append(self.cal_dice(labels, preds))
                jac_lst.append(self.cal_jaccard_index(labels, preds))

            metrics["epoch"].append(epoch)
            dice_mean, jac_mean = round(sum(dice_lst) / len(dice_lst), 4), round(sum(jac_lst) / len(jac_lst), 4)
            metrics["dice"].append(dice_mean)
            metrics["jac"].append(jac_mean)

            save_dir = "{}/{}".format(self.model_name, self.dataset_name)
            os.makedirs(save_dir, exist_ok=True)

            pd.DataFrame(metrics).to_excel("{}/metrics.xlsx".format(save_dir))

            progress_bar.update(1)
            logs = {"dice": dice_mean,"jac": jac_mean}
            progress_bar.set_postfix(**logs)
            
            




if __name__=="__main__":
    for model_name in "UNet", "SwinUNETR", "SegResNet", "FlexUNet", "DiNTS":
        for dataset_name in "ROSE", "OCTA500-3M", "OCTA500-6M":
            tm = TrainManager(model_name, dataset_name)
            tm.train()