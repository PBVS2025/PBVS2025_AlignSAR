import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import random
from PIL import Image
from torchvision import transforms

class SarEODataset(Dataset):
    def __init__(self, root_dir, transform=None):
        
        self.root_dir = root_dir
        self.transform = transform
        
        self.sar_root = os.path.join(root_dir, 'SAR_Train')
        self.eo_root = os.path.join(root_dir, 'EO_Train')
        
        self.classes = sorted(os.listdir(self.sar_root))
        self.cls_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.sar_paths = []
        self.eo_paths = []
        self.labels = []
        
        for class_name in self.classes:
            sar_class_dir = os.path.join(self.sar_root, class_name)
            eo_class_dir = os.path.join(self.eo_root, class_name)

            # sar_files = sorted(os.listdir(sar_class_dir))
            sar_files = sorted([f for f in os.listdir(sar_class_dir) 
                              if f.lower().endswith('.png')])

            for sar_file in sar_files:

                eo_file = sar_file
                if os.path.exists(os.path.join(eo_class_dir, eo_file)):
                    self.sar_paths.append(os.path.join(sar_class_dir, sar_file))
                    self.eo_paths.append(os.path.join(eo_class_dir, eo_file))
                    self.labels.append(self.cls_to_idx[class_name])
        
    def __len__(self):
        return len(self.sar_paths)

    def __getitem__(self, idx):

        sar_path = self.sar_paths[idx]
        sar_image = Image.open(sar_path).convert("RGB")

        eo_path = self.eo_paths[idx]
        eo_image = Image.open(eo_path).convert('RGB')

        if self.transform:
            sar_image = self.transform(sar_image)
            eo_image = self.transform(eo_image)
        else:
            to_tensor = transforms.ToTensor()
            sar_image = to_tensor(sar_image)
            eo_image = to_tensor(eo_image)

        label = self.labels[idx]

        return sar_image, eo_image, label