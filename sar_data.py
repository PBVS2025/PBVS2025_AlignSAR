import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import random
import torchvision.transforms as transforms

class PairedTransform:
    def __init__(self, resize_size=224, rotation_degree=30, flip_prob=0.5, vertical_flip_prob=0.5):
        self.resize_size = resize_size
        self.rotation_degree = rotation_degree
        self.flip_prob = flip_prob
        self.vertical_flip_prob = vertical_flip_prob

        # Common augmentations
        self.resize = transforms.Resize((resize_size, resize_size))  # 224x224 Resize 추가
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __call__(self, sar):
        flip_flag = random.random() < self.flip_prob 
        vflip_flag = random.random() < self.vertical_flip_prob
        angle = transforms.RandomRotation.get_params([-self.rotation_degree, self.rotation_degree])

        if flip_flag:
            sar = transforms.functional.hflip(sar)

        if vflip_flag:
            sar = transforms.functional.vflip(sar)

        sar = transforms.functional.rotate(sar, angle)

        # Resize → ToTensor → Normalize (-1~1)
        sar = self.normalize(self.to_tensor(self.resize(sar)))

        return sar

class SarDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.sar_root = os.path.join(root_dir, 'SAR_Train')

        self.classes = sorted(os.listdir(self.sar_root))
        self.cls_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.sar_paths = []
        self.labels = []

        for class_name in self.classes:
            sar_class_dir = os.path.join(self.sar_root, class_name)

            sar_files = sorted([f for f in os.listdir(sar_class_dir) if f.lower().endswith('.png')])

            for sar_file in sar_files:
                self.sar_paths.append(os.path.join(sar_class_dir, sar_file))
                self.labels.append(self.cls_to_idx[class_name])

    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.sar_paths)

    def __getitem__(self, idx):
        sar_path = self.sar_paths[idx]
        sar_image = Image.open(sar_path).convert("RGB")
        
        if self.transform:
            sar_image = self.transform(sar_image)
        else:
            resize = transforms.Resize((224, 224))  # Resize 추가
            to_tensor = transforms.ToTensor()
            normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            sar_image = normalize(to_tensor(resize(sar_image)))

        label = self.labels[idx]
        return sar_image, label

# transform = PairedTransform(resize_size=224, flip_prob=0.5, vertical_flip_prob=0.5)

# dataset = SarEODataset(root_dir="your_dataset_path", transform=transform)

# dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
