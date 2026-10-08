import os
import torch

import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

class Flatfish_dataset(Dataset):
    def __init__(self, root_dir, process):
        self.root_dir = root_dir
        self.type = process

        self.data = pd.read_csv(os.path.join(root_dir, self.type + '.csv'))
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        self.meta_cols = ['height', 'width', 'pixel', 'wdh', 'hdw',
                          'slope1', 'slope2', 'curvation', 'slope3', 'slope4', 'curvation2']
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.data.loc[idx, 'name']
        img_path = os.path.join(self.root_dir, "segment_images", name)
        image = Image.open(img_path)
        image = self.transform(image)

        label = int(self.data.loc[idx, 'target'])
        label = torch.tensor(label, dtype=torch.long)

        meta_vals = self.data.loc[idx, self.meta_cols].values.astype(np.float32)
        meta_info = torch.tensor(meta_vals, dtype=torch.float32)

        return image, label, meta_info


class Flatfish_dataset_v2(Dataset):
    def __init__(self, root_dir, process):
        self.root_dir = root_dir
        self.type = process

        self.data = pd.read_csv(os.path.join(root_dir, self.type + '.csv'))
        
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        self.meta_cols = ['slope1', 'slope2', 'curvation', 'slope3', 'slope4', 'curvation2']
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.data.loc[idx, 'name']
        img_path = os.path.join(self.root_dir, "segment_images", name)
        image = Image.open(img_path)
        image = self.transform(image)

        label = int(self.data.loc[idx, 'target'])
        label = torch.tensor(label, dtype=torch.long)

        meta_vals = self.data.loc[idx, self.meta_cols].values.astype(np.float32)
        meta_info = torch.tensor(meta_vals, dtype=torch.float32)

        return image, label, meta_info



class Flatfish_dataset_v3(Dataset):
    def __init__(self, root_dir, process, meta_cols=None):
        self.root_dir = root_dir
        self.type = process

        self.data = pd.read_csv(os.path.join(root_dir, self.type + '.csv'))

        self.rgb_transform = transforms.Compose([
                                                transforms.Resize((900, 1200)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                ])
        self.mask_transform = transforms.Compose([transforms.ToTensor()])

        if meta_cols is not None:
            self.meta_cols = meta_cols
        else:
            self.meta_cols = ['slope1', 'slope2', 'curvation', 'slope3', 'slope4', 'curvation2']
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.data.loc[idx, 'name']

        base_name = name.split('.')[0]

        possible_ext = ["jpg", "png", "jpeg"]
        rgb_path = None
        for ext in possible_ext:
            candidate = os.path.join(self.root_dir, "images", f"{base_name}.{ext}")
            if os.path.exists(candidate):
                rgb_path = candidate
                break

        if rgb_path is None:
            raise FileNotFoundError(f"No RGB image found for: {base_name}")
        

        mask_path = os.path.join(self.root_dir, "segment_images_v2", name)

        rgb_image = Image.open(rgb_path).convert("RGB")
        rgb_image = self.rgb_transform(rgb_image)

        mask_image = Image.open(mask_path)
        mask_image = self.mask_transform(mask_image)

        label = int(self.data.loc[idx, 'target'])
        label = torch.tensor(label, dtype=torch.long)

        meta_vals = self.data.loc[idx, self.meta_cols].values.astype(np.float32)
        meta_info = torch.tensor(meta_vals, dtype=torch.float32)

        return rgb_image, mask_image, label, meta_info




class Flatfish_dataset_v4(Dataset):
    def __init__(self, root_dir, process):
        self.root_dir = root_dir
        self.type = process

        self.data = pd.read_csv(os.path.join(root_dir, self.type + '.csv'))
        
        self.rgb_transform = transforms.Compose([
                                                transforms.Resize((900, 1200)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                ])
        self.mask_transform = transforms.Compose([transforms.ToTensor()])
        
        self.meta_cols = ['slope1', 'slope2', 'curvation', 'slope3', 'slope4', 'curvation2']
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name = self.data.loc[idx, 'name']

        base_name = name.split('.')[0]

        possible_ext = ["jpg", "png", "jpeg"]
        rgb_path = None
        for ext in possible_ext:
            candidate = os.path.join(self.root_dir, "images", f"{base_name}.{ext}")
            if os.path.exists(candidate):
                rgb_path = candidate
                break

        if rgb_path is None:
            raise FileNotFoundError(f"No RGB image found for: {base_name}")
        

        mask_path = os.path.join(self.root_dir, "segment_images_v2", name)

        rgb_image = Image.open(rgb_path).convert("RGB")
        rgb_image = self.rgb_transform(rgb_image)

        mask_image = Image.open(mask_path)
        mask_image = self.mask_transform(mask_image)

        label = int(self.data.loc[idx, 'target'])
        if label == 2:
            label = 1        
        label = torch.tensor(label, dtype=torch.long)

        meta_vals = self.data.loc[idx, self.meta_cols].values.astype(np.float32)
        meta_info = torch.tensor(meta_vals, dtype=torch.float32)

        return rgb_image, mask_image, label, meta_info
