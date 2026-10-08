import os

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset

# mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class Angle_cls_dataset(Dataset):
    def __init__(self, root_dir, process):
        self.root_dir = root_dir
        self.type = process

        self.normal_data_path = os.path.join(self.root_dir, self.type, 'normal')
        self.abnormal_data_path = os.path.join(self.root_dir, self.type, 'abnormal')

        self.normal_filenames = os.listdir(os.path.join(self.normal_data_path))
        self.normal_filenames = [os.path.join(self.normal_data_path, x) for x in self.normal_filenames]

        self.abnormal_filenames = os.listdir(os.path.join(self.abnormal_data_path))
        self.abnormal_filenames = [os.path.join(self.abnormal_data_path, x) for x in self.abnormal_filenames]

        self.filenames = self.normal_filenames + self.abnormal_filenames

        # self.labels = [0] * len(self.normal_filenames) + [1] * len(self.abnormal_filenames)

        self.transform = transforms.Compose([transforms.Resize(size = (256, 256)),
                                             transforms.ToTensor(),
                                             # transforms.Normalize(mean = mean, std = std),
                                             ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx])
        image = self.transform(image)

        if 'abnormal' in self.filenames[idx]:
            label = np.array([1])
        elif 'normal' in self.filenames[idx]:
            label = np.array([0])

        return image, label