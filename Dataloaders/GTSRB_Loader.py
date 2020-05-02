import torch
import os
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import random


class GTSRB(Dataset):
    base_folder = 'GTSRB'

    def __init__(self, root_dir, classes, W, H, transform=None):
        self.root_dir = root_dir
        self.W = W
        self.H = H
        self.sub_directory = 'Final_Training/Images'
        dir_path = os.path.join(self.root_dir, self.base_folder,self.sub_directory)
        csv_splits=[]
        names=[]
        for name in os.listdir(dir_path):
            csv_name = 'GT-' + str(name) + '.csv'
            csv_file_path = os.path.join(
                dir_path, name, csv_name)
            csv_data = pd.read_csv(csv_file_path).iloc[:,0]
            csv_splits = csv_splits + [x.split(';') for i, x in csv_data.items()]
            names = names + [name for i in range(len(csv_data))]
        # Random shuffle index
        # random.shuffle(csv_splits)
        self.names = names
        self.csv_data = csv_splits
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path_split = os.path.join(self.names[idx] ,self.csv_data[idx][0])
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                img_path_split)
        img = Image.open(img_path)
        img = img.resize((self.W,self.H))
        classId = torch.tensor(np.array([int(self.csv_data[idx][7])]))

        if self.transform is not None:
            img = self.transform(img)

        img_shape = list(map(int, self.csv_data[idx][1:3]))
        x1y1x2y2 = list(map(int, self.csv_data[idx][3:7]))

        return img, classId, img_shape, x1y1x2y2


class test_GTSRB(Dataset):
    base_folder = 'GTSRB'

    def __init__(self, root_dir, W, H, train=False, transform=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.W = W
        self.H = H
        self.root_dir = root_dir
        self.sub_directory = 'Final_Training/Images' if train else 'Final_Test/Images'
        self.csv_file_name = 'training.csv' if train else 'GT-final_test.csv'

        csv_file_path = os.path.join(
            root_dir, self.base_folder, self.sub_directory, self.csv_file_name)
        self.csv_data = pd.read_csv(csv_file_path)
        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.base_folder, self.sub_directory,
                                self.csv_data.iloc[idx,0].split(';')[0])
        img = Image.open(img_path)
        img = img.resize((self.W, self.H))
        classId = torch.tensor(np.array([int(self.csv_data.iloc[idx,0].split(';')[-1])]))

        if self.transform is not None:
            img = self.transform(img)


        img_shape = list(map(int,self.csv_data.iloc[0,0].split(';')[1:3]))
        x1y1x2y2 = list(map(int,self.csv_data.iloc[0, 0].split(';')[3:]))

        return img, classId, img_shape, x1y1x2y2