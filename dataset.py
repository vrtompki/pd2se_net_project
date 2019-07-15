"""
definition of datasets
Author: LucasX
"""
import json
import os
import sys
import tqdm
import torchvision
import numpy as np
from PIL import Image
from skimage import io
from skimage.transform import resize
from torch.utils.data import Dataset
from config import *
sys.path.append('../')


class PlantsDiseaseDataset(Dataset):
    """
    Plants Disease Dataset
    """
    def __init__(self, train_val='train', transform=None):
        """
        PyTorch Dataset definition
        :param train_val:
        :param transform:
        """
        train_json = TRAIN_ANN_PATH
        val_json = VAL_ANN_PATH
        imgs = []
        lbs = []

        if train_val == 'train':
            with open(train_json, mode='rt', encoding='utf-8') as f:
                for _ in json.load(f):
                    img_fp = os.path.join(TRAIN_IMG_PATH,
                                          _['image_id']).encode('ascii', 'ignore').decode('utf-8')
                    if os.path.exists(img_fp):
                        imgs.append(img_fp)
                        lbs.append(_['disease_class'])
            sub_lbs = []
            for idx in range(len(lbs)):
                if str(lbs[idx]) in MAP_61to45.keys():
                    lbs[idx] = MAP_61to45[str(lbs[idx])]
                    sub_lbs.append(lbs[idx])
            self.img_files = imgs
            self.labels = sub_lbs
        elif train_val == 'val':
            with open(val_json, mode='rt', encoding='utf-8') as f:
                for _ in json.load(f):
                    img_fp = os.path.join(VAL_IMG_PATH,
                                          _['image_id']).encode('ascii', 'ignore').decode('utf-8')
                    if os.path.exists(img_fp):
                        imgs.append(img_fp)
                        lbs.append(_['disease_class'])
            sub_lbs = []
            for idx in range(len(lbs)):
                if str(lbs[idx]) in MAP_61to45.keys():
                    lbs[idx] = MAP_61to45[str(lbs[idx])]
                    sub_lbs.append(lbs[idx])
            self.img_files = imgs
            self.labels = sub_lbs
        else:
            print('Invalid data type. Since it only supports [train/val]...')
            sys.exit(0)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path)
        transform_resize = torchvision.transforms.Compose([
            torchvision.transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            torchvision.transforms.ToTensor()
        ])
        image = transform_resize(image)
        # TODO add in transformations
        plant_lbl = self.labels[idx]['Plant ID']
        disease_lbl = self.labels[idx]['Disease ID']
        severity_lbl = self.labels[idx]['Severity ID']

        sample = {'image': image, 'label_1': plant_lbl, 'label_2': disease_lbl,
                  'label_3': severity_lbl, 'filename': self.img_files[idx]}
        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))
        return sample


class PlantsDiseaseInferenceDataset(Dataset):
    """
    Plants Disease Inference dataset
    """
    def __init__(self, transform=None):
        """
        PyTorch Dataset definition
        :param transform:
        """
        inference_base = '/var/log/PDR'
        img_files = []
        for img_f in os.listdir(os.path.join(inference_base, 'AgriculturalDisease_testA', 'images')):
            # img_fp = os.path.join(inference_base, 'AgriculturalDisease_testA', 'images', img_f)
            img_fp = os.path.join(inference_base, 'AgriculturalDisease_testA',
                                  'images', img_f).encode('ascii', 'ignore').decode('utf-8')
            if os.path.exists(img_fp):
                img_files.append(img_fp)

        self.img_files = img_files
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        image = io.imread(self.img_files[idx])
        sample = {'image': image, 'filename': self.img_files[idx]}
        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))
        return sample
