"""
definition of datasets
Author Vincent with influence from LucasX
"""
import json
import os
import sys
import pickle
import tqdm
import torchvision
import numpy as np
from PIL import Image
from skimage import io
from skimage.transform import resize
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import NearMiss
from torch.utils.data import Dataset
from config import *
sys.path.append('../')


# TODO apply SMOTE or Near Miss and handler for non flatten images
def retrieve_images(file_path=TRAIN_ANN_PATH, image_path=TRAIN_IMG_PATH, flatten=True, subset=1.0):
    images = []
    labels_1 = []
    labels_2 = []
    labels_3 = []
    with open(file_path, mode='rt', encoding='utf-8') as f:
        for _ in json.load(f):
            img_fp = os.path.join(image_path,
                                  _['image_id']).encode('ascii', 'ignore').decode('utf-8')
            if os.path.exists(img_fp):
                image = Image.open(img_fp)
                transform_resize = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
                    torchvision.transforms.ToTensor()
                ])
                lab = str(_['disease_class'])
                if lab in MAP_61to45.keys():
                    if flatten:
                        image = transform_resize(image).flatten().numpy()
                    images.append(np.transpose(image))
                    labels_1.append(MAP_61to45[lab]['Severity ID'])
                    labels_2.append(MAP_61to45[lab]['Disease ID'])
                    labels_3.append(MAP_61to45[lab]['Plant ID'])
    return {'Images': images, 'Severity ID': labels_1, 'Disease ID': labels_2, 'Plant ID': labels_3}


def imbalanced_sampler(input_data, input_labels, method='SMOTE'):
    if method == 'SMOTE':
        sampler = BorderlineSMOTE(n_jobs=4, random_state=RANDOM_STATE)
    elif method == 'Near Miss':
        sampler = NearMiss(n_jobs=4, random_state=RANDOM_STATE)
    else:
        print('Invalid sampler type. Only `SMOTE` (Borderline) and `Near Miss` are supported...')
        sys.exit(0)
    # TODO save samples by class to reduce file size
    max_class_num = np.max(input_labels)
    class_range = np.arange(1, max_class_num)
    x_sampled, y_sampled = sampler.fit_resample(input_data, input_labels)
    for i in class_range:
        idx = np.argwhere(y_sampled == i)
        pickle.dump(x_sampled[idx][:], open(method + '_Class_' + str(i) + '_data_samples.pkl', 'wb'))
        pickle.dump(y_sampled[idx], open(method + '_Class_' + str(i) + '_label_samples.pkl', 'wb'))
    return x_sampled, y_sampled


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
# TODO verify images that are saved are only for 45 classes....
        if train_val == 'train':
            with open(train_json, mode='rt', encoding='utf-8') as f:
                for _ in json.load(f):
                    img_fp = os.path.join(TRAIN_IMG_PATH,
                                          _['image_id']).encode('ascii', 'ignore').decode('utf-8')
                    if os.path.exists(img_fp):
                        label_to_match = _['disease_class']
                        if label_to_match in MAP_61to45.keys():
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


if __name__ == '__main__':
    data = retrieve_images()
    sampled_images_SMOTE, sampled_labels_SMOTE = imbalanced_sampler(input_data=data['Images'],
                                                                    input_labels=data['Plant ID'])
    sampled_images_Near_Miss, sampled_labels_Near_Miss = imbalanced_sampler(input_data=data['Images'],
                                                                            input_labels=data['Plant ID'],
                                                                            method='Near Miss')

