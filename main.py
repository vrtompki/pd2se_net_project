import torch
from torch import nn
from dataset import PlantsDiseaseDataset
from config import *
from model import PD2SEModel

if __name__ == '__main__':
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    criterion3 = nn.CrossEntropyLoss()

    train_dataset = PlantsDiseaseDataset()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4)
    mod = PD2SEModel()
