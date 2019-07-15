from torchvision import models
import torch
from torch import nn
from pytorchcv.models.shufflenetv2 import ShuffleUnit
import warnings
import torch.nn.functional as F
from config import *
warnings.filterwarnings("ignore", category=DeprecationWarning)


class BasicCNN(nn.Module):
    def __int__(self):
        super(BasicCNN, self).__init__()
        self.Layer0 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4)
        self.relu1 = nn.ReLU()

        self.Layer1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4)
        self.relu2 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.Layer2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4)
        self.relu3 = nn.ReLU()

        self.Layer3 = nn.Conv2d(in_channels=32, out_channels=256, kernel_size=4)
        self.relu4 = nn.ReLU()

        self.FC1 = nn.Linear(128*128*256, NUM_CLASSES_1)
        self.FC2 = nn.Linear(128*128*256, NUM_CLASSES_2)
        self.FC3 = nn.Linear(128*128*256, NUM_CLASSES_3)

    def forward(self, x):
        x = self.Layer0(x)
        x = self.relu1(x)

        x = self.Layer1(x)
        x = self.relu2(x)

        x = self.pool(x)

        x = self.Layer2(x)
        x = self.relu3(x)

        x = self.Layer3(x)
        x = self.relu4(x)

        x1 = F.softmax(self.FC1(x))
        x2 = F.softmax(self.FC2(x))
        x3 = F.softmax(self.FC3(x))

        return x1, x2, x3

# class ResNet_45(nn.Module):
#     def __int__(self, pretrained=False):
#         super(ResNet_45, self).__init__()
#         self.pre_train = pretrained
#         res_base = models.resnet50(pretrained=self.pre_train)
#         num_fts = res_base.fc.in_features
#         res_base.fc = nn.Linear(num_fts, 45)
#         if self.pre_train:
#             # freeze layers
class DiseaseModel(nn.Module):
    def __init__(self):
        super(DiseaseModel, self).__init__()
        base_mod = models.resnet50()
        num_fts = base_mod.fc.in_features
        base_mod.fc = nn.Linear(num_fts, NUM_CLASSES_3)
        base_path = 'C:\\Users\\Vincent\\Documents\\Big Data in Agriculture\\Datasets\\Liang\\pd2se_net_project\\models\\EPOCH8_loss_0.999_acc_0.67913-06-19'
        checkpoint = torch.load(base_path)
        base_mod.load_state_dict(checkpoint['model_state_dict'])
        children_list = list(base_mod.children())
        self.Layer0 = nn.Sequential(*children_list[0:4])
        self.Layer1 = nn.Sequential(children_list[4])
        self.Layer2 = nn.Sequential(children_list[5])
        self.Layer3 = nn.Sequential(children_list[6])
        self.Shuffle2_1 = ShuffleUnit(in_channels=1024, out_channels=1024, downsample=False,
                                      use_se=False, use_residual=False)
        self.Shuffle2_2 = ShuffleUnit(in_channels=1024, out_channels=1024, downsample=True,
                                      use_se=False, use_residual=False)
        self.Shuffle2_3 = ShuffleUnit(in_channels=1024, out_channels=2048, downsample=True,
                                      use_se=False, use_residual=False)
        self.FC = nn.Linear(2048 * 4 * 4, NUM_CLASSES_2)

    def forward(self, x):
        x = self.Layer0(x)
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = self.Shuffle2_3(self.Shuffle2_2(self.Shuffle2_1(x)))
        x = F.softmax(self.FC(x.view(-1, 2048*4*4)), dim=1)
        return x


class PlantModel(nn.Module):
    def __init__(self):
        super(PlantModel, self).__init__()
        base_mod = models.resnet50()
        num_fts = base_mod.fc.in_features
        base_mod.fc = nn.Linear(num_fts, NUM_CLASSES_3)
        base_path = 'C:\\Users\\Vincent\\Documents\\Big Data in Agriculture\\Datasets\\Liang\\pd2se_net_project\\models\\EPOCH8_loss_0.999_acc_0.67913-06-19'
        checkpoint = torch.load(base_path)
        base_mod.load_state_dict(checkpoint['model_state_dict'])
        children_list = list(base_mod.children())
        self.Layer0 = nn.Sequential(*children_list[0:4])
        self.Layer1 = nn.Sequential(children_list[4])
        self.Layer2 = nn.Sequential(children_list[5])
        self.Layer3 = nn.Sequential(children_list[6])
        self.Shuffle1_1 = ShuffleUnit(in_channels=512, out_channels=512, downsample=False,
                                      use_residual=False, use_se=False)
        self.Shuffle1_2 = ShuffleUnit(in_channels=512, out_channels=512, downsample=True,
                                      use_residual=False, use_se=False)
        self.Shuffle1_3 = ShuffleUnit(in_channels=512, out_channels=2048, downsample=True,
                                      use_residual=False, use_se=False)
        self.FC = nn.Linear(2048 * 8 * 8, NUM_CLASSES_1)

    def forward(self, x):
        x = self.Layer0(x)
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = self.Shuffle1_3(self.Shuffle1_2(self.Shuffle1_1(x)))
        x = F.softmax(self.FC(x.view(-1, 2048*8*8)), dim=1)
        return x


class PD2SEModel(nn.Module):
    def __init__(self):
        super(PD2SEModel, self).__init__()
        res_net_50_base = models.resnet50()
        children_list = list(res_net_50_base.children())

        self.Layer0 = nn.Sequential(*children_list[0:4])
        self.Layer1 = nn.Sequential(children_list[4])
        self.Layer2 = nn.Sequential(children_list[5])
        self.Layer3 = nn.Sequential(children_list[6])
        self.Layer4 = nn.Sequential(children_list[7])
        self.Shuffle1_1 = ShuffleUnit(in_channels=512, out_channels=512, downsample=False,
                                      use_residual=False, use_se=False)
        self.Shuffle1_2 = ShuffleUnit(in_channels=512, out_channels=512, downsample=True,
                                      use_residual=False, use_se=False)
        self.Shuffle1_3 = ShuffleUnit(in_channels=512, out_channels=2048, downsample=True,
                                      use_residual=False, use_se=False)
        self.Shuffle2_1 = ShuffleUnit(in_channels=1024, out_channels=1024, downsample=False,
                                      use_se=False, use_residual=False)
        self.Shuffle2_2 = ShuffleUnit(in_channels=1024, out_channels=1024, downsample=True,
                                      use_se=False, use_residual=False)
        self.Shuffle2_3 = ShuffleUnit(in_channels=1024, out_channels=2048, downsample=True,
                                      use_se=False, use_residual=False)
        self.FC1 = nn.Linear(2048*8*8, 9)
        self.FC2 = nn.Linear(2048*4*4, 27)
        self.FC3 = nn.Linear(2048*8*8, 45)

    def forward(self, x):
        severity_class = self.Layer0(x)
        severity_class = self.Layer1(severity_class)
        plant_class = self.Layer2(severity_class)
        disease_class = self.Layer3(plant_class)
        severity_class = self.Layer4(disease_class)
        plant_class = self.Shuffle1_3(self.Shuffle1_2(self.Shuffle1_1(plant_class)))
        disease_class = self.Shuffle2_3(self.Shuffle2_2(self.Shuffle2_1(disease_class)))
        severity_class = severity_class.view(-1, 2048*8*8)
        plant_class = plant_class.view(-1, 2048*8*8)
        disease_class = disease_class.view(-1, 2048*4*4)
        plant_class = F.softmax(self.FC1(plant_class), dim=1)
        disease_class = F.softmax(self.FC2(disease_class), dim=1)
        severity_class = F.softmax(self.FC3(severity_class), dim=1)
        return [plant_class, disease_class, severity_class]
