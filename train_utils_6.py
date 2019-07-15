import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
from torchvision import models
from model import PD2SEModel, DiseaseModel, PlantModel
from pytorchcv.models.shufflenetv2 import ShuffleUnit
from config import *
from dataset import PlantsDiseaseDataset
from datetime import datetime
from WindowsInhibitor import WindowsInhibitor
import sys
import os
import time
import copy
import pickle


def train_model(train_val=True):
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    criterion3 = nn.CrossEntropyLoss()

    train_dataset = PlantsDiseaseDataset(train_val='train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               num_workers=4,
                                               drop_last=True,
                                               shuffle=True)
    mod = PD2SEModel()

    opt = torch.optim.SGD(mod.parameters(), lr=1e-4, weight_decay=5e-5, momentum=0.9)
    if USE_GPU:
        torch.cuda.empty_cache()
        torch.cuda.init()
    mod.to(DEVICE)
    criterion1.to(DEVICE)
    criterion2.to(DEVICE)
    criterion3.to(DEVICE)

    best_loss = sys.maxsize
    best_acc = 0

    patience = 0

    running_acc1 = 0
    running_acc2 = 0
    running_acc3 = 0

    running_loss1 = 0
    running_loss2 = 0
    running_loss3 = 0

    train_loss1 = []
    train_loss2 = []
    train_loss3 = []

    val_loss1 = []
    val_loss2 = []
    val_loss3 = []

    train_acc1 = []
    train_acc2 = []
    train_acc3 = []

    val_acc1 = []
    val_acc2 = []
    val_acc3 = []
    print("*******************************************")
    print("*******************************************")
    for epoch in range(NUM_EPOCHS):
        print("\nNow training model.......")
        if epoch % 4 == 0:
            print("\nEpoch " + str(epoch + 1))
        mod.train()
        for batch in tqdm(train_loader, file=sys.stdout):
            img = batch['image']
            target1 = batch['label_1'] - 1
            target2 = batch['label_2'] - 1
            target3 = batch['label_3'] - 1

            img, target1, target2, target3 = img.to(DEVICE), \
                                             target1.to(DEVICE), \
                                             target2.to(DEVICE), \
                                             target3.to(DEVICE)
            out1, out2, out3 = mod(img)
            loss1 = criterion1(out1, target1)
            loss2 = criterion2(out2, target2)
            loss3 = criterion3(out3, target3)
            loss = loss1 + loss2 + loss3
            loss.backward()

            acc1 = torch.sum(torch.argmax(out1, 1) == target1).float() / out1.size()[0]
            acc2 = torch.sum(torch.argmax(out2, 1) == target2).float() / out2.size()[0]
            acc3 = torch.sum(torch.argmax(out3, 1) == target3).float() / out3.size()[0]

            opt.zero_grad()
            opt.step()
        if epoch % 24 == 0:
            print("\nPlant Class Loss: " + str(loss1.item()))
            print("Plant Class Acc.: " + str(acc1.item()))
            print("\nDisease Class Loss: " + str(loss2.item()))
            print("Disease Class Acc: " + str(acc2.item()))
            print("\nSeverity Class Loss: " + str(loss3.item()))
            print("Severity Class Acc: " + str(acc3.item()))
            print("\n*******************************************")
            print("*******************************************")
            print_info = False
        if train_val:
            mode = 'val'


def train_res50_base():
    res50_45 = models.resnet50(pretrained=True)
    num_fts = res50_45.fc.in_features
    res50_45.fc = nn.Linear(num_fts, NUM_CLASSES_3)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(res50_45.parameters(), lr=1e-4, weight_decay=5e-5, momentum=0.9)
    since = time.time()
    best_model_wts = copy.deepcopy(res50_45.state_dict())
    best_acc = 0.0
    patience = 0

    val_confusion_matrix = torch.zeros(NUM_CLASSES_3, NUM_CLASSES_3)
    train_confusion_matrix = torch.zeros(NUM_CLASSES_3, NUM_CLASSES_3)
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    val_losses = []
    best_loss = sys.maxsize
    train_dataset = PlantsDiseaseDataset(train_val='train')
    val_dataset = PlantsDiseaseDataset(train_val='val')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               num_workers=4,
                                               drop_last=True,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=BATCH_SIZE,
                                             num_workers=4,
                                             drop_last=True,
                                             shuffle=True)

    dataset_size = {'train': len(train_dataset),
                    'val': len(val_dataset)}

    loader = {'train': train_loader,
              'val': val_loader}
    if USE_GPU:
        torch.cuda.empty_cache()
        torch.cuda.init()
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    criterion.to(DEVICE)
    res50_45.to(DEVICE)
    for epoch in range(NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
        print('-' * 10)
        for mode in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0

            if mode == 'train':
                res50_45.train()
            else:
                res50_45.eval()

            for batch in tqdm(loader[mode], file=sys.stdout):
                opt.zero_grad()
                _input = batch['image'].to(DEVICE)
                _label = batch['label_3'] - 1
                _label = _label.to(DEVICE)

                with torch.set_grad_enabled(mode == 'train'):
                    out = res50_45(_input)
                    _, predictions = torch.max(out, 1)
                    loss = criterion(out, _label)
                    if mode == 'train':
                        loss.backward()
                        opt.step()

                running_loss += loss.item() * _input.size(0)
                running_corrects += torch.sum(predictions == _label.data)
            epoch_loss = running_loss / dataset_size[mode]
            epoch_acc = running_corrects.double() / dataset_size[mode]
            if mode == 'train':
                train_confusion_matrix[_label.data, predictions] += 1
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)

            else:
                val_confusion_matrix[_label.data, predictions] += 1
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(mode, epoch_loss, epoch_acc))
            if mode == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(res50_45.state_dict())
                    model_stats = 'EPOCH{}_loss_{:.3f}_acc_{:.3f}_'.format(epoch,
                                                                           epoch_loss, epoch_acc)
                    model_time = datetime.now().strftime("%d-%m-%y")
                    model_path = os.path.join(MODEL_DIR, "45_Classes_" + model_stats + model_time)
                    torch.save({'epoch': epoch,
                                'model_state_dict': res50_45.state_dict(),
                                'optimizer_state_dict': opt.state_dict(),
                                'loss': loss}, model_path)
                    print()
                elif epoch_loss >= best_loss:
                    patience += 1
                    print('\nPatience increased...')
                    print('Patience is: {}/{}\n'.format(patience, PATIENCE))
                elif epoch_loss < best_loss:
                    best_loss = epoch_loss
                    if patience is not 0:
                        patience = 0
                        print('\nPatience has been reset....')
                    print('Patience is: {}/{}\n'.format(patience, PATIENCE))
        if patience == PATIENCE:
            print('=' * 40)
            print('Patience tolerance has been met. Saving model and results....')
            break
    stop_time = datetime.now().strftime("%d-%m-%y")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation accuracy was: {:4f}%'.format(best_acc * 100))
    print('Best validation loss was: {:4f}'.format(best_loss))
    pickle.dump(train_losses, open('train_losses_45_' + stop_time + '_.pkl', 'wb'))
    pickle.dump(train_accuracies, open('train_acc_45_' + stop_time + '_.pkl', 'wb'))
    pickle.dump(train_confusion_matrix, open('train_conf_mat_45_' + stop_time + '_.pkl', 'wb'))
    pickle.dump(val_losses, open('val_losses_' + stop_time + '_.pkl', 'wb'))
    pickle.dump(val_accuracies, open('val_acc_' + stop_time + '_.pkl', 'wb'))
    pickle.dump(train_confusion_matrix, open('val_conf_mat' + stop_time + '_.pkl', 'wb'))

    # load best model weights
    res50_45.load_state_dict(best_model_wts)
    return res50_45


def train_res50_27():
    disease_mod = DiseaseModel()
    opt = torch.optim.SGD(disease_mod.parameters(), lr=1e-4, weight_decay=5e-5, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    since = time.time()
    best_model_wts = copy.deepcopy(disease_mod.state_dict())
    best_acc = 0.0
    patience = 0

    val_confusion_matrix = torch.zeros(NUM_CLASSES_2, NUM_CLASSES_2)
    train_confusion_matrix = torch.zeros(NUM_CLASSES_2, NUM_CLASSES_2)
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    val_losses = []
    best_loss = sys.maxsize
    train_dataset = PlantsDiseaseDataset(train_val='train')
    val_dataset = PlantsDiseaseDataset(train_val='val')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               num_workers=4,
                                               drop_last=True,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=BATCH_SIZE,
                                             num_workers=4,
                                             drop_last=True,
                                             shuffle=True)

    dataset_size = {'train': len(train_dataset),
                    'val': len(val_dataset)}

    loader = {'train': train_loader,
              'val': val_loader}
    if USE_GPU:
        torch.cuda.empty_cache()
        torch.cuda.init()
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    criterion.to(DEVICE)
    disease_mod.to(DEVICE)
    for epoch in range(NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
        print('-' * 10)
        for mode in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0

            if mode == 'train':
                disease_mod.train()
            else:
                disease_mod.eval()

            for batch in tqdm(loader[mode], file=sys.stdout):
                opt.zero_grad()
                _input = batch['image'].to(DEVICE)
                _label = batch['label_2'] - 1
                _label = _label.to(DEVICE)

                with torch.set_grad_enabled(mode == 'train'):
                    out = disease_mod(_input)
                    _, predictions = torch.max(out, 1)
                    loss = criterion(out, _label)
                    if mode == 'train':
                        loss.backward()
                        opt.step()

                running_loss += loss.item() * _input.size(0)
                running_corrects += torch.sum(predictions == _label.data)
            epoch_loss = running_loss / dataset_size[mode]
            epoch_acc = running_corrects.double() / dataset_size[mode]
            if mode == 'train':
                train_confusion_matrix[_label.data, predictions] += 1
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)

            else:
                val_confusion_matrix[_label.data, predictions] += 1
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(mode, epoch_loss, epoch_acc))
            if mode == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(disease_mod.state_dict())
                    model_stats = 'EPOCH{}_loss_{:.3f}_acc_{:.3f}_'.format(epoch,
                                                                           epoch_loss, epoch_acc)
                    model_time = datetime.now().strftime("%d-%m-%y")
                    model_path = os.path.join(MODEL_DIR, "27_Classes_" + model_stats + model_time)
                    torch.save({'epoch': epoch,
                                'model_state_dict': disease_mod.state_dict(),
                                'optimizer_state_dict': opt.state_dict(),
                                'loss': loss}, model_path)
                print()
                if epoch_loss >= best_loss:
                    patience += 1
                    print('Patience increased...')
                    print('Patience is: {}/{}\n'.format(patience, PATIENCE))
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    if patience is not 0:
                        patience = 0
                        print(' Patience has been reset....')
                    print('Patience is: {}/{}\n'.format(patience, PATIENCE))
        if patience == PATIENCE:
            print('=' * 40)
            print('Patience tolerance has been met. Saving model and results....')
            break
    stop_time = datetime.now().strftime("%d-%m-%y")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation accuracy was: {:4f}%'.format(best_acc * 100))
    print('Best validation loss was: {:4f}'.format(best_loss))
    pickle.dump(train_losses, open('train_losses_27_' + stop_time + '_.pkl', 'wb'))
    pickle.dump(train_accuracies, open('train_acc_27_' + stop_time + '_.pkl', 'wb'))
    pickle.dump(train_confusion_matrix, open('train_conf_mat_27_' + stop_time + '_.pkl', 'wb'))
    pickle.dump(val_losses, open('val_losses_27_' + stop_time + '_.pkl', 'wb'))
    pickle.dump(val_accuracies, open('val_acc_27_' + stop_time + '_.pkl', 'wb'))
    pickle.dump(train_confusion_matrix, open('val_conf_mat_27_' + stop_time + '_.pkl', 'wb'))

    # load best model weights
    disease_mod.load_state_dict(best_model_wts)
    return disease_mod


def train_res50_9():
    plant_mod = PlantModel()
    opt = torch.optim.SGD(plant_mod.parameters(), lr=1e-4, weight_decay=5e-5, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    since = time.time()
    best_model_wts = copy.deepcopy(plant_mod.state_dict())
    best_acc = 0.0
    patience = 0

    val_confusion_matrix = torch.zeros(NUM_CLASSES_1, NUM_CLASSES_1)
    train_confusion_matrix = torch.zeros(NUM_CLASSES_1, NUM_CLASSES_1)
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    val_losses = []
    best_loss = sys.maxsize
    train_dataset = PlantsDiseaseDataset(train_val='train')
    val_dataset = PlantsDiseaseDataset(train_val='val')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               num_workers=4,
                                               drop_last=True,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=BATCH_SIZE,
                                             num_workers=4,
                                             drop_last=True,
                                             shuffle=True)

    dataset_size = {'train': len(train_dataset),
                    'val': len(val_dataset)}

    loader = {'train': train_loader,
              'val': val_loader}
    if USE_GPU:
        torch.cuda.empty_cache()
        torch.cuda.init()
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    criterion.to(DEVICE)
    plant_mod.to(DEVICE)
    for epoch in range(NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
        print('-' * 10)
        for mode in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0

            if mode == 'train':
                plant_mod.train()
            else:
                plant_mod.eval()

            for batch in tqdm(loader[mode], file=sys.stdout):
                opt.zero_grad()
                _input = batch['image'].to(DEVICE)
                _label = batch['label_1'] - 1
                _label = _label.to(DEVICE)

                with torch.set_grad_enabled(mode == 'train'):
                    out = plant_mod(_input)
                    _, predictions = torch.max(out, 1)
                    loss = criterion(out, _label)
                    if mode == 'train':
                        loss.backward()
                        opt.step()

                running_loss += loss.item() * _input.size(0)
                running_corrects += torch.sum(predictions == _label.data)
            epoch_loss = running_loss / dataset_size[mode]
            epoch_acc = running_corrects.double() / dataset_size[mode]
            if mode == 'train':
                train_confusion_matrix[_label.data, predictions] += 1
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)

            else:
                val_confusion_matrix[_label.data, predictions] += 1
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(mode, epoch_loss, epoch_acc))
            if mode == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(plant_mod.state_dict())
                    model_stats = 'EPOCH{}_loss_{:.3f}_acc_{:.3f}_'.format(epoch,
                                                                           epoch_loss, epoch_acc)
                    model_time = datetime.now().strftime("%d-%m-%y")
                    model_path = os.path.join(MODEL_DIR, "9_Classes_" + model_stats + model_time)
                    torch.save({'epoch': epoch,
                                'model_state_dict': plant_mod.state_dict(),
                                'optimizer_state_dict': opt.state_dict(),
                                'loss': loss}, model_path)
                print()
                if epoch_loss >= best_loss:
                    patience += 1
                    print('Patience increased...')
                    print('Patience is: {}/{}\n'.format(patience, PATIENCE))
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    if patience is not 0:
                        patience = 0
                        print(' Patience has been reset....')
                    print('Patience is: {}/{}\n'.format(patience, PATIENCE))
        if patience == PATIENCE:
            print('=' * 40)
            print('Patience tolerance has been met. Saving model and results....')
            break
    stop_time = datetime.now().strftime("%d-%m-%y")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation accuracy was: {:4f}%'.format(best_acc * 100))
    print('Best validation loss was: {:4f}'.format(best_loss))
    pickle.dump(train_losses, open('train_losses_9_' + stop_time + '_.pkl', 'wb'))
    pickle.dump(train_accuracies, open('train_acc_9_' + stop_time + '_.pkl', 'wb'))
    pickle.dump(train_confusion_matrix, open('train_conf_mat_9_' + stop_time + '_.pkl', 'wb'))
    pickle.dump(val_losses, open('val_losses_9_' + stop_time + '_.pkl', 'wb'))
    pickle.dump(val_accuracies, open('val_acc_9_' + stop_time + '_.pkl', 'wb'))
    pickle.dump(train_confusion_matrix, open('val_conf_mat_9_' + stop_time + '_.pkl', 'wb'))

    # load best model weights
    plant_mod.load_state_dict(best_model_wts)
    return plant_mod


if __name__ == '__main__':

    osSleep = None
    # in Windows, prevent the OS from sleeping while we run
    if os.name == 'nt':
        osSleep = WindowsInhibitor()
        osSleep.inhibit()
    # base_mod = train_res50_base()
    second_phase_mod = train_res50_27()
    if osSleep:
        osSleep.uninhibit()
    # train_model()
