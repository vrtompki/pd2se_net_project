import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.autograd import Variable
import shutil
import json
from PIL import Image
from config import *


def check_dir(new_dir):
    if not os.path.isdir(new_dir):
        try:
            os.mkdir(new_dir)
        except OSError:
            print("Creation of the directory %s failed" % new_dir)
        else:
            print("Directory %s successfully created... " % new_dir)
    else:
        print("Finished checking directory existence...")


def split_data_by(split_by='Severity ID'):
    train_json = TRAIN_ANN_PATH
    val_json = VAL_ANN_PATH
    train_partition = {MAP_61to45[k][split_by]: [] for k in MAP_61to45.keys()}
    val_partition = {MAP_61to45[k][split_by]: [] for k in MAP_61to45.keys()}
    max_num_samples = 100

    # Setup class counters
    if split_by == 'Plant ID':
        sample_counters1 = np.zeros((NUM_CLASSES_1, 1))
        sample_counters2 = np.zeros((NUM_CLASSES_1, 1))
        total_counters1 = np.zeros((NUM_CLASSES_1, 1))
        total_counters2 = np.zeros((NUM_CLASSES_1, 1))
        no_classes = NUM_CLASSES_1
    elif split_by == 'Disease ID':
        sample_counters1 = np.zeros((NUM_CLASSES_2, 1))
        sample_counters2 = np.zeros((NUM_CLASSES_2, 1))
        total_counters1 = np.zeros((NUM_CLASSES_2, 1))
        total_counters2 = np.zeros((NUM_CLASSES_2, 1))
        no_classes = NUM_CLASSES_2
    else:
        sample_counters1 = np.zeros((NUM_CLASSES_3, 1))
        sample_counters2 = np.zeros((NUM_CLASSES_3, 1))
        total_counters1 = np.zeros((NUM_CLASSES_3, 1))
        total_counters2 = np.zeros((NUM_CLASSES_3, 1))
        no_classes = NUM_CLASSES_3

    # Set up plat configurations
    labels = str(np.arange(no_classes))
    x_marks = np.arange(no_classes)
    tick_widths = 0.35

    with open(train_json, mode='rt', encoding='utf-8') as f:
        for _ in json.load(f):
            img_fp = os.path.join(TRAIN_IMG_PATH,
                                  _['image_id']).encode('ascii', 'ignore').decode('utf-8')
            if os.path.exists(img_fp):
                label = _['disease_class']
                if str(label) in MAP_61to45.keys():
                    class_split = MAP_61to45[str(label)][split_by]
                    total_counters1[class_split-1] += 1
                    if sample_counters1[class_split-1] != max_num_samples:
                        train_partition[class_split].append({'image': _['image_id'], 'labels': MAP_61to45[str(label)]})
                        sample_counters1[class_split-1] += 1

                #     if os.path.isdir(TRAIN_IMG_PATH + str(class_split) + '/'):
                #         shutil.copy2(img_fp, TRAIN_IMG_PATH + str(class_split) + '/')
                #     else:
                #         os.mkdir(TRAIN_IMG_PATH + str(class_split) + '/')
                # # imgs.append(img_fp)
                # # lbs.append(_['disease_class'])
                # # tr_lbs.append(_['disease_class'])
    with open(val_json, mode='rt', encoding='utf-8') as f:
        for _ in json.load(f):
            img_fp = os.path.join(VAL_IMG_PATH,
                                  _['image_id']).encode('ascii', 'ignore').decode('utf-8')
            if os.path.exists(img_fp):
                label = _['disease_class']
                if str(label) in MAP_61to45.keys():
                    class_split = MAP_61to45[str(label)][split_by]
                    total_counters2[class_split-1] += 1
                    if sample_counters2[class_split-1] != max_num_samples:
                        val_partition[class_split].append({'image': _['image_id'], 'labels': MAP_61to45[str(label)]})
                        sample_counters2[class_split-1] += 1
    #                 class_split = MAP_61to45[str(label)][split_by]
    #                 check_dir(VAL_IMG_PATH + str(class_split) + '/')
    #                 shutil.copy2(img_fp, VAL_IMG_PATH + str(class_split) + '/')
    # TODO plot the stacked grouped bar graphs, use the image id for key and randomly sample from each class pool
    # TODO plot Sankey diagram and pie chart for class portions and flow from task to task
    # # sub_lbs = []
    # # for idx in range(len(lbs)):
    # #     if str(lbs[idx]) in MAP_61to45.keys():
    # #         lbs[idx] = MAP_61to45[str(lbs[idx])]
    # #         sub_lbs.append(lbs[idx])
    # # train_img_files = imgs
    # # train_labels = sub_lbs
    # # imgs = []
    # # lbs = []
    # #
    # # with open(val_json, mode='rt', encoding='utf-8') as f:
    # #     for _ in json.load(f):
    # #         img_fp = os.path.join(VAL_IMG_PATH,
    # #                               _['image_id']).encode('ascii', 'ignore').decode('utf-8')
    # #         if os.path.exists(img_fp):
    # #             imgs.append(img_fp)
    # #             lbs.append(_['disease_class'])
    # #             val_lbs.append(_['disease_class'])
    # # sub_lbs = []
    # # for idx in range(len(lbs)):
    # #     if str(lbs[idx]) in MAP_61to45.keys():
    # #         lbs[idx] = MAP_61to45[str(lbs[idx])]
    # #         sub_lbs.append(lbs[idx])
    # # val_img_files = imgs
    # # val_labels = sub_lbs
    print()


def plot_weights(model=None, image=None):
    # Plot the weights, plots
    outputs = []
    names = []
    moduleList = list(model.features.modules)
    for layer in moduleList[1:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))

    output_im = []
    for i in outputs:
        i = i.squeeze(0)
        output_im.append(i.cpu().numpy())

    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (30, 50)

    for i in range(len(output_im)):
        a = fig.add_subplot(8, 4, i + 1)
        imgplot = plt.imshow(output_im[i])
        plt.axis('off')
        a.set_title(names[i].partition('(')[0], fontsize=30)

    plt.savefig('layer_outputs.jpg', bbox_inches='tight')
    model.load_saved_states()


def make_heatmap(model, image, true_class, k=8, stride=8):
    """
    Input image is of size (1, c, w, h) typically (1, 3, 224, 224) for vgg16
    true_class is a number corresponding to imagenet classes
    k in the filter size (c, k, k)
    """
    heatmap = torch.zeros(int(((image.shape[2] - k) / stride) + 1), int(((image.shape[3] - k) / stride) + 1))
    image = image.data

    i = 0
    a = 0
    while i <= image.shape[3] - k:
        j = 0
        b = 0
        while j <= image.shape[2] - k:
            h_filter = torch.ones(image.shape)
            h_filter[:, :, j:j + k, i:i + k] = 0
            temp_image = Variable((image.cuda() * h_filter.cuda()).cuda())
            temp_softmax = model(temp_image)
            temp_softmax = torch.nn.functional.softmax(temp_softmax).data[0]
            heatmap[a][b] = temp_softmax[true_class]
            j += stride
            b += 1
        print(a)
        i += stride
        a += 1

    image = image.squeeze()

    true_image = image.transpose(0, 1)
    true_image = true_image.transpose(1, 2)
    # Un-Normalize image
    true_image = true_image * torch.Tensor([0.229, 0.224, 0.225]).cuda() + torch.Tensor([0.485, 0.456, 0.406]).cuda()

    # Plot both images
    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (20, 20)

    a = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(true_image)
    plt.title('Original Image')
    plt.axis('off')

    # Normalize heatmap
    heatmap = heatmap - heatmap.min()
    #     heatmap = heatmap/heatmap.max()
    heatmap = np.uint8(255 * heatmap)

    a = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(heatmap)
    plt.title('Heatmap')
    plt.axis('off')

    return heatmap


def filter_outputs(model, image, layer_to_visualize):
    if layer_to_visualize < 0:
        layer_to_visualize += 31
    output = None
    name = None
    modulelist = list(model.features.modules)
    for count, layer in enumerate(modulelist[1:]):
        image = layer(image)
        if count == layer_to_visualize:
            output = image
            name = str(layer)

    filters = []
    output = output.data.squeeze()
    for i in range(output.shape[0]):
        filters.append(output[i, :, :])

    fig = plt.figure()
    plt.rcParams["figure.figsize"] = (10, 10)

    for i in range(int(np.sqrt(len(filters))) * int(np.sqrt(len(filters)))):
        fig.add_subplot(np.sqrt(len(filters)), np.sqrt(len(filters)), i + 1)
        imgplot = plt.imshow(filters[i])
        plt.axis('off')


def get_confusion_mat(model, dataloaders, nb_classes):
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders['val']):
            inputs = inputs.to(DEVICE)
            classes = classes.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    print(confusion_matrix)
    # per class accuracy
    print(confusion_matrix.diag() / confusion_matrix.sum(1))

def plot_metrics(loss, acc):
    loss


if __name__ == '__main__':
    split_data_by()
