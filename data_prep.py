import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import spacy
import pandas as pd
import re
import tqdm
import time
import torchvision
from torch.utils import data
import torchvision.models as models
from torchvision import transforms, utils
import numpy as np
import sys
import cv2
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2
import random

TENSOR_TYPE = 'torch.FloatTensor'


#hyperparams
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 2




class MyDataset(data.Dataset):
    def __init__(self, x, y):
        'Initialization'
        self.x = x
        self.y = y
        self.raw_data = x
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        raw = self.raw_data[index]
        
        return self.x[index], self.y[index]

def load_image(folder, image_path, label, data, labels, device):
    xml_path = folder + 'annotations/' + image_path.split('.')[0] + '.xml'
    if not os.path.exists(xml_path):
        return
    image = cv2.imread(folder + image_path)
    root = ET.parse(xml_path).getroot()

    #convert to tensor
    trans_tensor = transforms.Compose([transforms.ToTensor()])
    # trans_tensor_normalize = transforms.Compose([transforms.ToTensor(),
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    for child in root:
        if child.tag == 'object':
            bbox = [int(child[-1][0].text), int(child[-1][1].text), int(child[-1][2].text), int(child[-1][3].text)]
            eye = cv2.resize(image[bbox[1]:bbox[3], bbox[0]:bbox[2]], IMAGE_SIZE)
            data.append(trans_tensor(eye))
            labels.append(label)

def load_image_numpy(folder, image_path, label, data, labels):
    xml_path = folder + 'annotations/' + image_path.split('.')[0] + '.xml'
    if not os.path.exists(xml_path):
        return
    image = cv2.imread(folder + image_path)
    root = ET.parse(xml_path).getroot()
    for child in root:
        if child.tag == 'object':
            bbox = [int(child[-1][0].text), int(child[-1][1].text), int(child[-1][2].text), int(child[-1][3].text)]
            data.append(cv2.resize(image[bbox[1]:bbox[3], bbox[0]:bbox[2]], IMAGE_SIZE))
            labels.append(label)

def create_train_data(data0, data1, label0, label1, device):
    print("Augmenting and Balancing Dataset")

    # mean_tuple = [0.485, 0.456, 0.406]
    # std_tuple = [0.229, 0.224, 0.225]

    mean_tuple = [68.61011534, 80.11501091, 103.89215555]
    std_tuple = [39.50290728, 39.2612195, 42.93534639]

    adjusted_mean_tuple = [x/255 for x in mean_tuple]
    adjusted_std_tuple = [x/255 for x in std_tuple]

    adjusted_mean_tuple = [0.5, 0.5, 0.5]
    adjusted_std_tuple = [0.5, 0.5, 0.5]

    #convert to tensor then normalize
    trans_tensor_normalize = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=mean_tuple, std=std_tuple)])
    trans_normalize = transforms.Compose([transforms.Normalize(mean=mean_tuple, std=std_tuple)])

    #healthy data
    data0_data = []
    for val in data0:
        #append initial
        data0_data.append(trans_normalize(val))
        #convert to PIL
        img_pil = torchvision.transforms.functional.to_pil_image(val)
        #augment
        # img_pil_transformed = torchvision.transforms.functional.rotate(img_pil, 90)
        # data0_data.append(trans_tensor_normalize(img_pil_transformed))
        img_pil_transformed = torchvision.transforms.functional.rotate(img_pil, 180)
        data0_data.append(trans_tensor_normalize(img_pil_transformed))
        # img_pil_transformed = torchvision.transforms.functional.rotate(img_pil, 270)
        # data0_data.append(trans_tensor_normalize(img_pil_transformed))
    
    print("Healthy Augmented Data Count: ", len(data0_data))

    data0_labels = [torch.from_numpy(np.array([1,0])) for x in range(len(data0_data))]

    #############################################################################################
    #unhealthy data
    data1_data = []
    data1_labels = []
    for val in data1:
        #append initial
        data1_data.append(trans_normalize(val))
        #convert to PIL
        img_pil = torchvision.transforms.functional.to_pil_image(val)
        #augment
        img_pil_transformed = torchvision.transforms.functional.rotate(img_pil, 90)
        data1_data.append(trans_tensor_normalize(img_pil_transformed))
        img_pil_transformed = torchvision.transforms.functional.rotate(img_pil, 180)
        data1_data.append(trans_tensor_normalize(img_pil_transformed))
        img_pil_transformed = torchvision.transforms.functional.rotate(img_pil, 270)
        data1_data.append(trans_tensor_normalize(img_pil_transformed))
        # #changing saturation
        # img_pil_transformed = torchvision.transforms.functional.adjust_saturation(img_pil, 2)
        # data1_data.append(trans_tensor_normalize(img_pil_transformed))
        #change brightness
        img_pil_transformed = torchvision.transforms.functional.adjust_brightness(img_pil, 2)
        data1_data.append(trans_tensor_normalize(img_pil_transformed))
        # #changing contrast
        # img_pil_transformed = torchvision.transforms.functional.adjust_contrast(img_pil, 2)
        # data1_data.append(trans_tensor_normalize(img_pil_transformed))
        # #adjusting gamma
        # img_pil_transformed = torchvision.transforms.functional.adjust_gamma(img_pil, 2, gain=1)
        # data1_data.append(trans_tensor_normalize(img_pil_transformed))
        # img_pil_transformed = torchvision.transforms.functional.adjust_gamma(img_pil, 0.5, gain=1)
        # data1_data.append(trans_tensor_normalize(img_pil_transformed))
        # #adjusting hue, 0.5 complete shift in color scale
        # img_pil_transformed = torchvision.transforms.functional.adjust_hue(img_pil, 0.5)
        # data1_data.append(trans_tensor_normalize(img_pil_transformed))
        # img_pil_transformed = torchvision.transforms.functional.adjust_hue(img_pil, -0.5)
        # data1_data.append(trans_tensor_normalize(img_pil_transformed))
        
    
    print("UnHealthy Augmented Data Count: ", len(data1_data))
    
    data1_labels = [torch.from_numpy(np.array([0,1])) for x in range(len(data1_data))]

    return data0_data + data1_data, data0_labels + data1_labels

#TODO: extract label creation into it's own function for modularity
def create_train_labels():
    print("Creating Labels for Training")

def create_test_data(data0, label0):
    print("Processing Test Data")
    # mean_tuple = [0.485, 0.456, 0.406]
    # std_tuple = [0.229, 0.224, 0.225]

    mean_tuple = [68.61011534, 80.11501091, 103.89215555]
    std_tuple = [39.50290728, 39.2612195, 42.93534639]

    adjusted_mean_tuple = [x/255 for x in mean_tuple]
    adjusted_std_tuple = [x/255 for x in std_tuple]

    adjusted_mean_tuple = [0.5, 0.5, 0.5]
    adjusted_std_tuple = [0.5, 0.5, 0.5]

    #convert to tensor then normalize
    trans_normalize = transforms.Compose([transforms.Normalize(mean=mean_tuple, std=std_tuple)])

    data = []
    labels = []

    for index in range(len(data0)):
        data.append(trans_normalize(data0[index]))
        if label0[index] == 0:
            labels.append(torch.from_numpy(np.array([1,0])))
        else:
            labels.append(torch.from_numpy(np.array([0,1])))

    return data, labels

def load_data():
    print("Loading Dataset")

    #use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    healthy_data, unhealthy_data = [], []
    healthy_labels, unhealthy_labels = [], []

    print('Loading Cancer Images')
    for image_path in os.listdir('data/unhealthy/left/'):
        if '.jpeg' not in image_path:
            continue
        load_image('data/unhealthy/left/', image_path, 1, unhealthy_data, unhealthy_labels, device)

    for image_path in os.listdir('data/unhealthy/right/'):
        if '.jpeg' not in image_path:
            continue
        load_image('data/unhealthy/right/', image_path, 1, unhealthy_data, unhealthy_labels, device)
    
    print('Loading Normal Images')
    for image_path in os.listdir('data/healthy/'):
        if '.jpeg' not in image_path:
            continue
        load_image('data/healthy/', image_path, 0, healthy_data, healthy_labels, device)

    print("Healthy Data Count: ", len(healthy_data))
    print("Unhealthy Data Count: ", len(unhealthy_data))

    train_data, train_labels = create_train_data(healthy_data, unhealthy_data, healthy_labels, unhealthy_labels, device)

    my_train_dataset = MyDataset(train_data, train_labels)
    
    print("Train data Count: ", len(train_data))
    print("Train label Count: ", len(train_labels))


    #load unknown data
    unknown_data = []
    unknown_labels = []
    raw_unknown_data = []
    raw_unknown_labels = []
    for image_path in os.listdir('data/test/unhealthy/'):
        if '.jpeg' not in image_path:
            continue
        load_image('data/test/unhealthy/', image_path, 1, unknown_data, unknown_labels, device)
        load_image('data/test/unhealthy/', image_path, 1, raw_unknown_data, raw_unknown_labels, device)
    for image_path in os.listdir('data/test/healthy/'):
        if '.jpeg' not in image_path:
            continue
        load_image('data/test/healthy/', image_path, 0, unknown_data, unknown_labels, device)
        load_image('data/test/healthy/', image_path, 0, raw_unknown_data, raw_unknown_labels, device)

    test_data, test_labels = create_test_data(unknown_data, unknown_labels)

    print("Test Data Count: ", len(test_data))
    print("Test Label Count: ", len(test_labels))
    
    my_test_dataset = MyDataset(test_data, test_labels)

    return my_train_dataset, my_test_dataset, raw_unknown_data

def normalize_images(data):
    mean = np.mean(data, axis=(0, 1, 2))
    std = np.std(data, axis=(0, 1, 2))
    print("mean ", mean)
    print("std ", std)
    return (data - mean) / std

def calculate_mean():
    print("calculating mean and std")

    #use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    healthy_data, unhealthy_data = [], []
    healthy_labels, unhealthy_labels = [], []

    print('Loading Cancer Images')
    for image_path in os.listdir('data/unhealthy/left/'):
        if '.jpeg' not in image_path:
            continue
        load_image_numpy('data/unhealthy/left/', image_path, 1, unhealthy_data, unhealthy_labels)

    for image_path in os.listdir('data/unhealthy/right/'):
        if '.jpeg' not in image_path:
            continue
        load_image_numpy('data/unhealthy/right/', image_path, 1, unhealthy_data, unhealthy_labels)
    
    print('Loading Normal Images')
    for image_path in os.listdir('data/healthy/'):
        if '.jpeg' not in image_path:
            continue
        load_image_numpy('data/healthy/', image_path, 0, healthy_data, healthy_labels)  
    unknown_data = []
    unknown_labels = []
    for image_path in os.listdir('data/test/unhealthy/'):
        if '.jpeg' not in image_path:
            continue
        load_image_numpy('data/test/unhealthy/', image_path, 1, unknown_data, unknown_labels)
    for image_path in os.listdir('data/test/healthy/'):
        if '.jpeg' not in image_path:
            continue
        load_image_numpy('data/test/healthy/', image_path, 0, unknown_data, unknown_labels)
    
    result = healthy_data + unhealthy_data + unhealthy_data

    print("results shape: ", np.array(result).shape)
    result = normalize_images(result)

if __name__ == "__main__":
    load_data()
    # calculate_mean()

