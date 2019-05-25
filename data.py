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
IMAGE_SIZE = (100, 48)
NUM_CLASSES = 2




class MyDataset(data.Dataset):
    def __init__(self, x, y, scale_factor):
        'Initialization'
        self.x = x
        temp_lis = []
        for val in y:
            if val == 1:
                temp_lis.append([0,1])
            else:
                temp_lis.append([1,0])
        one_labels = np.array(temp_lis)
        self.y = one_labels
        self.scale_factor = scale_factor
        self.raw_data = x
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        images = torch.from_numpy(self.x[index]).type(TENSOR_TYPE)
        images = images.permute(2,1,0)

        if self.scale_factor != -1:
            target = torch.zeros(3, self.scale_factor, self.scale_factor).type(TENSOR_TYPE)
            target[:, :IMAGE_SIZE[0], :IMAGE_SIZE[1]] = images
        else:
            target = images
        
        labels = torch.from_numpy(self.y[index]).type(TENSOR_TYPE)

        raw = self.raw_data[index]
        
        return target, labels, raw

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
            data.append(trans_tensor(eye).to(device))
            labels.append(label)

def create_train_data(data0, data1, label0, label1, device):
    print("Augmenting and Balancing Dataset")

    #convert to tensor then normalize
    trans_tensor_normalize = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    #healthy data
    data0_data = []
    for val in data0:
        #append initial
        data0_data.append(val)
        #convert to PIL
        img_pil = torchvision.transforms.functional.to_pil_image(val)
        #augment
        img_pil_transformed = torchvision.transforms.functional.rotate(img_pil, 90)
        data0_data.append(trans_tensor_normalize(img_pil_transformed))
        img_pil_transformed = torchvision.transforms.functional.rotate(img_pil, 180)
        data0_data.append(trans_tensor_normalize(img_pil_transformed))
        img_pil_transformed = torchvision.transforms.functional.rotate(img_pil, 270)
        data0_data.append(trans_tensor_normalize(img_pil_transformed))
    
    print("Healthy Augmented Data Count: ", len(data0_data))

    data0_labels = [torch.from_numpy(np.array([1,0])).to(device) for x in range(len(data0_data))]

    #############################################################################################
    #unhealthy data
    data1_data = []
    data1_labels = []
    for val in data1:
        #append initial
        data1_data.append(val)
        #convert to PIL
        img_pil = torchvision.transforms.functional.to_pil_image(val)
        #augment
        img_pil_transformed = torchvision.transforms.functional.rotate(img_pil, 90)
        data1_data.append(trans_tensor_normalize(img_pil_transformed))
        img_pil_transformed = torchvision.transforms.functional.rotate(img_pil, 180)
        data1_data.append(trans_tensor_normalize(img_pil_transformed))
        img_pil_transformed = torchvision.transforms.functional.rotate(img_pil, 270)
        data1_data.append(trans_tensor_normalize(img_pil_transformed))
        #changing saturation
        img_pil_transformed = torchvision.transforms.functional.adjust_saturation(img_pil, 2)
        data1_data.append(trans_tensor_normalize(img_pil_transformed))
        #change brightness
        img_pil_transformed = torchvision.transforms.functional.adjust_brightness(img_pil, 2)
        data1_data.append(trans_tensor_normalize(img_pil_transformed))
        #changing contrast
        img_pil_transformed = torchvision.transforms.functional.adjust_contrast(img_pil, 2)
        data1_data.append(trans_tensor_normalize(img_pil_transformed))
        #adjusting gamma
        img_pil_transformed = torchvision.transforms.functional.adjust_gamma(img_pil, 2, gain=1)
        data1_data.append(trans_tensor_normalize(img_pil_transformed))
        img_pil_transformed = torchvision.transforms.functional.adjust_gamma(img_pil, 0.5, gain=1)
        data1_data.append(trans_tensor_normalize(img_pil_transformed))
        #adjusting hue, 0.5 complete shift in color scale
        img_pil_transformed = torchvision.transforms.functional.adjust_hue(img_pil, 0.5)
        data1_data.append(trans_tensor_normalize(img_pil_transformed))
        img_pil_transformed = torchvision.transforms.functional.adjust_hue(img_pil, -0.5)
        data1_data.append(trans_tensor_normalize(img_pil_transformed))
        
    
    print("UnHealthy Augmented Data Count: ", len(data1_data))
    
    data1_labels = [torch.from_numpy(np.array([0,1])).to(device) for x in range(len(data1_data))]

    return data0_data + data1_data, data0_labels + data1_labels

#TODO: extract label creation into it's own function for modularity
def create_train_labels():
    print("Creating Labels for Training")

def create_test_data(data0, label0):
    print("Processing Test Data")

    return [], []



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
    



if __name__ == "__main__":
    load_data()

