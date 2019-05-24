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

def load_image(folder, image_path, label, data, labels):
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

def load_data():
    print("Loading Dataset")

    global TENSOR_TYPE
    if torch.cuda.is_available():
        TENSOR_TYPE = 'torch.cuda.FloatTensor'

    healthy_data, unhealthy_data = [], []
    healthy_labels, unhealthy_labels = [], []

    print('Loading Cancer Images')
    for image_path in os.listdir('data/unhealthy/left/'):
        if '.jpeg' not in image_path:
            continue
        load_image('data/unhealthy/left/', image_path, 1, unhealthy_data, unhealthy_labels)

    for image_path in os.listdir('data/unhealthy/right/'):
        if '.jpeg' not in image_path:
            continue
        load_image('data/unhealthy/right/', image_path, 1, unhealthy_data, unhealthy_labels)
    
    print('Loading Normal Images')
    for image_path in os.listdir('data/healthy/'):
        if '.jpeg' not in image_path:
            continue
        load_image('data/healthy/', image_path, 0, healthy_data, healthy_labels)

