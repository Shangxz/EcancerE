import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext
from torchtext import data
import spacy
import pandas as pd
import re
import tqdm
import time
from torch.utils import data
import torchvision.models as models
from torchvision import transforms, utils
import numpy as np
import sys
import cv2
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from load_data import load_data


class MyDataset(data.Dataset):
    def __init__(self, x, y, scale_factor, raw_data=[]):
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
        self.raw_data = raw_data
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        if torch.cuda.is_available():
            images = torch.from_numpy(self.x[index]).type('torch.cuda.FloatTensor')
            images = images.permute(2,1,0)

            if self.scale_factor != -1:
                target = torch.zeros(3, self.scale_factor, self.scale_factor).type('torch.cuda.FloatTensor')
                target[:, :100, :48] = images
            else:
                target = images
            
            labels = torch.from_numpy(self.y[index]).type('torch.cuda.LongTensor')
        else:
            images = torch.from_numpy(self.x[index]).type('torch.FloatTensor')
            images = images.permute(2,1,0)

            if self.scale_factor != -1:
                target = torch.zeros(3, self.scale_factor, self.scale_factor).type('torch.FloatTensor')
                target[:, :100, :48] = images
            else:
                target = images
            
            labels = torch.from_numpy(self.y[index]).type('torch.LongTensor')
        
        if self.raw_data != []:
            raw = self.raw_data[index]
        else:
            raw = []
        
        return target, labels, raw


def demo_test(testloader, raw_data, net, device):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels, raw = data
            images = images.to(device)
            labels = labels.to(device)
            print(images.shape)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            print(outputs)

            for index in range(raw.shape[0]):
                image = cv2.resize(raw[index].numpy(), (0,0), fx=3, fy=3)
                truth = ''

                if (labels.data.cpu().numpy()[index][0] == 1):
                    truth = "0"
                else:
                    truth = "1"

                cv2.namedWindow('Cancer?: ' + str(predicted.data.cpu().numpy()[index]) + " Label: " + truth, cv2.WINDOW_NORMAL)
                cv2.imshow('Cancer?: ' + str(predicted.data.cpu().numpy()[index])  + " Label: " + truth, image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            total += labels.size(0)
            correct += (predicted == torch.max(labels, 1)[1]).sum().item()
            

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

def main():
    print("Demo Time!")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, train_labels, test_data, test_labels, unknown_data, unknown_labels, raw_unknown_data = load_data()
    model = models.resnet50(pretrained=False).to(device)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2).to(device)

    #loading the 90% accurate one
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('checkpoints/90'))
    else:
        model = torch.load('checkpoints/90', map_location='cpu')

    my_unknown_dataset = MyDataset(unknown_data, unknown_labels, 224, raw_unknown_data)
    unknownloader = torch.utils.data.DataLoader(my_unknown_dataset, batch_size=5,
                                        shuffle=False)

    # my_raw_unknown_dataset = MyDataset(raw_unknown_data, unknown_labels, 224)
    # raw_unknownloader = torch.utils.data.DataLoader(my_unknown_dataset, batch_size=5,
    #                             shuffle=False)

    demo_test(unknownloader, raw_unknown_data, model, device)



if __name__ == "__main__":
    main()