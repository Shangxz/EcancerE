import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
import os as os
import json

from load_data import load_data

import data_prep

def train(trainloader, unknownloader, net, criterion, optimizer, device, num_epoch):
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        net.train()
        start = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            # print("images", images)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)

            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print(loss.item())

        end = time.time()
        print('[epoch %d] loss: %.3f eplased time %.3f' %
                (epoch + 1, running_loss / 100, end-start))
        start = time.time()
        running_loss = 0.0
        # validate(testloader, net, device)
        test(unknownloader, net, device)

        
    print('Finished Training')

def validate(testloader, net, device):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.max(labels, 1)[1]).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

def test(testloader, net, device):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.max(labels, 1)[1]).sum().item()
    print('Accuracy of the network on the unknown images: %d %%' % (
        100 * correct / total))
    #save state_dict
    if (int(100 * correct / total) > 85):
        torch.save(net.state_dict(), "./checkpoints/" + str(int(100 * correct / total)))
    #save entire model
    # torch.save(net, "checkpoints/" + str(100 * correct / total))

def demo_test(testloader, raw_data, net, device):
    correct = 0
    total = 0
    net.eval()
    # counter = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            print(outputs)
            _, predicted = torch.max(outputs.data, 1)

            for index in range(len(raw_data)):
                img = raw_data[index].permute(1,2,0)
                image = cv2.resize(img.numpy(), (0,0), fx=3, fy=3)
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

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def main():
    ## Load Config file, make sure type of object are correct.
    config:json = {}
    if len(sys.argv) < 2: 
        print("No argv provided, using DEFAULT config profile.")
        with open('./config/default.json') as jsonConfig:  
            config = json.load(jsonConfig)
    else:
        print("Using ", sys.argv[1] + " config profile.")
        with open('./config/' + sys.argv[1] + '.json') as jsonConfig:  
            config = json.load(jsonConfig)
    
    # get cuda device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## Create checkpoint path
    if not os.path.exists("./checkpoints"):
        os.makedirs("./checkpoints")

    #default hyperparams
    num_epoch = config['batch_size']
    batch_size = config['num_epoch']

    if config['cnn_type'] == "resnet50":
        model = models.resnet50(pretrained=True).to(device)
        # model = models.resnet50(pretrained=False).to(device)
        # feature extraction, disable to finetune whole model
        # for name, param in model.named_parameters():
            # if ("layer4" not in name):
                # if ("layer3" not in name):
                    # if ("layer2" not in name):
            # param.requires_grad = False
            # print(name)

        num_ftrs = model.fc.in_features

        model.fc = nn.Sequential(
                # nn.Dropout(0.5),
                nn.Linear(num_ftrs, 2)
            ).to(device)

        # params_to_update = []

        # for name, param in model.named_parameters():
        #     if param.requires_grad == True:
        #         params_to_update.append(param)
        #         print("\t",name)

        new_train_dataset, new_test_dataset, _ = data_prep.load_data()

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=config["lr"])

        trainloader = torch.utils.data.DataLoader(new_train_dataset, batch_size=config['batch_size'],
                                              shuffle=True)

        unknownloader = torch.utils.data.DataLoader(new_test_dataset, batch_size=22,
                                            shuffle=False)

        model.train()
        train(trainloader, unknownloader, model, criterion, optimizer, device, config['num_epoch'])
        exit()

    elif config['cnn_type'] == "demo":
        print("Demo Time!")
        model = models.resnet50(pretrained=False).to(device)
        # model = models.vgg11_bn(pretrained=True).to(device)

        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
                # nn.Dropout(0.5),
                nn.Linear(num_ftrs, 2)
            ).to(device)
        # num_ftrs = model.classifier[6].in_features
        # model.classifier[6] = nn.Linear(num_ftrs, 2).to(device)

        #loading the 99% accurate one
        if torch.cuda.is_available():
            model.load_state_dict(torch.load('checkpoints/100'))
        else:
            model = torch.load('checkpoints/100', map_location='cpu')

        new_train_dataset, new_test_dataset, raw_unknown_data = data_prep.load_data()

        unknownloader = torch.utils.data.DataLoader(new_test_dataset, batch_size=22,
                                            shuffle=False)
        
        # my_raw_unknown_dataset = MyDataset(raw_unknown_data, unknown_labels, 224)
        # raw_unknownloader = torch.utils.data.DataLoader(my_unknown_dataset, batch_size=5,
        #                             shuffle=False)
        
        demo_test(unknownloader, raw_unknown_data, model, device)
        exit()
    
    else:
        print("No Model Provided!")
        exit()

if __name__ == "__main__":
    main()


