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

from load_data import load_data


class SimpleCNN (nn.Module):

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Linear(14720, 1024)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x

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
            # print(self.x[index])
            # images = torch.from_numpy(np.ascontiguousarray(self.x[index])).type('torch.cuda.FloatTensor')
            # cv2.namedWindow('original_numpy', cv2.WINDOW_NORMAL)
            # cv2.imshow('original_numpy', self.x[index])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # images = torch.from_numpy(np.flip(self.x[index], axis=0).copy()).type('torch.cuda.FloatTensor')
            # images = np.ascontiguousarray(self.x[index].transpose(2,0,1)[np.newaxis, :])
            images = torch.from_numpy(self.x[index]).type('torch.cuda.FloatTensor')
            # cv2.namedWindow('original', cv2.WINDOW_NORMAL)
            # cv2.imshow('original', images.cpu().numpy())
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # plt.imshow(images.cpu().numpy()/255)
            # plt.show()
            # print(images.shape)
            # images = images.squeeze()
            # print(images.shape)
            # exit()
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

        # print(target.shape)
        # temp = target.permute(2,1,0)
        # cv2.namedWindow('target', cv2.WINDOW_NORMAL)
        # cv2.imshow('target', temp.cpu().numpy())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        if self.raw_data != []:
            raw = self.raw_data[index]
        else:
            raw = []
        
        return target, labels, raw

def train(trainloader, testloader, unknownloader, net, criterion, optimizer, device, num_epoch):
    
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        net.train()
        start = time.time()
        running_loss = 0.0
        for i, (images, labels, _) in enumerate(trainloader):
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
        validate(testloader, net, device)
        test(unknownloader, net, device)

        
    print('Finished Training')

def validate(testloader, net, device):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels, _ = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.max(labels, 1)[1]).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    #save state_dict
    # torch.save(net.state_dict(), "checkpoints/" + str(int(100 * correct / total)))
    #save entire model
    # torch.save(net, "checkpoints/" + str(100 * correct / total))

def test(testloader, net, device):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels, _ = data
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
        torch.save(net.state_dict(), "checkpoints/" + str(int(100 * correct / total)))
    #save entire model
    # torch.save(net, "checkpoints/" + str(100 * correct / total))

def demo_test(testloader, raw_data, net, device):
    correct = 0
    total = 0
    net.eval()
    # counter = 0
    with torch.no_grad():
        for data in testloader:
            images, labels, raw = data
            # raw = raw_data[counter]
            # counter += 1
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            # print(raw.shape)
            print(predicted)

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
            # print(outputs)
            # print(labels)
            

    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def main():
    #gpu device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data, train_labels, test_data, test_labels, unknown_data, unknown_labels, raw_unknown_data = load_data()
    print("# of 1s in train", train_labels.count(1))
    print("# of 0s in train", train_labels.count(0))
    # print("# of 1s in test", test_labels.count(1))
    # print("# of 0s in test", test_labels.count(0))

    # for image in train_data:
    #     cv2.namedWindow('target', cv2.WINDOW_NORMAL)
    #     cv2.imshow('target', image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # cv2.imshow('d', data[0])
    # cv2.waitKey(0)
    # exit()

    #default hyperparams
    num_epoch = 50
    batch_size = 100

    if sys.argv[1] == "simple":
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)
        my_train_dataset = MyDataset(train_data, train_labels, -1)
        my_test_dataset = MyDataset(test_data, test_labels, -1)
    
    elif sys.argv[1] == "resnet18":
        batch_size = 64
        num_epoch = 100
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
        model.fc = nn.Linear(num_ftrs, 2).to(device)

        params_to_update = []

        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)

        my_train_dataset = MyDataset(train_data, train_labels, 224)
        my_test_dataset = MyDataset(test_data, test_labels, 224)
        my_unknown_dataset = MyDataset(unknown_data, unknown_labels, 224)

        # aug = transforms.Compose([
        #     transforms.RandomHorizontalFlip(p=1),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #     transforms.RandomVerticalFlip(p=1)
        # ])
        # my_train_dataset = aug(my_train_dataset)

        unknownloader = torch.utils.data.DataLoader(my_unknown_dataset, batch_size=5,
                                            shuffle=False)

        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(params_to_update, lr = 0.003, momentum= 0.9)
        optimizer = optim.Adam(params_to_update, lr=0.001)
        # optimizer = optim.Adam(model.parameters(), lr=0.0001)

        trainloader = torch.utils.data.DataLoader(my_train_dataset, batch_size=batch_size,
                                              shuffle=True)
        testloader = torch.utils.data.DataLoader(my_test_dataset, batch_size=batch_size,
                                              shuffle=False)
        
        model.train()
        train(trainloader, testloader, unknownloader, model, criterion, optimizer, device, num_epoch)
        exit()
       
    elif sys.argv[1] == "resnet34":
        model = models.resnet34(pretrained=True).to(device)
        #feature extraction
        # for param in model.parameters():
        #     param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2).to(device)
        my_train_dataset = MyDataset(train_data, train_labels, 224)
        my_test_dataset = MyDataset(test_data, test_labels, 224)
        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum= 0.9)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    elif sys.argv[1] == "vgg11":
        num_epoch = 1
        model = models.vgg11_bn(pretrained=True).to(device)
        # for param in model.parameters():
        #     param.requires_grad = False
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 2).to(device)
        my_train_dataset = MyDataset(train_data, train_labels, 224)
        my_test_dataset = MyDataset(test_data, test_labels, 224)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.003)

    elif sys.argv[1] == "demo":
        print("Demo Time!")
        model = models.resnet50(pretrained=False).to(device)
        # model = models.vgg11_bn(pretrained=True).to(device)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2).to(device)
        # num_ftrs = model.classifier[6].in_features
        # model.classifier[6] = nn.Linear(num_ftrs, 2).to(device)

        #loading the 99% accurate one
        if torch.cuda.is_available():
            model.load_state_dict(torch.load('checkpoints/90'))
        else:
            model = torch.load('checkpoints/79', map_location='cpu')

        my_unknown_dataset = MyDataset(unknown_data, unknown_labels, 224, raw_unknown_data)
        unknownloader = torch.utils.data.DataLoader(my_unknown_dataset, batch_size=5,
                                            shuffle=False)
        
        # my_raw_unknown_dataset = MyDataset(raw_unknown_data, unknown_labels, 224)
        # raw_unknownloader = torch.utils.data.DataLoader(my_unknown_dataset, batch_size=5,
        #                             shuffle=False)
        
        demo_test(unknownloader, raw_unknown_data, model, device)
        exit()
    
    else:
        print("No Model Provided!")
        exit()
    
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.NLLLoss()

    # optimizer = optim.Adam(model.parameters(), lr=0.003)
    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.003, momentum= 0.9)

    trainloader = torch.utils.data.DataLoader(my_train_dataset, batch_size=batch_size,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(my_test_dataset, batch_size=batch_size,
                                              shuffle=False)

    model.train()
    train(trainloader, testloader, unknownloader, model, criterion, optimizer, device, num_epoch)

    # model.eval()
    # test(testloader, model, device)


if __name__ == "__main__":
    main()


