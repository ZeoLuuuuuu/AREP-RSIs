import torch.nn as nn
import torch

class A_ConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, 6)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(64, 128, 5)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        self.max_pool4 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128, num_classes, 4)
        self.bn5 = nn.BatchNorm2d(num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        s = self.conv1(x)
        s = self.bn1(s)
        s = self.relu1(s)
        s = self.max_pool1(s)
        # print(s.shape)
        s = self.conv2(s)
        s = self.bn2(s)
        s = self.relu2(s)
        s = self.max_pool2(s)
        # print(s.shape)
        s = self.conv3(s)
        s = self.bn3(s)
        s = self.relu3(s)
        s = self.max_pool3(s)
        # print(s.shape)
        s = self.conv4(s)
        s = self.bn4(s)
        s = self.relu4(s)
        s = self.dropout(s)
        s = self.max_pool4(s)
        # print(s.shape)
        s = self.conv5(s)
        s = self.bn5(s)
        sf = torch.flatten(s, 1)
        s = self.softmax(sf)
        # print(s.shape)
        return s

class A_ConvNet_fusarship(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, 6)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(64, 128, 5)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        self.max_pool4 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 128, 7)
        self.bn4_1 = nn.BatchNorm2d(128)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.max_pool4_1 = nn.MaxPool2d(2)


        self.conv5 = nn.Conv2d(128, num_classes, 11)
        self.bn5 = nn.BatchNorm2d(num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        s = self.conv1(x)
        s = self.bn1(s)
        s = self.relu1(s)
        s = self.max_pool1(s)
        # print(s.shape)
        s = self.conv2(s)
        s = self.bn2(s)
        s = self.relu2(s)
        s = self.max_pool2(s)
        # print(s.shape)
        s = self.conv3(s)
        s = self.bn3(s)
        s = self.relu3(s)
        s = self.max_pool3(s)
        # print(s.shape)
        s = self.conv4(s)
        s = self.bn4(s)
        s = self.relu4(s)
        s = self.dropout(s)
        s = self.max_pool4(s)
        # print(s.shape)

        ######################################
        s = self.conv4_1(s)
        s = self.bn4_1(s)
        s = self.relu4_1(s)
        s = self.dropout(s)
        s = self.max_pool4_1(s)
        # print(s.shape)
        ########################################
        s = self.conv5(s)
        s = self.bn5(s)
        sf = torch.flatten(s, 1)
        s = self.softmax(sf)
        # print(s.shape)
        return s
class A_ConvNet_uc(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, 6)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(64, 128, 5)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        self.max_pool4 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.max_pool4_1 = nn.MaxPool2d(2)


        self.conv5 = nn.Conv2d(256, num_classes, 4)
        self.bn5 = nn.BatchNorm2d(num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        s = self.conv1(x)
        s = self.bn1(s)
        s = self.relu1(s)
        s = self.max_pool1(s)
        # print(s.shape)
        s = self.conv2(s)
        s = self.bn2(s)
        s = self.relu2(s)
        s = self.max_pool2(s)
        # print(s.shape)
        s = self.conv3(s)
        s = self.bn3(s)
        s = self.relu3(s)
        s = self.max_pool3(s)
        # print(s.shape)
        s = self.conv4(s)
        s = self.bn4(s)
        s = self.relu4(s)
        s = self.dropout(s)
        s = self.max_pool4(s)
        # print(s.shape)

        ######################################
        s = self.conv4_1(s)
        s = self.bn4_1(s)
        s = self.relu4_1(s)
        s = self.dropout(s)
        s = self.max_pool4_1(s)
        # print(s.shape)
        ########################################
        s = self.conv5(s)
        s = self.bn5(s)
        sf = torch.flatten(s, 1)
        s = self.softmax(sf)
        # print(s.shape)
        return s