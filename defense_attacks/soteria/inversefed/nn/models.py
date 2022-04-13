"""Define basic models and translate some torchvision stuff."""
"""Stuff from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py."""
import torch
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import AvgPool2d
import torchvision
import torch.nn as nn



from collections import OrderedDict
import numpy as np
from ..utils import set_random_seed




def construct_model(model, num_classes=10, seed=None, num_channels=3, modelkey=None):
    """Return various models."""
    if modelkey is None:
        if seed is None:
            model_init_seed = np.random.randint(0, 2**16 - 10)
        else:
            model_init_seed = seed
    else:
        model_init_seed = modelkey
    set_random_seed(model_init_seed)

    if model == "ConvBig":
        model = RefConvLarge()
    elif model in ['ConvNet', 'ConvNet64']:
        model = ConvNet(width=64, num_channels=num_channels, num_classes=num_classes)
    elif model == 'LeNet':
        model = LeNet(num_channels=num_channels, num_classes=num_classes)
    else:
        raise NotImplementedError('Model not implemented.')

    print(f'Model initialized with random key {model_init_seed}.')
    return model, model_init_seed



class ConvNet(torch.nn.Module):
    """ConvNetBN."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super().__init__()
        self.model = torch.nn.Sequential(OrderedDict([
            ('conv0', torch.nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
            ('bn0', torch.nn.BatchNorm2d(1 * width)),
            ('relu0', torch.nn.ReLU()),

            ('conv1', torch.nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn1', torch.nn.BatchNorm2d(2 * width)),
            ('relu1', torch.nn.ReLU()),

            ('conv2', torch.nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn2', torch.nn.BatchNorm2d(2 * width)),
            ('relu2', torch.nn.ReLU()),

            ('conv3', torch.nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn3', torch.nn.BatchNorm2d(4 * width)),
            ('relu3', torch.nn.ReLU()),

            ('conv4', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn4', torch.nn.BatchNorm2d(4 * width)),
            ('relu4', torch.nn.ReLU()),

            ('conv5', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn5', torch.nn.BatchNorm2d(4 * width)),
            ('relu5', torch.nn.ReLU()),

            ('pool0', torch.nn.MaxPool2d(3)),

            ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', torch.nn.BatchNorm2d(4 * width)),
            ('relu6', torch.nn.ReLU()),

            ('conv6', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', torch.nn.BatchNorm2d(4 * width)),
            ('relu6', torch.nn.ReLU()),

            ('conv7', torch.nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn7', torch.nn.BatchNorm2d(4 * width)),
            ('relu7', torch.nn.ReLU()),

            ('pool1', torch.nn.MaxPool2d(3)),
            ('flatten', torch.nn.Flatten())
            #('linear', torch.nn.Linear(36 * width, num_classes))
        ]))
        self.linear = torch.nn.Linear(36 * width, num_classes)
        self.feature = None

    def forward(self, input):
        self.feature = self.model(input)
        return self.linear(self.feature)

    def extract_feature(self):
        return self.feature


class LeNet(nn.Module):
    def __init__(self, num_classes=10, num_channels=3):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(num_channels, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, num_classes),
            #act(),
            #nn.Linear(256, 100)
        )
        self.feature = None
        
    def forward(self, x):
        out = self.body(x)
        self.feature = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(self.feature)
        return out

    def extract_feature(self):
        return self.feature

class CustomConv(nn.Module):
    """Custom CNN-Class for our benchmarking"""
    def __init__(self, conv_layers, linear_layers, in_channels=3, classes=10):
        super().__init__()
        conv_layers_ = []
        
        for (c, k_x, k_y, s) in conv_layers:
            conv_layers_.append(nn.Conv2d(in_channels=in_channels, out_channels=c, kernel_size=(k_x,k_y), stride=(s,s), padding=0))
            conv_layers_.append(nn.ReLU())
            in_channels = c

        lin_layers_ = []
        for (l_in, l_out) in linear_layers:
            lin_layers_.append(nn.Linear(in_features=l_in, out_features=l_out))
            lin_layers_.append(nn.ReLU())
        # Models here don't use softmax 
        lin_layers_ = lin_layers_[:-1]

        self.conv_layers = nn.Sequential(*conv_layers_)
        self.lin_layers = nn.Sequential(*lin_layers_)
        self.feature = None

    def forward(self, x):
        x = self.conv_layers(x)
        self.feature = torch.flatten(x, 1)
        return self.lin_layers(self.feature)

    def extract_feature(self):
        return self.feature

class RefConv(nn.Module):
    """Benchmarking class
    """
    def __init__(self):
        super().__init__()
        conv_layers_ = [
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2), stride=2)
        ]
    
        lin_layers_1_ = [nn.Linear(in_features=5184, out_features=100), nn.ReLU()]
        lin_layers_2_ = [
            nn.Linear(in_features=100, out_features=10),
        ]
        # Models here don't use softmax 
        # lin_layers_[-1] = nn.Softmax()

        self.conv_layers = nn.Sequential(*conv_layers_)
        self.lin_layers_1 = nn.Sequential(*lin_layers_1_)
        self.lin_layers_2 = nn.Sequential(*lin_layers_2_)
        self.feature = None

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        self.feature = self.lin_layers_1(x)
        return self.lin_layers_2(self.feature)

    def extract_feature(self):
        return self.feature

class RefConvLarge(nn.Module):
    """Benchmarking class
    """
    def __init__(self):
        super().__init__()
        conv_layers_ = [
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2), stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2,2), stride=2)
        ]
    
        lin_layers_1_ = [nn.Linear(in_features=5184, out_features=2000), nn.ReLU(),]
        lin_layers_2_ = [
            nn.Linear(in_features=2000, out_features=1000), nn.ReLU(),
            nn.Linear(in_features=1000, out_features=10),
        ]

        self.conv_layers = nn.Sequential(*conv_layers_)
        self.lin_layers_1 = nn.Sequential(*lin_layers_1_)
        self.lin_layers_2 = nn.Sequential(*lin_layers_2_)
        self.feature = None

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        self.feature = self.lin_layers_1(x)
        return self.lin_layers_2(self.feature)

    def extract_feature(self):
        return self.feature