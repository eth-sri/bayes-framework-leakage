import numpy as np
from pprint import pprint

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data

from sklearn.model_selection import train_test_split

import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d","--directory", type=str, default='./models', required=False, help="Where to save models")
    parser.add_argument("-w","--weight_init", type=float, default=0.05, required=False, help="Weight initialization. Important - too small causes divergences, too high and you get infinite gradients")
    parser.add_argument("-e","--epochs", type=int, default=100, required=False, help="Number of epochs to train for")
    parser.add_argument("-i","--iters", type=int, default=1e9, required=False, help="Number of iterations to train for")
    parser.add_argument("-n", "--network", type=str, choices=['ffn', 'lenet'], default='ffn', help="What network to train")
    parser.add_argument("-a", "--activation", type=str, choices=['ReLU', 'Tanh', 'Sigmoid'], default='ReLU', help="What activation to use")
    parser.add_argument("-ps","--precode_size", type=int, default=256, required=False, help="Number of hidden units in PRECODE layer")

    return parser.parse_args()

args = get_args()

if not os.path.exists(args.directory):
    os.makedirs(args.directory)


torch.manual_seed(50)

transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                             ])
trainset = torchvision.datasets.CIFAR10(root='~/.torch', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=8)
testset = torchvision.datasets.CIFAR10(root='~/.torch', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=8)

n_classes=10

import inversefed
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative')

def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-args.weight_init, args.weight_init)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-args.weight_init, args.weight_init)

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

class LeNet_PRECODE(nn.Module):
    # https://arxiv.org/pdf/2108.04725.pdf
    def __init__(self, num_classes, hidden_size, activation):
        super(LeNet_PRECODE, self).__init__()
        act = getattr(nn, activation)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Flatten(),
            nn.Linear(768, hidden_size),
            act(),
        )
        self.hidden2mu = nn.Linear(hidden_size, hidden_size)
        self.hidden2log_var = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 768), act(),
            nn.Linear(768, num_classes)
        )
        self.hidden_size = hidden_size
    def encode(self, x):
        hidden = self.encoder(x)
        self.hidden = hidden
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)
        return mu, log_var
    
    def decode(self, x):
        x = self.decoder(x)
        return x
    
    def reparametrize(self, mu, log_var, z):
        sigma = torch.exp(0.5*log_var)
        self.sigma = sigma
        return mu + sigma*z
    
    def forward(self, x, z):
        mu, log_var = self.encode(x)
        self.log_var = log_var
        self.mu = log_var
        hidden = self.reparametrize(mu, log_var, z)
        output = self.decoder(hidden)
        return output
    
class FFN_PRECODE(nn.Module):
    # https://arxiv.org/pdf/2108.04725.pdf
    def __init__(self, num_classes, hidden_size, activation):
        super(FFN_PRECODE, self).__init__()
        act = getattr(nn, activation)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 500),
            act(),
            nn.Linear(500, 500),
            act(),
            nn.Linear(500, 500),
            act(),
            nn.Linear(500, 500),
            act(),
            nn.Linear(500, 500),
            act(),
            nn.Linear(500, hidden_size),
            act(),
        )
        self.hidden2mu = nn.Linear(hidden_size, hidden_size)
        self.hidden2log_var = nn.Linear(hidden_size, hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 500), act(),
            nn.Linear(500, num_classes)
        )
        self.hidden_size = hidden_size

    def encode(self, x):
        hidden = self.encoder(x)
        self.hidden = hidden
        mu = self.hidden2mu(hidden)
        log_var = self.hidden2log_var(hidden)
        return mu, log_var
    
    def decode(self, x):
        x = self.decoder(x)
        return x
    
    def reparametrize(self, mu, log_var, z):
        sigma = torch.exp(0.5*log_var)
        self.sigma = sigma
        return mu + sigma*z
    
    def forward(self, x, z):
        mu, log_var = self.encode(x)
        self.log_var = log_var
        self.mu = log_var
        hidden = self.reparametrize(mu, log_var, z)
        output = self.decoder(hidden)
        return output

if args.network == 'ffn':
    net = FFN_PRECODE( n_classes, args.precode_size, args.activation ).to(**setup)
elif args.network == 'lenet':
    net = LeNet_PRECODE( n_classes, args.precode_size, args.activation ).to(**setup)
else:
    assert False, 'Bad Network.'

net.apply(weights_init)

optimizer_train = optim.Adam(net.parameters(),lr=0.005, weight_decay=1e-5)
scheduler_train = optim.lr_scheduler.ExponentialLR(optimizer_train, gamma=0.95)

j = 0 
setup2 = setup.copy()
setup2.pop('dtype')
for epoch in range(args.epochs):
    avg_error = 0
    avg_kl = 0
    
    for i, data in enumerate(trainloader,0):
        if j % 100 == 0:
            print(j)
            torch.save({
                'epoch': epoch,
                'it': j,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer_train.state_dict(),
                }, '{}/CIFAR10_{}_{}_{}.ckpt'.format(args.directory, args.network, args.activation, j))
        if (j+1) % args.iters == 0:
            exit()
        inputs,label = data[0].to(**setup), data[1].to(**setup2)
        inputs,label = Variable(inputs),Variable(label)
        optimizer_train.zero_grad()
        seed = torch.randn( [inputs.size()[0], net.hidden_size] ).to(**setup)
        outputs_benign = net(inputs,seed)
        mu, log_var = net.mu, net.log_var
        kl_loss = (-0.5*(1+log_var - mu**2 - torch.exp(log_var)).sum(dim=1)).mean(dim=0)
        loss_benign = cross_entropy_for_onehot(outputs_benign, label_to_onehot(label, num_classes=n_classes))
        avg_error += loss_benign.item()
        avg_kl += kl_loss.item()
        final_loss = loss_benign + 0.003*kl_loss
        final_loss.backward()
        optimizer_train.step()
        j += 1
    scheduler_train.step()
    acc = 0.0
    total = 0
    for i,data in enumerate(testloader,0):
        inputs,label = data[0].to(**setup), data[1].to(**setup2)
        seed = torch.randn( [inputs.size()[0], net.hidden_size] ).to(**setup2)
        y_pred = net(inputs,seed)
        predicted = torch.argmax(y_pred,axis=1)
        acc += torch.sum( predicted == label ).item()
        total += inputs.size()[0]
    test_accuracy = acc / total
    acc = 0.0
    total = 0
    for i,data in enumerate(trainloader,0):
        inputs,label = data[0].to(**setup), data[1].to(**setup2)
        seed = torch.randn( [inputs.size()[0], net.hidden_size] ).to(**setup)
        y_pred = net(inputs,seed)
        predicted = torch.argmax(y_pred,axis=1)
        acc += torch.sum( predicted == label ).item()
        total += inputs.size()[0]
    train_accuracy = acc / total
    print('Epoch', epoch, 'Train Accuracy', train_accuracy, 'Test Acc:', test_accuracy, 'Error:', avg_error, 'KL:', avg_kl )

