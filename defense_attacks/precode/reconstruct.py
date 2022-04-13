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
torch.manual_seed(50)

import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data

from sklearn.model_selection import train_test_split

import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m","--model", type=str, required=True, help="Model path")
    parser.add_argument("-a", "--activation", type=str, choices=['ReLU', 'Tanh', 'Sigmoid'], default='ReLU', help="What activation to use")
    parser.add_argument("-ps","--precode_size", type=int, default=256, required=False, help="Number of hidden units in PRECODE layer")

    return parser.parse_args()

args = get_args()


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
setup = inversefed.utils.system_startup(args={'device':'cpu'})
defs = inversefed.training_strategy('conservative')
        
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

checkpoint = torch.load(args.model)
net = FFN_PRECODE(n_classes, args.precode_size, args.activation).to(**setup)
net.load_state_dict(checkpoint['model_state_dict'])

defs.augmentations = []
loss_fn, trainloader, validloader =  inversefed.construct_dataloaders('CIFAR10', defs,shuffle=False)
net.eval();
seed = torch.randn(net.hidden_size).to(**setup).requires_grad_(False)

dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]

tens=[]
for image_idx in range(20):
    seed = torch.randn(net.hidden_size).to(**setup).requires_grad_(False)
    img, label = trainloader.dataset[image_idx]
    img = img.view([1]+list(img.size())).to(**setup)
    label = torch.as_tensor((label,), device=setup['device'])

    dummy_seed_init = torch.randn(net.hidden_size )
    dummy_seed = torch.Tensor(dummy_seed_init).to(**setup).requires_grad_(True)
    im = img.clone().detach().mul_(ds).add_(dm).clamp_(0, 1)[0]
    
    net.zero_grad()
    target_loss, _, _ = loss_fn(net(img, seed), label)
    input_gradient = torch.autograd.grad(target_loss, net.parameters())
    input_gradient = [grad.detach() for grad in input_gradient]
    print( 'Label:', torch.argmax( net(img, seed) ).item(), 'GT Label:', label.item() ) 

    inv_layer_idx = 1

    net(img, seed)
    idx = input_gradient[inv_layer_idx].abs().argmax().detach().cpu()
    derivative_inv = input_gradient[inv_layer_idx-1][idx] /  input_gradient[inv_layer_idx][idx]

    print( 'Reconstruction error:', np.mean(np.abs((img.reshape(-1) - derivative_inv).detach().cpu().numpy()) ), 'Max Reconstruction error:', np.max(np.abs((img.reshape(-1) - derivative_inv).detach().cpu().numpy()) )  )
    derivative_inv = derivative_inv.reshape( img.shape ) 
    im = derivative_inv.clone().detach().mul_(ds).add_(dm).clamp_(0, 1)[0]
    im = img.clone().detach().mul_(ds).add_(dm).clamp_(0, 1)[0]
    
    im_deriv = derivative_inv.clone().detach().mul_(ds).add_(dm).clamp_(0, 1)[0].numpy()
    tens.append( im_deriv.reshape(1,*im_deriv.shape)  )

from torchvision.utils import make_grid
inp = torch.from_numpy(np.concatenate(tens))
grid = make_grid(inp, nrow=10, padding=2, pad_value=1).cpu().numpy()
plt.axis('off')
plt.imshow(np.transpose(grid, (1, 2, 0)), interpolation='nearest')
plt.savefig('precode.pdf', bbox_inches='tight', pad_inches=0) 
