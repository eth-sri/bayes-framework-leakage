import os, sys
sys.path.insert(0, './')
import inversefed
import torch
import torchvision
seed=23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import random
random.seed(seed)

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import inversefed
import torchvision.transforms as transforms
import argparse
from autoaugment import SubPolicy
from inversefed.data.data_processing import _build_cifar100, _get_meanstd
from inversefed.data.loss import LabelSmoothing
from inversefed.utils import Cutout
import torch.nn.functional as F
import policy
from benchmark.comm import create_model, build_transform, preprocess, create_config


parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--aug_list', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--optim', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--mode', default=None, required=True, type=str, help='Mode.')
parser.add_argument('--rlabel', default=False, type=bool, help='rlabel')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=-1, type=int, help='Vision epoch.')
parser.add_argument('--steps', default=0, type=int, help='Individual steps in the first epoch to consider.')
parser.add_argument('--batch_size', default=128, type=int, help='Individual steps in the first epoch to consider.')
parser.add_argument('--resume', default=0, type=int, help='rlabel')

opt = parser.parse_args()
num_images = 1


# init env
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative')

if opt.epochs != -1:
    defs.epochs = opt.epochs
    defs.steps = None
elif opt.steps > 0:
    defs.epochs = 1
    defs.steps = opt.steps
else:
    assert False, "Invalid configuration"
defs.batch_size = opt.batch_size


# init training
arch = opt.arch
trained_model = True
mode = opt.mode
assert mode in ['normal', 'aug', 'crop']

config = create_config(opt)


def create_save_dir():
    if opt.steps:
        return 'benchmark/images_ConvNet/data_{}_arch_{}_steps_{}_bs_{}_optim_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.steps, opt.batch_size, opt.optim, opt.mode, \
        opt.aug_list, opt.rlabel)
    else:
        return 'benchmark/images_ConvNet/data_{}_arch_{}_epoch_{}_optim_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.epochs, opt.optim, opt.mode, \
        opt.aug_list, opt.rlabel)


def reconstruct(idx, model, loss_fn, trainloader, validloader):

    if opt.data == 'cifar100' or opt.data == "cifar10":
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    elif opt.data == 'FashionMinist':
        dm = torch.Tensor([0.1307]).view(1, 1, 1).cuda()
        ds = torch.Tensor([0.3081]).view(1, 1, 1).cuda()
    else:
        raise NotImplementedError

    # prepare data
    ground_truth, labels = [], []
    while len(labels) < num_images:
        img, label = trainloader.dataset[idx]   # TODO original testloader
        idx += 1
        if label not in labels:
            labels.append(torch.as_tensor((label,), device=setup['device']))
            ground_truth.append(img.to(**setup))

    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels)
    model.zero_grad()
    target_loss, _, _ = loss_fn(model(ground_truth), labels)
    param_list = [param for param in model.parameters() if param.requires_grad]
    input_gradient = torch.autograd.grad(target_loss, param_list)


    # attack
    print('ground truth label is ', labels)
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=num_images)
    if opt.data == 'cifar100' or opt.data == 'cifar10':
        shape = (3, 32, 32)
    elif opt.data == 'FashionMinist':
        shape = (1, 32, 32)

    # Print statistics on the gradient
    #print(input_gradient[0].shape)
    #print(input_gradient[1].shape)
    #print(f"Gradient stats: Mean: {torch.mean(torch.abs(input_gradient[0]))} Median: {torch.median(torch.abs(input_gradient[0]))} Max: {torch.max(torch.abs(input_gradient[0]))} ")

    if opt.rlabel:
        output, stats = rec_machine.reconstruct(input_gradient, None, img_shape=shape) # reconstruction label
    else:
        output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=shape) # specify label

    output_denormalized = output * ds + dm
    input_denormalized = ground_truth * ds + dm
    mean_loss = torch.mean((input_denormalized - output_denormalized) * (input_denormalized - output_denormalized))
    
    

    save_dir = create_save_dir()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torchvision.utils.save_image(output_denormalized.cpu().clone(), '{}/rec_{}.jpg'.format(save_dir, idx))
    torchvision.utils.save_image(input_denormalized.cpu().clone(), '{}/ori_{}.jpg'.format(save_dir, idx))


    test_mse = (output_denormalized.detach() - input_denormalized).pow(2).mean().cpu().detach().item()
    delta = (output_denormalized.detach() - input_denormalized).pow(2).sum().cpu().detach().item()
    feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean().cpu().detach().item()
    test_psnr = inversefed.metrics.psnr(output_denormalized, input_denormalized)

    print(f"after optimization, MSE Denorm {test_mse:.4f} | FEAT Norm: {feat_mse:.4f} | Delta: {delta:.4f} | PSNR: {test_psnr:.4f}")

    return {'test_mse': test_mse,
        'feat_mse': feat_mse,
        'test_psnr': test_psnr,
        'delta': delta
    }




def create_checkpoint_dir():
    return 'checkpoints/data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel)


def main():
    global trained_model
    print(opt)
    print("Starting attack")
    loss_fn, trainloader, validloader = preprocess(opt, defs, valid=False)
    model = create_model(opt)
    model.to(**setup)
    if opt.epochs == 0:
        trained_model = False
        
    if trained_model:
        checkpoint_dir = create_checkpoint_dir()
        if 'normal' in checkpoint_dir:
            checkpoint_dir = checkpoint_dir.replace('normal', 'crop')
        if opt.steps:
            filename = os.path.join(checkpoint_dir, f'{opt.arch}_bs_{opt.batch_size}_steps_{opt.steps}.pth')
        else:
            filename = os.path.join(checkpoint_dir, str(defs.epochs) + '.pth')

        if not os.path.exists(filename) and opt.epochs > 0:
            filename = os.path.join(checkpoint_dir, str(defs.epochs - 1) + '.pth')

        print(filename)
        assert os.path.exists(filename)
        model.load_state_dict(torch.load(filename))

    if opt.rlabel:
        for name, param in model.named_parameters():
            if 'fc' in name:
                param.requires_grad = False

    model.eval()
    sample_list = [i for i in range(100)]
    metric_list = list()
    mse_loss = 0
    for attack_id, idx in enumerate(sample_list):
        if idx < opt.resume:
            continue
        print('attach {}th in {}'.format(idx, opt.aug_list))
        metric = reconstruct(idx, model, loss_fn, trainloader, validloader)
        metric_list.append(metric)
    save_dir = create_save_dir()
    np.save('{}/metric.npy'.format(save_dir), metric_list)



if __name__ == '__main__':
    main()
