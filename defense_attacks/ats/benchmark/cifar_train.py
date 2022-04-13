import os, sys
sys.path.insert(0, './')
import torch
import torchvision
seed=23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import random
random.seed(seed)

import inversefed
import argparse
import policy

from benchmark.comm import create_model, preprocess



policies = policy.policies

parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=-1, type=int, help='Vision epoch.')
parser.add_argument('--steps', default=0, type=int, help='Individual steps in the first epoch to consider.')
parser.add_argument('--batch_size', default=32, type=int, help='Individual steps in the first epoch to consider.')
parser.add_argument('--aug_list', default=None, required=True, type=str, help='Augmentation method.')
parser.add_argument('--tiny', action="store_true", help='Use a tiny dataset.')
parser.add_argument('--mode', default=None, required=True, type=str, help='Mode.')
parser.add_argument('--rlabel', default=False, type=bool, help='remove label.')
parser.add_argument('--evaluate', default=False, type=bool, help='Evaluate')

opt = parser.parse_args()

# init env
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative'); 

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
tiny = opt.tiny
mode = opt.mode
assert mode in ['normal', 'aug', 'crop']


def create_save_dir():
    if tiny:
        return 'checkpoints/tiny_data_{}_arch_{}'.format(opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel)
    else:
        return 'checkpoints/data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel)


def main():
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

    loss_fn, trainloader, validloader = preprocess(opt, defs, tiny=tiny)

    # init model
    model = create_model(opt)
    model.to(**setup)
    save_dir = create_save_dir()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    if opt.steps > 0:
        file = f'{save_dir}/{arch}_bs_{defs.batch_size}_steps_{defs.steps}.pth'
    else:
        file = f'{save_dir}/{arch}_{defs.epochs}.pth'

    inversefed.train(model, loss_fn, trainloader, validloader, defs, setup=setup, save_dir=save_dir)
    torch.save(model.state_dict(), f'{file}')
    model.eval()


def evaluate():
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative'); defs.epochs=opt.epochs
    loss_fn, trainloader, validloader = preprocess(opt, defs, valid=False)
    model = create_model(opt)
    model.to(**setup)
    root = create_save_dir()

    filename = os.path.join(root, '{}_{}.pth'.format(opt.arch, opt.epochs))
    print(filename)
    if not os.path.exists(filename):
        assert False

    print(filename)
    model.load_state_dict(torch.load(filename))
    model.eval()
    stats = {'valid_losses':list(), 'valid_Accuracy':list()}
    inversefed.training.training_routine.validate(model, loss_fn, validloader, defs, setup=setup, stats=stats)
    print(stats)

if __name__ == '__main__':
    if opt.evaluate:
        evaluate()
        exit(0)
    main()
