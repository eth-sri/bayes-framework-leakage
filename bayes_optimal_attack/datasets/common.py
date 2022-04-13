import torch
import numpy as np
import jax.numpy as jnp
from datasets.distributions import get_distribution
from torchvision import datasets, transforms


# Dataset Meta-variables
dataset_cfg = {
    "MNIST": {
        'std': (0.3081,),
        'mean': (0.1307,),
        'inv_std': (1/0.3081),
        'inv_mean': (-0.1307/0.3081,)
    },
    "FashionMNIST": {
        'std': (0.3530,),
        'mean': (0.2860,),
        'inv_std': (1/0.3530),
        'inv_mean': (-0.2860/0.3530,)
    },
    "SmallMNIST": {
        'std': (0.3081,),
        'mean': (0.1307,),
        'inv_std': (1/0.3081),
        'inv_mean': (-0.1307/0.3081,)
    },
    "BinaryMNIST": {
        'std': (0.3081,),
        'mean': (0.1307,),
        'inv_std': (1/0.3081),
        'inv_mean': (-0.1307/0.3081,)
    },
    "CIFAR10": {
        'std': (0.2023, 0.1994, 0.2010),
        'mean': (0.4914, 0.4822, 0.4465),
        'inv_std': (1/0.2023, 1/0.1994, 1/0.2010),
        'inv_mean': (-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010)
    },
    "CIFAR100": {
        'std': (0.2675, 0.2565, 0.2761),
        'mean': (0.5071, 0.4867, 0.4408),
        'inv_std': (1/0.2675, 1/0.2565, 1/0.2761),
        'inv_mean': (-0.5071/0.2675, -0.4867/0.2565, -0.4408/0.2761)
    },
    "ImageNet": {
        'std': (0.229, 0.224, 0.225),
        'mean': (0.485, 0.456, 0.406),
        'inv_std': (1/0.229, 1/0.224, 1/0.225),
        'inv_mean': (-0.485/0.229, -0.456/0.224, -0.406/225)
    }
}


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class ToJNP(object):
    def __call__(self, pic):
        return np.array(pic, dtype=jnp.float32)


def split_data(train_data, test_data, train_batch_size, test_batch_size, k_batches, n_clients):
    train_data_chunks = torch.utils.data.random_split(train_data, [len(train_data)//n_clients]*n_clients)
    test_data_chunks = torch.utils.data.random_split(test_data, [len(test_data)//n_clients]*n_clients)

    train_loaders, test_loaders, train_loaders_def, test_loaders_def = [], [], [], []
    for i in range(n_clients):
        train_loaders += [torch.utils.data.DataLoader(train_data_chunks[i], batch_size=train_batch_size, collate_fn=numpy_collate, shuffle=True, drop_last=True)]
        test_loaders += [torch.utils.data.DataLoader(test_data_chunks[i], batch_size=test_batch_size, collate_fn=numpy_collate, shuffle=False, drop_last=True)]
        train_loaders_def += [torch.utils.data.DataLoader(train_data_chunks[i], batch_size=train_batch_size*k_batches, collate_fn=numpy_collate, shuffle=True, drop_last=True)]
        test_loaders_def += [torch.utils.data.DataLoader(test_data_chunks[i], batch_size=test_batch_size*k_batches, collate_fn=numpy_collate, shuffle=False, drop_last=True)]
    return (train_loaders, test_loaders), (train_loaders_def, test_loaders_def)
    

def get_dataset_by_name(dataset, train_batch_size, test_batch_size, k_batches=1, rng=None, normalize=True, n_clients=None):
    if dataset.startswith('dist_'):
        return get_distribution(dataset, train_batch_size, test_batch_size, k_batches=k_batches, rng=rng, n_clients=n_clients)
    elif dataset == "MNIST":
        train_data = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) if normalize else ToJNP(),
            ToJNP()]))
        test_data = datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) if normalize else ToJNP(),
            ToJNP()]))
        
        n_targets = 10
        dummy_input = np.random.rand(*(train_batch_size, 28, 28, 1))
        dummy_labels = np.random.randint(0, 9, (train_batch_size))
        if n_clients is None:
            train_loader_net = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, collate_fn=numpy_collate, shuffle=True)
            test_loader_net = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, collate_fn=numpy_collate, shuffle=False)
            train_loader_def = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size*k_batches, collate_fn=numpy_collate, shuffle=True)
            test_loader_def = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size*k_batches, collate_fn=numpy_collate, shuffle=False)
            return (train_loader_net, test_loader_net), (train_loader_def, test_loader_def), n_targets, (dummy_input, dummy_labels)
        else:
            (train_loaders, test_loaders), (train_loaders_def, test_loaders_def) = split_data(train_data, test_data, train_batch_size, test_batch_size, k_batches, n_clients)
            return (train_loaders, test_loaders), (train_loaders_def, test_loaders_def), n_targets, (dummy_input, dummy_labels)
    elif dataset == "FashionMNIST":
        train_data = datasets.FashionMNIST('../data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3530,), (0.2860,)) if normalize else ToJNP(),
            ToJNP()]))
        test_data = datasets.FashionMNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3530,), (0.2860,)) if normalize else ToJNP(),
            ToJNP()]))
        
        n_targets = 10
        dummy_input = np.random.rand(*(train_batch_size, 28, 28, 1))
        dummy_labels = np.random.randint(0, 9, (train_batch_size))
        if n_clients is None:
            train_loader_net = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, collate_fn=numpy_collate, shuffle=True)
            test_loader_net = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, collate_fn=numpy_collate, shuffle=False)
            train_loader_def = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size*k_batches, collate_fn=numpy_collate, shuffle=True)
            test_loader_def = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size*k_batches, collate_fn=numpy_collate, shuffle=False)
            return (train_loader_net, test_loader_net), (train_loader_def, test_loader_def), n_targets, (dummy_input, dummy_labels)
        else:
            (train_loaders, test_loaders), (train_loaders_def, test_loaders_def) = split_data(train_data, test_data, train_batch_size, test_batch_size, k_batches, n_clients)
            return (train_loaders, test_loaders), (train_loaders_def, test_loaders_def), n_targets, (dummy_input, dummy_labels)
    elif dataset == "SmallMNIST":
        train_data = datasets.MNIST('../data', train=True, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) if normalize else ToJNP(),
            ToJNP()]))
        test_data = datasets.MNIST('../data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) if normalize else ToJNP(),
            ToJNP()]))

        tr_data, tr_targets, te_data, te_targets = [], [], [], []
        for c in range(10):
            c_tr_data, c_tr_targets = train_data.data[train_data.targets == c], train_data.targets[train_data.targets == c]
            c_te_data, c_te_targets = test_data.data[test_data.targets == c], test_data.targets[test_data.targets == c]
            tr_perm, te_perm = np.random.permutation(c_tr_data.shape[0])[:100], np.random.permutation(c_te_targets.shape[0])[:100]
            tr_data, tr_targets = tr_data + [c_tr_data[tr_perm]], tr_targets + [c_tr_targets[tr_perm]]
            te_data, te_targets = te_data + [c_te_data[te_perm]], te_targets + [c_te_targets[te_perm]]
        train_data.data, train_data.targets = torch.cat(tr_data, dim=0), torch.cat(tr_targets)
        test_data.data, test_data.targets = torch.cat(te_data, dim=0), torch.cat(te_targets)

        train_loader_net = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, collate_fn=numpy_collate, shuffle=True)
        test_loader_net = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, collate_fn=numpy_collate, shuffle=False)

        train_loader_def = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size*k_batches, collate_fn=numpy_collate, shuffle=True)
        test_loader_def = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size*k_batches, collate_fn=numpy_collate, shuffle=False)
        
        n_targets = 10
        dummy_input = np.random.rand(*(train_batch_size, 28, 28, 1))
        dummy_labels = np.random.randint(0, 9, (train_batch_size))
        return (train_loader_net, test_loader_net), (train_loader_def, test_loader_def), n_targets, (dummy_input, dummy_labels)
    elif dataset == "CIFAR10":
        train_data = datasets.CIFAR10('../data', train=True, download=True, transform=transforms.Compose([
                                                       # transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                       ToJNP()]))
        test_data = datasets.CIFAR10('../data', train=False, download=True, transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                        ToJNP()]))

        n_targets = 10
        dummy_input = np.einsum("bijk -> bjki", np.random.rand(*(train_batch_size, 3, 32, 32)))
        dummy_labels = np.random.randint(0, 9, (train_batch_size))
        
        if n_clients is None:
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, collate_fn=numpy_collate, shuffle=True, drop_last=True)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, collate_fn=numpy_collate, shuffle=False, drop_last=True)
            train_loader_def = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size*k_batches, collate_fn=numpy_collate, shuffle=True, drop_last=True)
            test_loader_def = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size*k_batches, collate_fn=numpy_collate, shuffle=False, drop_last=True)
            return (train_loader, test_loader), (train_loader_def, test_loader_def), n_targets, (dummy_input, dummy_labels)
        else:
            (train_loaders, test_loaders), (train_loaders_def, test_loaders_def) = split_data(train_data, test_data, train_batch_size, test_batch_size, k_batches, n_clients)
            return (train_loaders, test_loaders), (train_loaders_def, test_loaders_def), n_targets, (dummy_input, dummy_labels)

    elif dataset == "CIFAR100":
        raise NotImplementedError
    elif dataset == "ImageNet":
        raise NotImplementedError
    else:
        assert False, f"Unknown dataset name {dataset}"
