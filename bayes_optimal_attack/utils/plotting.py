import torch 
from torchvision import transforms
from torchvision.utils import make_grid
from datasets.common import dataset_cfg
import numpy as np
import matplotlib.pyplot as plt


def visualize(input, path_to_save, idx=0, source="CIFAR10", show=False, batch_size=1):
    plt.close('all')
    input = np.array(input)
    input = np.einsum("bijk -> bkij", input)
    inv_normalize = transforms.Normalize(mean=dataset_cfg[source]['inv_mean'], std=dataset_cfg[source]['inv_std'])
    for i in range(input.shape[0]):
        input[i] = inv_normalize(torch.Tensor(np.array(input[i]))).cpu().detach().numpy()
    input = np.clip(input, 0, 1)
    input = torch.from_numpy(input)
    grid = make_grid(input, nrow=batch_size, padding=0).cpu().numpy()
    plt.imshow(np.transpose(grid, (1, 2, 0)), interpolation='nearest')
    
    if show:
        plt.show()
        return None
    else:
        plt.savefig(path_to_save)
        return None
