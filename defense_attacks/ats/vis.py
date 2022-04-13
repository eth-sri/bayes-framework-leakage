import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.image as mpimg
import numpy as np
import os

def vis_grid(images, size_x, size_y, title, path):

    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(size_x, size_y),  # creates grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )

    fig.suptitle(title, fontsize=8)
    for ax, im in zip(grid, images):
        ax.imshow(im)

    plt.savefig(path)
    plt.close()

def load_images(folder_path):
    images_orig = []
    images_rec = []
    for img_path in sorted(glob.glob(f'{folder_path}/*.jpg')):
        if "rec" in img_path:
            images_rec.append(mpimg.imread(img_path))
        else:
            images_orig.append(mpimg.imread(img_path))

    return images_orig, images_rec

if __name__ == "__main__":
    
    # Config
    # NOTE: Please set these fields corresponding to your setup
    # Further you might want to adapt the paths for the dirctories used below to accomodate another folder structure or Model
    use_steps = True # If you want to go over all step attacks or over the epoch attacks
    ctrs = [1000] # Steps or epochs (e.g. [1,2,5,10,20,50,100,200,500,1000])
    augs = ["", "21-13-3", "7-4-15", "21-13-3+7-4-15"] # Selected augmentations


    for ctr in ctrs:
        for aug in augs:
            if use_steps:
                path = f"./benchmark/images_ConvNet/data_cifar10_arch_ConvNet_steps_{ctr}_bs_32_optim_inversed_mode_aug_auglist_{aug}_rlabel_False"
                title = f"step_{ctr}"
            else:
                path = f"./benchmark/images_ConvNet/data_cifar10_arch_ConvNet_epoch_{ctr}_optim_inversed_mode_aug_auglist_{aug}_rlabel_False"
                title = f"epoch_{ctr}_auglist_{aug}_rlabel_False"
            images_orig, images_rec = load_images(path)
            target_title_ori = "a_summary_ori.png"
            target_title_rec = "a_summary_rec.png"
            sx = 4
            sy = 5
            if len(images_orig) == 100:
                sx = sy = 10
            
            vis_grid(images_orig, sx, sy, title+" ORIGINAL", os.path.join(path, target_title_ori))
            vis_grid(images_rec, sx, sy, title+" RECONSTRUCTED", os.path.join(path, target_title_rec))

            # Compute statistics
            stats = np.load(os.path.join(path, "metric.npy"), allow_pickle=True)

            test_mse = []
            feat_mse = []
            psnr = []
            delta = []

            for stat in stats:
                test_mse.append(stat['test_mse'])
                feat_mse.append(stat['feat_mse'])
                psnr.append(stat['test_psnr'])
                delta.append(stat['delta'])

            test_mse = np.array(test_mse)
            feat_mse = np.array(feat_mse)
            psnr = np.array(psnr)
            delta = np.array(delta)

            with open(os.path.join(path, "stats.txt"), "w") as f:
                f.write(f"Denorm MSE: Mean {np.mean(test_mse):.4f} Std. {np.std(test_mse):.4f} Median {np.median(test_mse):.4f} Min {np.min(test_mse):.4f} Max {np.max(test_mse):.4f}\n")
                f.write(f"PSNR: Mean {np.mean(psnr):.4f} Std. {np.std(psnr):.4f} Median {np.median(psnr):.4f} Min {np.min(psnr):.4f} Max {np.max(psnr):.4f}\n")
                f.write(f"DELTA: Mean {np.mean(delta):.4f} Std. {np.std(delta):.4f} Median {np.median(delta):.4f} Min {np.min(delta):.4f} Max {np.max(delta):.4f}\n")


            print("Done")