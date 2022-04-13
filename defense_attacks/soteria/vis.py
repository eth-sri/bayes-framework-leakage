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
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.savefig(path)
    plt.close()
    #plt.show()

def load_images(folder_path):
    images_orig = []
    images_rec = []
    for img_path in sorted(glob.glob(f'{folder_path}/*.png')):
        if "rec" in img_path:
            images_rec.append(mpimg.imread(img_path))
        else:
            images_orig.append(mpimg.imread(img_path))

    return images_orig, images_rec

if __name__ == "__main__":

    # NOTE: Adapt these settings, you might need to adapt the path's used below
    use_steps = True
    model = "ConvBig"
    defense = 'ours'
    attack = "attack"
    layer = -4
    pruning_rate = 80.0
    tv = 0.0004
    
    for ctr in [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]:
        if use_steps:
            path = f"./results_{layer}_{attack}/steps_{ctr}_{model}-{defense}-{pruning_rate}-tv-{tv}"
            title = f"step_{ctr}"
        else:
            path = f"./results_{layer}_{attack}/epochs_{ctr}_{model}-{defense}-{pruning_rate}-tv-{tv}"
            title = f"epoch_{ctr}"
        images_orig, images_rec = load_images(path)
        target_title_ori = "a_summary_ori.png"
        target_title_rec = "a_summary_rec.png"
        sx = 4
        sy = 5
        if len(images_orig) == 100:
            sx = sy = 10
        if len(images_orig) == 10:
            sx = 2
            sy = 5
        
        vis_grid(images_orig, sx, sy, title+" ORIGINAL", os.path.join(path, target_title_ori))
        vis_grid(images_rec, sx, sy, title+" RECONSTRUCTED", os.path.join(path, target_title_rec))

        # Compute statistics
        stats = np.load(os.path.join(path, "metric.npy"), allow_pickle=True)

        mse = []
        denorm_mse = []
        psnr = []
        delta = []

        for stat in stats:
            mse.append(stat['mse'])
            denorm_mse.append(stat['denorm_mse'])
            psnr.append(stat['psnr'])
            delta.append(stat['delta'])

        mse = np.array(mse)
        denorm_mse = np.array(denorm_mse)
        psnr = np.array(psnr)
        delta = np.array(delta)

        with open(os.path.join(path, "stats.txt"), "w") as f:
            f.write(f"MSE: Mean {np.mean(mse):.4f} Std. {np.std(mse):.4f} Median {np.median(mse):.4f} Min {np.min(mse):.4f} Max {np.max(mse):.4f}\n")
            f.write(f"Denorm MSE: Mean {np.mean(denorm_mse):.4f} Std. {np.std(denorm_mse):.4f} Median {np.median(denorm_mse):.4f} Min {np.min(denorm_mse):.4f} Max {np.max(denorm_mse):.4f}\n")
            f.write(f"PSNR: Mean {np.mean(psnr):.4f} Std. {np.std(psnr):.4f} Median {np.median(psnr):.4f} Min {np.min(psnr):.4f} Max {np.max(psnr):.4f}\n")
            f.write(f"DELTA: Mean {np.mean(delta):.4f} Std. {np.std(delta):.4f} Median {np.median(delta):.4f} Min {np.min(delta):.4f} Max {np.max(delta):.4f}\n")


        print("Done")