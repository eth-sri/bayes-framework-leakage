import jax
import jax.numpy as jnp
import jax.scipy.optimize as jso
import numpy as np
import time
import torch
from tqdm import trange
from utils.flax_losses import flax_cross_entropy_loss, flax_compute_metrics, flax_get_attack_loss_and_update
from utils.util import generate_init_img
from utils.measures import l2_distance, compute_noise_l2
from utils.plotting import visualize
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial import distance_matrix
from flax import linen as nn
from skimage.metrics import peak_signal_noise_ratio
from torchvision import transforms
from datasets.common import dataset_cfg


def compute_matching(at_inputs, inputs, delta, dataset, batch_size, metric='l2'):
    if (metric == 'mse' or metric == 'psnr') and (not dataset.startswith('dist_gaussian')):
        at_inputs = np.array(at_inputs)
        inputs = np.array(inputs)
        inv_normalize = transforms.Normalize(mean=dataset_cfg[dataset]['inv_mean'], std=dataset_cfg[dataset]['inv_std'])
        at_inputs = np.einsum("bijk -> bkij", at_inputs)
        inputs = np.einsum("bijk -> bkij", inputs)
        for i in range(inputs.shape[0]):
            at_inputs[i] = inv_normalize(torch.Tensor(np.array(at_inputs[i]))).cpu().detach().numpy()
            inputs[i] = inv_normalize(torch.Tensor(np.array(inputs[i]))).cpu().detach().numpy()

    k_batches = inputs.shape[0] // batch_size

    all_diff, all_below_delta = [], []
    for idx in range(k_batches):
        at_batch_inputs = at_inputs[idx*batch_size:(idx+1)*batch_size]
        batch_inputs = inputs[idx*batch_size:(idx+1)*batch_size]
        cost = np.zeros((batch_size, batch_size))
        for i in range(batch_size):
            for j in range(batch_size):
                if metric == 'l2':
                    cost[i, j] = np.sqrt(((at_batch_inputs[i] - batch_inputs[j])**2).sum())
                elif metric == 'mse':
                    cost[i, j] = ((at_batch_inputs[i] - batch_inputs[j])**2).sum() / at_batch_inputs[i].size
                elif metric == 'psnr':
                    cost[i, j] = -peak_signal_noise_ratio(at_batch_inputs[i], batch_inputs[j], data_range=1) # ???
                else:
                    assert False
        row_ind, col_ind = linear_sum_assignment(cost)
        if metric == 'psnr':
            cost = -cost
        diff = cost[row_ind, col_ind].mean()
        below_delta = (cost[row_ind, col_ind] < delta).mean()
        all_diff += [diff]
        all_below_delta += [below_delta]
    return np.mean(all_diff), np.mean(all_below_delta)

def get_att_losses(net, def_log_prob, prior, args):
    at_opt, _, _, at_update = flax_get_attack_loss_and_update(
            net, def_log_prob, prior, args.optimizer, args.att_lr, args.batch_size, args=args)
    return at_opt, at_update
        
def attack_via_grad_opt_flax( root_dir,  compiled_att_funcs, rng, net, defend_grad, def_log_prob, defense_params, state, inputs, targets, n_targets, prior, args=None):
    at_opt, at_update = compiled_att_funcs
    net_params = state.params
    rng, defense_rng = jax.random.split(rng)
    noisy_grads = defend_grad(rng, net_params, defense_params, inputs, targets)

    dummy_grad = jax.grad(lambda p: flax_cross_entropy_loss(log_probs=net.apply({'params': p}, inputs), labels=targets))(net_params)
    noise = compute_noise_l2(None, inputs, defense_params, args, dummy_grad)

    
    if args.noise_sched:
        start = min(args.att_fac_start * (noise**2), 1.0)
        alpha = np.exp(1.0/(args.att_epochs) * np.log(1/start))
        minmax = 1.0
    else:
        start = args.att_fac_start
        alpha = np.exp(1.0/(args.att_epochs) * np.log(args.att_total_var/start))
        minmax = args.att_total_var

    curr_fac = start
    
    rng, init_rng = jax.random.split(rng)
    inv_mean = np.array( dataset_cfg[args.dataset]['inv_mean'] )
    inv_std = np.array( dataset_cfg[args.dataset]['inv_std'] )

    at_img = generate_init_img(init_rng, inputs, args.dataset, args.att_init, prior)
    at_opt_state = at_opt.init(at_img)
    for idx in range(args.att_restarts):
        rng, init_rng = jax.random.split(rng)
        at_img = generate_init_img(init_rng, inputs, args.dataset, args.att_init, prior)
        at_opt_state = at_opt.init(at_img)

        if args.visualize:
            visualize(inputs, f'{root_dir}/reference.png', source=args.dataset, batch_size=args.batch_size)
            visualize(at_img, f"{root_dir}/at_img_0.png", source=args.dataset, batch_size=args.batch_size)

        curr_fac = start

        for at_iter in range(2*args.att_epochs):
            start_time = time.time()
            curr_fac = np.minimum( curr_fac * alpha, minmax )  if alpha >= 1.0 else np.maximum( curr_fac * alpha, minmax )
            #curr_fac = 1.0 if at_iter >= args.att_epochs else curr_fac * alpha

            rng, at_iter_rng = jax.random.split(rng)
            at_img, at_opt_state, att_loss = at_update(at_iter_rng, net_params, defense_params, at_opt_state, at_img, targets, noisy_grads, curr_fac, inv_mean, inv_std)
            end_time = time.time()
            if (at_iter+1) % args.vis_step == 0 and args.visualize:
                visualize(at_img, f"{root_dir}/at_img_{at_iter+1}.png", source=args.dataset, batch_size=args.batch_size)
            if args.verbose and at_iter % 5 == 0:
                diff, below_delta = compute_matching(at_img, inputs, args.delta, args.dataset, args.batch_size)
                print(f'at_iter={at_iter}, curr_fac={curr_fac:.3f}, att_loss={att_loss:.3f}, diff={diff:.3f}, below_delta={below_delta:.3f}, runtime={end_time-start_time:.3f}')
    if args.verbose:
        print('')

    diff, below_delta = compute_matching(at_img, inputs, args.delta, args.dataset, args.batch_size)
    diff_mse, _ = compute_matching(at_img, inputs, args.delta, args.dataset, args.batch_size, metric='mse')
    diff_psnr, _ = compute_matching(at_img, inputs, args.delta, args.dataset, args.batch_size, metric='psnr')

    metrics = {'diff': diff, 'diff_mse': diff_mse, 'diff_psnr': diff_psnr, 'below_delta': below_delta}
    return at_img, metrics
