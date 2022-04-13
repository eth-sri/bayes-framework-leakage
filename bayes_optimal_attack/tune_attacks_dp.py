import csv
import os
from args_factory import get_args
from datetime import datetime
import itertools
from typing import Generator, Mapping, Tuple
import errno
import cloudpickle

import json
import jax
import jax.numpy as jnp
import jax.scipy.optimize as jso
import numpy as np
import optax
import torch
import time

from jax.tree_util import tree_flatten, tree_unflatten

from tqdm import tqdm, trange
from datasets.common import get_dataset_by_name
from datasets.distributions import get_prior
from utils.util import generate_init_img, load_model_flax_fed, store_model_flax_fed, get_image_from_loader
from utils.flax_losses import flax_cross_entropy_loss, flax_compute_metrics, flax_get_train_methods, flax_get_attack_loss_and_update
from utils.plotting import visualize
from models.base import get_network
from models.base_flax import get_flax_network, MLP_Flax
from utils.measures import get_acc_metric, l2_distance, compute_noise_l2

from args_factory import get_args
from defenses.defense import get_defense
from defenses.noise_defenses import get_evaluate_nets, get_defend_grad, MIN_SIGMA, MAX_SIGMA
from attacks.grad_attack import attack_via_grad_opt_flax, compute_matching, get_att_losses


from flax import linen as nn
from flax.training import train_state


torch.manual_seed(10)
np.random.seed(10)

import unicodedata
import re

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')

def aggregate_batch_metrics(batch_metrics):
    batch_metrics_np = jax.device_get(batch_metrics)
    agg_metrics_np = {k: np.mean([metrics[k] for metrics in batch_metrics_np]) for k in batch_metrics_np[0]}
    return agg_metrics_np

def attack_flax(args, net_state=None, def_state=None, client_id=None):
    rng = jax.random.PRNGKey(0)
    rng, dataset_rng = jax.random.split(rng)

    _, (train_loaders, test_loaders), n_targets, (dummy_input, dummy_targets) = get_dataset_by_name(
        args.dataset, args.batch_size, args.batch_size, k_batches=args.k_batches, rng=dataset_rng, n_clients=args.n_clients)
    train_loader, test_loader = train_loaders[client_id], test_loaders[client_id]
    prior = get_prior(args, args.dataset, dataset_rng)

    net = get_flax_network(args.network)
    create_train_state, _, eval_step = flax_get_train_methods(net, dummy_input)
    rng, init_rng = jax.random.split(rng)
    if net_state is None:
        net_state = create_train_state(init_rng, learning_rate=args.learning_rate)

    if args.defense is not None:
        rng, defense_rng = jax.random.split(rng)
        dummy_grad = jax.grad(lambda p: flax_cross_entropy_loss(log_probs=net.apply({'params': p}, dummy_input), labels=dummy_targets))(net_state.params)
        _, def_perturb_grads, def_log_prob, init_defense_params = get_defense(args.defense, defense_rng, args.batch_size, dummy_input, dummy_grad)
    else:
        def_perturb_grads, def_log_prob, defense_params = None, None, None

    if args.defense is not None:
        if def_state is None:
            def_state = train_state.TrainState.create(apply_fn=net.apply, params=init_defense_params, tx=optax.adam(learning_rate=args.defense_lr))
            if args.path is not None:
                net_state, def_state = load_model_flax_fed(args.path, net, dummy_input, net_state, def_state, client_id, args=args)
        defense_params = def_state.params
    else:
        if args.path is not None:
            net_state, _ = load_model_flax_fed(args.path, net, dummy_input, net_state, None, client_id=client_id, args=args)
    start_time = time.time()
    batch_metrics = []
    for inputs, targets in train_loader:
        if len(inputs.shape) == 4:
            inputs = np.einsum('bijk -> bjki', inputs)
        batch = {'image': inputs, 'label': targets}
        metrics = eval_step(net_state.params, batch)
        metrics = jax.device_get(metrics)
        batch_metrics.append(metrics)
    train_metrics_np = aggregate_batch_metrics(batch_metrics)

    batch_metrics = []
    for inputs, targets in test_loader:
        if len(inputs.shape) == 4:
            inputs = np.einsum('bijk -> bjki', inputs)
        batch = {'image': inputs, 'label': targets}
        metrics = eval_step(net_state.params, batch)
        metrics = jax.device_get(metrics)
        batch_metrics.append(metrics)
    test_metrics_np = aggregate_batch_metrics(batch_metrics)

    defend_grad, nodefend_grad = get_defend_grad(net, def_perturb_grads, args.batch_size)
    
    trainset = []
    for i, (inputs, targets) in enumerate(train_loader):
        trainset.append( (i, (inputs, targets)) )
        if (args.n_attack is not None) and i+1 >= args.n_attack:
            break
    end_time = time.time()
    init_time = end_time - start_time

    return net, def_log_prob, prior, init_time, rng, trainset, train_metrics_np, test_metrics_np, defend_grad, defense_params, net_state, n_targets, args

def execute_attack(net, def_log_prob, prior, init_time, rng, trainset, train_metrics_np, test_metrics_np, defend_grad, defense_params, net_state, n_targets, compiled_att_funcs, args):
    batch_metrics = []
    start_time = time.time()
    for i, (inputs, targets) in trainset:
        rng, att_rng = jax.random.split(rng)
        root_dir = f'out_img/{i}'
        args.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)
        if len(inputs.shape) == 4:
            inputs = np.einsum("bijk -> bjki", inputs)
        _, metrics = attack_via_grad_opt_flax(root_dir, compiled_att_funcs, att_rng, net, defend_grad, def_log_prob, defense_params, net_state, inputs, targets, n_targets, prior, args)
        batch_metrics.append(metrics)
        if (args.n_attack is not None) and i+1 >= args.n_attack:
            break
    end_time = time.time()
    metrics_np = aggregate_batch_metrics(batch_metrics)
    res_metrics_np = {
        'train_accuracy': train_metrics_np['accuracy'],
        'train_loss': train_metrics_np['loss'],
        'test_accuracy': test_metrics_np['accuracy'],
        'test_loss': test_metrics_np['loss'],
        'below_delta': metrics_np['below_delta'],
        'diff': metrics_np['diff'],
        'diff_mse': metrics_np['diff_mse'],
        'diff_psnr': metrics_np['diff_psnr'],
        'runtime': end_time - start_time + init_time,
    }
    return res_metrics_np

if __name__ == "__main__":
    test_params = { 'att_total_var' : [1, 0.5, 1.5, 2],
      'att_fac_start' : [1, 0.5, 1.5, 2],
      'att_lr': [1.0, 0.8, 1.2],
      'exp_decay_factor': [0.97, 0.95, 0.96, 0.98, 0.94],
      'att_exp_layers': [False,True],
      'reg_tv': [ 0.25, 0.1, 0.3, 0.5],
      'reg_clip': [ 0.75, 0.9, 0.7, 0.5],
      }


    args = get_args()
    
    if args.reg_tv==1.0 and args.reg_clip==0.0:
        del test_params['reg_tv']
        del test_params['reg_clip']

    attack_args = attack_flax(args, client_id=0)
    args = attack_args[-1]
    test_params['att_lr'] = [ args.att_lr * lr for lr in test_params['att_lr'] ]
    print( test_params['att_lr'] )
    net, def_log_prob, prior = attack_args[:3]

    outs = []
    best_psnr = -1
    combos = list( itertools.product( *list( test_params.values() ) ) )
    print ( 'Number of combinations:', len(combos), flush=True )
    factor = 10000000
    best_factors = []
    while factor > 5e-6:
        combo = list( zip( list( test_params.keys() ), combos[0] ) )
        for attr, val in combo:
            setattr(args, attr, val)

        args.att_total_var = factor
        args.att_fac_start = factor
        args.n_attack = 1

        compiled_att_funcs = get_att_losses(net, def_log_prob, prior, args)
        
        outs.append( ( combo, execute_attack(*attack_args[:-1], compiled_att_funcs, args) ) )
        
        psnr = outs[-1][-1]['diff_psnr']
        if np.abs( psnr - best_psnr ) < 0.001:
            best_factors.append( factor )
        elif psnr > best_psnr:
            best_psnr = psnr
            best_factors = [ factor ]
        
        factor /= 2
         
    best_factor_bin = np.mean( best_factors )

    best_psnr = -1
    for i in np.arange( 0.0, 1.0, 0.05 ):
        factor = best_factor_bin * 2 * i  + best_factor_bin * 0.5 * (1-i)
        args.att_total_var = factor
        args.att_fac_start = factor
        args.n_attack = 1

        compiled_att_funcs = get_att_losses(net, def_log_prob, prior, args)

        outs.append( ( combo, execute_attack(*attack_args[:-1], compiled_att_funcs, args) ) )

        psnr = outs[-1][-1]['diff_psnr']
        if psnr > best_psnr:
            best_psnr = psnr
            best_factor = factor

    print ( 'Best Factor_Bin:', best_factor_bin, 'Best_factor_Lin:', best_factor, 'Best_psnr:', best_psnr, flush=True )
    
    best_psnr = -1
    test_params['att_total_var'] = [ best_factor * var for var in test_params['att_total_var'] ]
    test_params['att_fac_start'] = [ best_factor * var for var in test_params['att_fac_start'] ]
    combos = list( itertools.product( *list( test_params.values() ) ) )

    idxs = [i for i,n in enumerate( test_params.keys()) if n.startswith('reg')]
    if not len(idxs) == 0:
        combos = [c for c in combos if np.isclose( np.sum( np.array(c)[idxs] ), 1.0, atol=1e-1 )  ]

    print ( 'Number of combinations:', len(combos), flush=True )
    for i, combo in enumerate( combos ):
        combo = list( zip( list( test_params.keys() ), combo ) )
        for attr, val in combo:
            setattr(args, attr, val)

        args.n_attack = 1

        compiled_att_funcs = get_att_losses(net, def_log_prob, prior, args)
        
        outs.append( ( combo, execute_attack(*attack_args[:-1], compiled_att_funcs, args) ) )
        runtime = outs[-1][-1]['runtime']
        
        psnr = outs[-1][-1]['diff_psnr']
        if psnr > best_psnr:
            best_psnr = psnr
            best_params = combo

        combo.append( ('psnr', psnr))
        combo.append( ('runtime', runtime ) )
        print( 'Combination', i, combo, flush=True )
    print( 'Best params:', best_params, flush=True )
    with open(slugify('l2_step100_MNIST_' + args.att_metric + '_' + args.path) + '.pickle', 'wb') as handle:
        cloudpickle.dump(outs, handle)
        
    for attr, val in best_params:
        setattr(args, attr, val)
    args.n_attack = 10

    compiled_att_funcs = get_att_losses(net, def_log_prob, prior, args)    
    exec_best = execute_attack(*attack_args[:-1], compiled_att_funcs, args)
    print( exec_best, flush=True )
