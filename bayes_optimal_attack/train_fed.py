from typing import Generator, Mapping, Tuple
import os
import errno
import argparse
import csv
import cloudpickle
from datetime import datetime

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
from utils.util import generate_init_img, load_model_flax_fed, store_model_flax_fed, get_image_from_loader, ball_get_random
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
    batch_metrics = []
    compiled_att_funcs = get_att_losses(net, def_log_prob, prior, args)
    for i, (inputs, targets) in enumerate(train_loader):
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
        'runtime': end_time - start_time,
    }
    return res_metrics_np



def train_flax(args):
    epochs = args.epochs
    batch_size = args.batch_size
    dataset = args.dataset
    network = args.network

    net = get_flax_network(network)
    rng = jax.random.PRNGKey(0)

    # Make the dataset.
    rng, dataset_rng = jax.random.split(rng)
    (train_loader, test_loader), _, n_targets, (dummy_input, dummy_targets) = get_dataset_by_name(
        dataset, batch_size, batch_size, k_batches=args.k_batches, rng=dataset_rng)

    create_train_state, train_step, eval_step = flax_get_train_methods(net, dummy_input)

    def train_epoch(state, train_loader, batch_size, epoch, rng):
        batch_metrics = []
        for inputs, targets in tqdm(train_loader):
            rng, iter_rng = jax.random.split(rng)
            if len(inputs.shape) == 4:
                inputs = np.einsum('bijk -> bjki', inputs)
            batch = {'image': inputs, 'label': targets}
            state, metrics = train_step(state, batch, iter_rng)
            batch_metrics.append(metrics)
        epoch_metrics_np = aggregate_batch_metrics(batch_metrics)
        return state, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'], epoch_metrics_np['grad_l2']

    def eval_model(params, test_loader):
        batch_metrics = []
        for inputs, targets in test_loader:
            if len(inputs.shape) == 4:
                inputs = np.einsum('bijk -> bjki', inputs)
            batch = {'image': inputs, 'label': targets}
            metrics = eval_step(params, batch)
            metrics = jax.device_get(metrics)
            batch_metrics.append(metrics)
        test_metrics_np = aggregate_batch_metrics(batch_metrics)
        return test_metrics_np['loss'], test_metrics_np['accuracy']

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, args.learning_rate)

    for epoch in range(epochs):
        state, train_loss, train_acc, grad_l2 = train_epoch(state, train_loader, batch_size, epoch, rng)
        print('[train] epoch: %d, loss: %.4f, accuracy: %.2f, grad_l2: %.2f' % (epoch, train_loss, train_acc, grad_l2))
    test_loss, test_acc = eval_model(state.params, test_loader)
    print('[test] loss: %.4f, accuracy: %.2f' % (test_loss, test_acc))

    name = f"{args.dataset}_{args.network}_epochs_{args.epochs}_time_{datetime.now().strftime('%H-%M-%S')}_flax.pickle"
    path = os.path.join(*[args.prefix, name])
    store_model_flax(path, state)


if __name__ == "__main__":
    args = get_args()
    assert not (args.attack and args.defend), "Cannot attack and defend at the same time"

    if args.attack:
        print(attack_flax(args, client_id=0))
    elif args.defend:
        args.path = defend_flax(args)
        # args.att_epochs = 300
        # args.att_init = 'random'
        # attack_flax(args)
    else:
        train_flax(args)


