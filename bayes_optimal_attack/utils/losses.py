from typing import Mapping, Tuple, Iterable
import jax
import tree
from utils.measures import total_variation
import jax.numpy as jnp
import numpy as np
import optax


@jax.jit
def l2_loss(params: Iterable[jnp.ndarray]) -> jnp.ndarray:
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in params)


@jax.jit
def l2_inner_gradient_matching_loss(grads_1, grads_2):
    # Here we can compute the actual loss based on the target gradients and the attack gradients
    res = jax.tree_multimap(lambda x, y: x-y, grads_1, grads_2)
    loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(res))
    return loss


@jax.jit
def l2_outer_gradient_matching_loss(net, params, state, inputs, targets, target_grads, factor=1.0):
    grads, state = jax.grad(loss, has_aux=True)(params, state, inputs, targets)
    loss = factor * l2_inner_gradient_matching_loss(grads, target_grads)
    return loss


# Training loss functions
def get_loss_and_update(net: Transformed, loss_str: str, opt_str: str, n_targets: int) -> any:

    # Get the correct loss function
    if loss_str == "CE":
        def loss(params: hk.Params, state: hk.State, input: np.ndarray, targets: np.ndarray) -> jnp.ndarray:
            """Compute the CE-loss of the network"""
            logits, state = net.apply(params, state, None, input)
            labels = jax.nn.one_hot(targets, n_targets)

            softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
            softmax_xent /= labels.shape[0]

            return softmax_xent, state
    elif loss_str == "L2":
        def loss(params: hk.Params, input: np.ndarray, targets: np.ndarray) -> jnp.ndarray:
            """Compute the L2-loss of the network"""
            # TODO
            logits = net.apply(params, input)
            labels = jax.nn.one_hot(targets, n_targets)

            l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
            softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
            softmax_xent /= labels.shape[0]

            return softmax_xent + 1e-4 * l2_loss
    elif loss_str == "CE-L2":
        def loss(params: hk.Params, state: hk.State, input: np.ndarray, targets: np.ndarray) -> jnp.ndarray:
            """Compute the CE-loss of the network, Regularize with L2 on the parameters"""
            logits, state = net.apply(params, state, None, input)
            labels = jax.nn.one_hot(targets, n_targets)

            l2_params = [p for ((mod_name, _), p) in tree.flatten_with_path(params) if 'batchnorm' not in mod_name]

            l2_l = l2_loss(l2_params)
            softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
            softmax_xent /= labels.shape[0]

            return softmax_xent + 1e-4 * l2_l, state

    ##### Choose the optimizer and define the update functions #####

    if opt_str == "adam":
        opt = optax.adam(1e-2)  # TODO Learning rate passing
    elif opt_str == "sgd":
        opt = optax.sgd()
    elif opt_str == "adagrad":
        opt = optax.adagrad()
    else:
        assert False, f"Unknown optimizer: {opt_str}"

    @jax.jit
    def update(
        params: hk.Params,
        state: hk.State,
        opt_state: optax.OptState,
        input: np.ndarray, targets: np.ndarray
    ) -> Tuple[hk.Params, optax.OptState]:
        """Learning rule (stochastic gradient descent)."""
        grads, state = jax.grad(loss, has_aux=True)(params, state, input, targets)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, state

    # We maintain avg_params, the exponential moving average of the "live" params.
    # avg_params is used only for evaluation (cf. https://doi.org/10.1137/0330046)
    @jax.jit
    def ema_update(params, avg_params):
        return optax.incremental_update(params, avg_params, step_size=0.001)
    
    return loss, opt, update, ema_update


def cos_sim(x, y):
    return 1 - jnp.sum(x * y) / (jnp.linalg.norm(x, 2) * jnp.linalg.norm(y, 2))

# Attack loss functions
def get_attack_loss_and_update(net: Transformed, loss_str: str, opt_str: str, n_targets: int,
                               pre_factor: float = 1.0, gradient_arg: int = 3, args: any = None) -> any:
    # Get the correct loss function
    if loss_str == "CE":
        def loss(params: hk.Params, state: hk.State, input: np.ndarray, targets: np.ndarray) -> jnp.ndarray:
            """Compute the CE-loss of the network"""
            logits, state = net.apply(params, state, None, input)
            labels = jax.nn.one_hot(targets, n_targets)

            softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
            softmax_xent /= labels.shape[0]

            return softmax_xent, state
    elif loss_str == "CE-L2":
        def loss(params: hk.Params, input: np.ndarray, targets: np.ndarray) -> jnp.ndarray:
            """Compute the CE-loss of the network, Regularize with L2 on the parameters"""
            logits = net.apply(params, input)
            labels = jax.nn.one_hot(targets, n_targets)

            l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
            softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
            softmax_xent /= labels.shape[0]

            return softmax_xent + 1e-4 * l2_loss

    # Choose the optimizer and define the update functions
    if opt_str == "adam":
        opt = optax.adam(learning_rate=args.learning_rate)
    elif opt_str == "sgd":
        opt = optax.sgd(learning_rate=args.learning_rate, momentum=0.9, nesterov=True)
    elif opt_str == "adagrad":
        opt = optax.adagrad(learning_rate=args.learning_rate)
    else:
        assert False, f"Unknown optimizer: {opt_str}"

    def at_internal(params, grads, state, inputs, targets):
        att_grads, state = jax.grad(loss, has_aux=True)(params, state, inputs, targets)

        # Here we can compute the actual loss based on the target gradients and the attack gradients
        if args.attack_loss == 'l2':
            res = jax.tree_multimap(lambda x, y: x-y, grads, att_grads)
            att_loss = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(res))
        elif args.attack_loss == 'cos':
            res = jax.tree_multimap(lambda x, y: cos_sim(x, y), grads, att_grads)
            att_loss = sum(jnp.square(p) for p in jax.tree_leaves(res))
        else:
            assert False, 'Unknown attack loss!'

        # TV-loss
        tv_loss = 0
        acc_loss = 0
        if args.attack_total_variation > 0:
            tv_loss = args.attack_total_variation * total_variation(inputs)
        if args.attack_accuracy_reg > 0:
            logits = net.apply(params, inputs)
            labels = jax.nn.one_hot(targets, n_targets)
            l2_dist = jax.tree_multimap(lambda x, y: x-y, logits, labels)
            l2_dist = 0.5 * jnp.sum(jnp.square(l2_dist))
            acc_loss = args.attack_accuracy_reg * l2_dist
        return pre_factor * (att_loss + tv_loss + acc_loss), state

    @jax.jit
    def att_update(
        params: np.ndarray,
        opt_state: optax.OptState,
        state: hk.State,
        inputs: np.ndarray,
        targets: np.ndarray,
        grads: any
    ) -> Tuple[hk.Params, optax.OptState]:
        # Derivative w.r.t gradient_arg
        att_grad, state = jax.grad(at_internal, gradient_arg, has_aux=True)(params, grads, state, inputs, targets)

        att_loss, _ = at_internal(params, grads, state, inputs, targets)

        updates, opt_state = opt.update(att_grad, opt_state)
        if gradient_arg == 3:
            new_att_value = optax.apply_updates(inputs, updates)    # Here it can either be the image (for loss minimization)
        elif gradient_arg == 0:
            new_att_value = optax.apply_updates(params, updates)    # Or the network weights for maximization
        return new_att_value, opt_state, att_loss, state

    return loss, opt, at_internal, att_update


