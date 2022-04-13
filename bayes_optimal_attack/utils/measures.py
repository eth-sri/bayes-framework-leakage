import numpy as np
import jax
import jax.numpy as jnp


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k """
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def get_acc_metric(net: any, n_targets: int):
    # Evaluation metric (classification accuracy).
    @jax.jit
    def accuracy(params: hk.Params, state: hk.State, input: np.ndarray, targets: np.ndarray) -> jnp.ndarray:
        predictions, state = net.apply(params, state, None, input)
        # targets = one_hot(targets, n_targets)
        return jnp.mean(jnp.argmax(predictions, axis=-1) == targets)

    @jax.jit
    def cum_sum(params: hk.Params, state: hk.State, input: np.ndarray, targets: np.ndarray) -> jnp.ndarray:
        predictions, state = net.apply(params, state, None, input)
        return jnp.sum(jnp.argmax(predictions, axis=-1) == targets)
    return accuracy, cum_sum


@jax.jit
def total_variation(x: np.ndarray) -> float:
    x_diff = x - jnp.roll(x, -1, axis=1)
    y_diff = x - jnp.roll(x, -1, axis=2)
    grad_norm2 = x_diff**2 + y_diff**2 + 1e-7
    # grad_norm1 = jnp.abs(x_diff) + jnp.abs(y_diff) + 1e-7
    # return jnp.sum(grad_norm1, axis=(1, 2, 3))
    return jnp.sum(grad_norm2, axis=(1, 2, 3))


@jax.jit
def l2_distance(x, y):
    return jnp.sqrt(jnp.square(x - y).reshape(x.shape[0], -1).sum(axis=1))
    # return jnp.square(x - y).reshape(x.shape[0], -1).sum(axis=1)


def compute_noise_l2(evaluate_nets, inputs, defense_params, args, dummy_grad):
    if args.defense.startswith('learned_net'):
        log_var_grads = evaluate_nets(defense_params, inputs, dummy_grad)
        num_params = sum(jax.tree_leaves(jax.tree_map(lambda x: jnp.prod(x.size), log_var_grads)))
        noise_l2 = sum(jax.tree_leaves(jax.tree_map(lambda x: jnp.sum(jnp.exp(0.5*x)), log_var_grads))) / num_params
    elif args.defense.startswith('learned_noise'):
        num_params = sum(jax.tree_leaves(jax.tree_map(lambda x: jnp.prod(x.size), defense_params)))
        noise_l2 = sum(jax.tree_leaves(jax.tree_map(lambda x: jnp.sum(x), defense_params))) / num_params
    else:
        return defense_params[0]
    return noise_l2

