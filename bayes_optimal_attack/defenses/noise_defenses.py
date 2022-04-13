import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax.tree_util import tree_flatten, tree_unflatten
from tensorflow_probability.substrates import jax as tfp
from utils.flax_losses import flax_cross_entropy_loss

tfd = tfp.distributions

# MIN_LOG_VAR = 2 * jnp.log(0.02)
# MIN_LOG_VAR = 2 * jnp.log(0.01)
MIN_LOG_VAR = 2 * jnp.log(0.001)
MIN_SIGMA = 1e-2
MAX_SIGMA = 1.0



@jax.jit
def add_noise_unbatched(rng, true_grad, dist):
    return jax.tree_map(lambda x: x + dist.sample(x.shape, seed=rng), true_grad)


@jax.jit
def add_learned_noise_unbatched(rng, true_grad, sigma_grads):
    unit_gauss = tfd.Normal(0, 1)
    true_grad_flat, g_tree = tree_flatten(true_grad)
    rngs = jax.random.split(rng, len(true_grad_flat))
    tree_rngs = tree_unflatten(g_tree, rngs)
    noisy_g = jax.tree_multimap(lambda g, sigma, iter_rng: g + sigma * unit_gauss.sample(g.shape, seed=iter_rng), true_grad, sigma_grads, tree_rngs)
    return noisy_g


def get_defend_grad(net, def_perturb_grads, batch_size):

    @jax.jit
    def get_noisy_grad(perturb_rng, net_params, defense_params, input, target):
        inputs, targets = jnp.expand_dims(input, axis=0), jnp.array([target])
        g = jax.grad(lambda p: flax_cross_entropy_loss(log_probs=net.apply({'params': p}, inputs), labels=targets))(net_params)
        return def_perturb_grads(perturb_rng, g, defense_params, inputs, batched=False)

    @jax.jit
    def nodefend_grad(net_params, inputs, targets):
        @jax.jit
        def single_grad(input, target):
            inputs, targets = jnp.expand_dims(input, axis=0), jnp.array([target])
            return jax.grad(lambda p: flax_cross_entropy_loss(log_probs=net.apply({'params': p}, inputs), labels=targets))(net_params)
        v = jax.vmap(single_grad, (0, 0))(inputs, targets)
        v = jax.tree_map(lambda x: x.reshape((-1, batch_size) + x.shape[1:]), v)
        return jax.tree_map(lambda x: jnp.mean(x, axis=1), v)

    @jax.jit
    def defend_grad(rng, net_params, defense_params, inputs, targets):
        rngs = jax.random.split(rng, inputs.shape[0] + 1)
        rng, perturb_rngs = rngs[0], rngs[1:]
        v = jax.vmap(get_noisy_grad, (0, None, None, 0, 0))(perturb_rngs, net_params, defense_params, inputs, targets)
        v = jax.tree_map(lambda x: x.reshape((-1, batch_size) + x.shape[1:]), v)
        noisy_grads = jax.tree_map(lambda x: jnp.mean(x, axis=1), v)
        return noisy_grads

    return defend_grad, nodefend_grad


def learned():

    def perturb_grads(rng, true_grads, defense_params, inputs, batched=True):
        sigma_grads,  = defense_params
        if batched:
            return add_learned_noise(rng, true_grads, sigma_grads)
        else:
            return add_learned_noise_unbatched(rng, true_grads, sigma_grads)

    @jax.jit
    def log_prob(grad, true_grad, defense_params, inputs, batch_size):
        sigma_grads,  = defense_params
        sigma_grads = jax.tree_map(lambda x: x/jnp.sqrt(batch_size), sigma_grads)
        d = jax.tree_multimap(lambda x, y, sigma: -jnp.log(sigma) - 0.5 * (x - y)**2 / jnp.square(sigma),
                              grad, true_grad, sigma_grads)
        d = jax.tree_map(lambda x: x.sum(axis=np.arange(1, len(x.shape))), d)
        return sum(jax.tree_leaves(d))

    return perturb_grads, log_prob


class DefenseMLP(nn.Module):
    num_feats: int
    log_sigma: jnp.float32

    @nn.compact
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(features=100)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.num_feats)(x)
        log_sigma_grads = nn.log_softmax(x) + jnp.log(x.size) + self.log_sigma
        sigma_grads = nn.softmax(x) * jnp.exp(self.log_sigma)
        return log_sigma_grads, sigma_grads


def get_evaluate_nets(nets):

    # @jax.jit
    def evaluate_nets(defense_params, inputs, true_grads):
        nets_params,  = defense_params
        log_sigma_grads, sigma_grads = jax.tree_multimap(lambda net, net_params: 2*(net.apply(net_params, inputs)), nets, nets_params)
        log_sigma_grads = jax.tree_multimap(lambda lvg, g: lvg.reshape((inputs.shape[0],) + g.shape[1:]), log_sigma_grads, true_grads)
        sigma_grads = jax.tree_multimap(lambda lvg, g: lvg.reshape((inputs.shape[0],) + g.shape[1:]), sigma_grads, true_grads)
        log_sigma_grads = jax.tree_map(lambda x: jnp.clip(x, a_min=jnp.log(MIN_SIGMA)), log_var_grads)
        sigma_grads = jax.tree_map(lambda x: jnp.clip(x, a_min=MIN_SIGMA))
        return log_var_grads

    def evaluate_nets_unbatched(defense_params, inputs, true_grads):
        nets_params,  = defense_params
        log_var_grads = jax.tree_multimap(lambda lvg, g: lvg.reshape(g.shape), log_var_grads, true_grads)
        log_var_grads = jax.tree_map(lambda x: jnp.clip(x, a_min=MIN_LOG_VAR), log_var_grads)
        return log_var_grads

    return evaluate_nets, evaluate_nets_unbatched


def learned_net(nets):
    noise_perturb_grads, noise_log_prob = learned()
    evaluate_nets, evaluate_nets_unbatched = get_evaluate_nets(nets)

    def perturb_grads(rng, true_grads, defense_params, inputs, batched=True):
        if batched:
            log_var_grads = evaluate_nets(defense_params, inputs, true_grads)
        else:
            log_var_grads = evaluate_nets_unbatched(defense_params, inputs, true_grads)
        return noise_perturb_grads(rng, true_grads, (log_var_grads,), inputs, batched)

    @jax.jit
    def log_prob(noisy_grad, true_grad, defense_params, inputs, batch_size):
        log_var_grads = evaluate_nets(defense_params, inputs, true_grad)
        return noise_log_prob(noisy_grad, true_grad, (log_var_grads,), inputs, batch_size)

    return perturb_grads, log_prob


def dp_gaussian():

    def perturb_grads(rng, true_grads, defense_params, inputs, batched=True):
        """
        Given list of true gradients (for each element in the batch), add noise sampled from N(0, std_dev^2) to each component independently.

        TODO: Implement proper DP, for now this is just heuristic which adds Gaussian noise (missing clipping and other details)
        TODO: This should be vectorized using vmap
        """
        std_dev,  = defense_params
        return add_noise_unbatched(rng, true_grads, tfd.Normal(0, std_dev))

    @jax.jit
    def log_prob(grad, true_grad, defense_params, inputs, batch_size):
        """ Given estimated true gradient (averaged over the batch), compute probability that reported noisy gradient equald to grad. """
        std_dev,  = defense_params
        dist = tfd.Normal(0, std_dev/jnp.sqrt(batch_size))
        log_prob = sum(jax.tree_leaves(jax.tree_multimap(lambda x, y: jnp.sum(dist.log_prob(x - y), axis=np.arange(1, len(x.shape))), grad, true_grad)))
        return log_prob

    return perturb_grads, log_prob


def dp_laplacian():

    def perturb_grads(rng, true_grads, defense_params, inputs, batched=True):
        scale,  = defense_params
        return add_noise_unbatched(rng, true_grads, tfd.Laplace(0, scale))

    @jax.jit
    def log_prob(grad, true_grad, defense_params, inputs, batch_size):
        scale, = defense_params
        dist = tfd.Laplace(0, scale)
        log_prob = sum(jax.tree_leaves(jax.tree_multimap(lambda x, y: jnp.sum(dist.log_prob(x - y), axis=np.arange(1, len(x.shape))), grad, true_grad)))
        return log_prob

    # Wrong probability
    # @jax.jit
    # def log_prob(grad, true_grad, defense_params, inputs, batch_size):
    #     """ Given estimated true gradient (averaged over the batch), compute probability that reported noisy gradient equald to grad. """
    #     std_dev,  = defense_params
    #     dist = tfd.Normal(0, std_dev/jnp.sqrt(batch_size))
    #     log_prob = sum(jax.tree_leaves(jax.tree_multimap(lambda x, y: jnp.sum(dist.log_prob(x - y), axis=np.arange(1, len(x.shape))), grad, true_grad)))
    #     return log_prob

    return perturb_grads, log_prob


def soft_pruning(dummy_grad, p, rng):

    g_flat, g_tree = tree_flatten(dummy_grad)
    prune_grads = []
    for g_val in g_flat:
        rng, iter_rng = jax.random.split(rng)
        prune_grads += [jax.random.bernoulli(rng, 1 - p, g_val.shape).astype(jnp.float32)]
    prune_grads = tree_unflatten(g_tree, prune_grads)

    def perturb_grads(rng, true_grads, defense_params, inputs, batched=True):
        p, scale, = defense_params
        true_grads_pruned = jax.tree_multimap(lambda x, y: x * y, true_grads, prune_grads)
        return add_noise_unbatched(rng, true_grads_pruned, tfd.Laplace(0, scale))

    @jax.jit
    def log_prob(grad, true_grad, defense_params, inputs, batch_size):
        p, scale, = defense_params
        dist = tfd.Laplace(0, scale)
        true_grad_pruned = jax.tree_multimap(lambda x, y: x * y, true_grad, prune_grads)
        log_prob = sum(jax.tree_leaves(jax.tree_multimap(lambda x, y: jnp.sum(dist.log_prob(x - y), axis=np.arange(1, len(x.shape))), grad, true_grad_pruned)))
        return log_prob

    return perturb_grads, log_prob


def soft_gaussian_pruning(dummy_grad, p, rng):

    g_flat, g_tree = tree_flatten(dummy_grad)
    prune_grads = []
    for g_val in g_flat:
        rng, iter_rng = jax.random.split(rng)
        prune_grads += [jax.random.bernoulli(rng, 1 - p, g_val.shape).astype(jnp.float32)]
    prune_grads = tree_unflatten(g_tree, prune_grads)

    def perturb_grads(rng, true_grads, defense_params, inputs, batched=True):
        p, std_dev, = defense_params
        true_grads_pruned = jax.tree_multimap(lambda x, y: x * y, true_grads, prune_grads)
        return add_noise_unbatched(rng, true_grads_pruned, tfd.Normal(0, std_dev))

    @jax.jit
    def log_prob(grad, true_grad, defense_params, inputs, batch_size):
        p, std_dev, = defense_params
        dist = tfd.Normal(0, std_dev)
        true_grad_pruned = jax.tree_multimap(lambda x, y: x * y, true_grad, prune_grads)
        log_prob = sum(jax.tree_leaves(jax.tree_multimap(lambda x, y: jnp.sum(dist.log_prob(x - y), axis=np.arange(1, len(x.shape))), grad, true_grad_pruned)))
        return log_prob

    return perturb_grads, log_prob
