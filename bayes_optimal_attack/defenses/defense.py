from jax.tree_util import tree_flatten, tree_unflatten
from defenses.noise_defenses import dp_gaussian, dp_laplacian, soft_pruning, soft_gaussian_pruning, learned, learned_net, DefenseMLP
from tensorflow_probability.substrates import jax as tfp
import jax
import jax.numpy as jnp

tfd = tfp.distributions


def get_defense(defense_name, rng, batch_size, dummy_input, dummy_grad):
    tokens = defense_name.split('_')
    if tokens[0] == 'dp':
        if tokens[1] == 'gaussian':
            std_dev = float(tokens[2])
            def_perturb_grads, def_log_prob = dp_gaussian()
            return None, def_perturb_grads, def_log_prob, (std_dev,)
        elif tokens[1] == 'laplacian':
            assert batch_size == 1
            scale = float(tokens[2])
            def_perturb_grads, def_log_prob = dp_laplacian()
            return None, def_perturb_grads, def_log_prob, (scale,)
        else:
            assert False
    elif tokens[0] == 'soft' and tokens[1] == 'gaussian' and tokens[2] == 'pruning':
        p = float(tokens[3])
        std_dev = float(tokens[4])
        rng, prune_rng = jax.random.split(rng)
        def_perturb_grads, def_log_prob = soft_gaussian_pruning(dummy_grad, p, prune_rng)
        return None, def_perturb_grads, def_log_prob, (p, std_dev,)
    elif tokens[0] == 'soft' and tokens[1] == 'pruning':
        p = float(tokens[2])
        scale = float(tokens[3])
        rng, prune_rng = jax.random.split(rng)
        def_perturb_grads, def_log_prob = soft_pruning(dummy_grad, p, prune_rng)
        return None, def_perturb_grads, def_log_prob, (p, scale,)
    elif tokens[0] == 'learned' and tokens[1] == 'noise':
        scale = float(tokens[2])
        g_flat, g_tree = tree_flatten(dummy_grad)

        # log_var_grads = []
        # for g_val in g_flat:
        #     log_var_grads += [2 * jnp.log(scale * jnp.ones(g_val.shape))]
        # log_var_grads = tree_unflatten(g_tree, log_var_grads)

        sigma_grads = []
        for g_val in g_flat:
            sigma_grads += [scale * jnp.ones(g_val.shape)]
        sigma_grads = tree_unflatten(g_tree, sigma_grads)

        def_perturb_grads, def_log_prob = learned()
        return None, def_perturb_grads, def_log_prob, (sigma_grads,)
        # return None, def_perturb_grads, def_log_prob, (log_var_grads,)
    elif tokens[0] == 'learned' and tokens[1] == 'net':
        scale = float(tokens[2])
        g_flat, g_tree = tree_flatten(dummy_grad)
        nets_flat, params_flat = [], []
        for g_val in g_flat:
            rng, iter_rng = jax.random.split(rng)
            net = DefenseMLP(num_feats=g_val.size, log_sigma=jnp.log(scale))
            nets_flat += [net]
            params_flat += [net.init(iter_rng, dummy_input)]
        nets = tree_unflatten(g_tree, nets_flat)
        nets_params = tree_unflatten(g_tree, params_flat)
        def_perturb_grads, def_log_prob = learned_net(nets)
        def_params = (nets_params, )
        return nets, def_perturb_grads, def_log_prob, def_params
    else:
        assert False
