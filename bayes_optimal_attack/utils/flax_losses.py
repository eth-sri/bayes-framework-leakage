import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from utils.measures import total_variation
from utils.util import ball_get_random
from datasets.distributions import VAE_Prior, GMM_Prior


@jax.jit
def flax_cross_entropy_loss(*, log_probs, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=log_probs.shape[1])
    return -jnp.mean(jnp.sum(one_hot_labels * log_probs, axis=-1))


@jax.jit
def flax_compute_metrics(*, log_probs, labels):
    loss = flax_cross_entropy_loss(log_probs=log_probs, labels=labels)
    accuracy = jnp.mean(jnp.argmax(log_probs, -1) == labels)
    metrics = {'loss': loss, 'accuracy': accuracy}
    return metrics


@jax.jit
def cos_sim(x, y):
    x_l2_sqr = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(x))
    y_l2_sqr = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(y))
    return 1 - jnp.sum(x * y) / (jnp.sqrt(x_l2_sqr + 1e-7) * jnp.sqrt(y_l2_sqr + 1e-7))


@jax.jit
def l2_dist(x, y):
    x_l2_sqr = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(x))
    y_l2_sqr = sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(y))
    return jnp.sum(jnp.square(x/jnp.sqrt(x_l2_sqr+1e-7) - y/jnp.sqrt(y_l2_sqr+1e-7)))

@jax.jit
def clip_prior(x, mean, std ):
    x_unorm = (x - mean)/ std  
    dist_clip = jnp.sum( jnp.mean( jnp.square(  x_unorm - jnp.clip( x_unorm, 0.0, 1.0) ), axis=0) )
    return dist_clip

def flax_get_train_methods(net, dummy_input):
    def create_train_state(rng, learning_rate, momentum=None, opt='adam'):
        params = net.init(rng, dummy_input)['params']
        if opt == 'adam':
            tx = optax.adam(learning_rate)
        elif opt == 'sgd':
            tx = optax.sgd(learning_rate, momentum)
        return train_state.TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state, batch, rng=None):
        def loss_fn(params):
            log_probs = net.apply({'params': params}, batch['image'])
            loss = flax_cross_entropy_loss(log_probs=log_probs, labels=batch['label'])
            return loss, log_probs
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, log_probs), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = flax_compute_metrics(log_probs=log_probs, labels=batch['label'])
        metrics['grad_l2'] = sum(jax.tree_leaves(jax.tree_map(lambda x: (x**2).sum(), grads)))
        return state, metrics

    @jax.jit
    def eval_step(params, batch):
        log_probs = net.apply({'params': params}, batch['image'])
        return flax_compute_metrics(log_probs=log_probs, labels=batch['label'])

    return create_train_state, train_step, eval_step


# Attack loss functions
def flax_get_attack_loss_and_update(net, def_log_prob, prior, opt_str, learning_rate, batch_size, args, is_train=False):
    @jax.jit
    def get_orig_grads(net_params, inputs, targets):
        @jax.jit
        def single_grad(input, target):
            inputs, targets = jnp.expand_dims(input, axis=0), jnp.array([target])
            return jax.grad(lambda p: flax_cross_entropy_loss(log_probs=net.apply({'params': p}, inputs), labels=targets))(net_params)
        v = jax.vmap(single_grad, (0, 0))(inputs, targets)
        v = jax.tree_map(lambda x: x.reshape((args.k_batches, batch_size) + x.shape[1:]), v)
        return jax.tree_map(lambda x: jnp.mean(x, axis=1), v)

    exp_schedule = optax.exponential_decay(learning_rate, args.exp_decay_steps, args.exp_decay_factor)  # TODO: Tune this
    opt = optax.adam(learning_rate=exp_schedule)

    @jax.jit
    def at_internal(rng, params, defense_params, grads, inputs, targets, fac, inv_mean=0.0, inv_std=1.0):
        att_grads = get_orig_grads(params, inputs, targets)

        def_logp, prior_logp = 0, 0
        if def_log_prob is not None:
            def_logp = def_log_prob(grads, att_grads, defense_params, inputs, batch_size)
        if prior is not None:
            prior_logp = prior.log_prob(inputs).mean()

        if len(inputs.shape) == 4:
            tot_var = args.attack_total_variation * total_variation(inputs).mean()
        else:
            tot_var = 0.0


        if args.att_metric is not None:
            layer_weights = np.arange(len(grads), 0, -1)
            layer_weights = np.exp( layer_weights )
            layer_weights = layer_weights / np.sum( layer_weights )
            layer_weights = layer_weights / layer_weights[0]
            layer_weights = np.repeat(layer_weights, 2)
            if not args.att_exp_layers:
                layer_weights = np.repeat([1.0], 2*len(grads))
            trs = jax.tree_structure( grads )
            layer_weights = jax.tree_unflatten(trs, layer_weights)
            if args.att_inv_sigma:
                weights = jax.tree_multimap(lambda s: ( jnp.sqrt(batch_size) /s ).reshape(1, *s.shape ), defense_params[0])
            else:
                weights = np.repeat([1.0], 2*len(grads))
                weights = jax.tree_unflatten(trs, weights)
 
            if args.att_metric == 'l2':
                res = jax.tree_multimap(lambda x, y, w, lw:  lw*jnp.sum(jnp.multiply(jnp.square(x-y),w)), grads, att_grads, weights, layer_weights)
                l2_loss = sum(p for p in jax.tree_leaves(res))
                att_loss = l2_loss
            elif args.att_metric == 'l1':
                res = jax.tree_multimap(lambda x, y, w, lw:  lw*jnp.sum(jnp.multiply(jnp.abs(x-y),w)), grads, att_grads, weights, layer_weights)
                l1_loss = sum(p for p in jax.tree_leaves(res))
                att_loss = l1_loss
            elif args.att_metric == 'cos_sim':
                res = jax.tree_multimap(lambda x, y, w, lw: lw*cos_sim(jnp.multiply(x,w), jnp.multiply(y,w)), grads, att_grads, weights, layer_weights)
                att_loss = sum(p for p in jax.tree_leaves(res))
            elif args.att_metric == 'cos_sim_global':
                dot = jax.tree_multimap(lambda x, y, w, lw: -lw*jnp.sum( jnp.multiply( jnp.multiply(x,w), jnp.multiply(y,w) ) ), grads, att_grads, weights, layer_weights)
                dot = sum(p for p in jax.tree_leaves(dot))
                norm1 = jax.tree_multimap(lambda x,w: jnp.sum( jnp.multiply(jnp.multiply(x,w),jnp.multiply(x,w)) ), grads, weights)
                norm1 = sum(p for p in jax.tree_leaves(norm1))
                norm2 = jax.tree_multimap(lambda x,w: jnp.sum(  jnp.multiply(jnp.multiply(x,w),jnp.multiply(x,w)) ), att_grads, weights)
                norm2 = sum(p for p in jax.tree_leaves(norm2))
                att_loss = 1 + dot / (jnp.sqrt(norm1 + 1e-7) * jnp.sqrt(norm2 + 1e-7))
            else:
                assert False
            clip_err = clip_prior(inputs, inv_mean, inv_std )
            att_loss /= args.k_batches
            tot_loss = att_loss * fac + args.reg_tv * tot_var + args.reg_clip * clip_err
        else:
            if len(inputs.shape) == 4:
                tot_loss = -def_logp * fac + tot_var
            else:
                tot_loss = -def_logp * fac - prior_logp
            tot_loss = tot_loss.sum(0)
        return tot_loss

    @jax.jit
    def at_internal_train(rng, params, defense_params, grads, inputs, targets, fac):
        d = jnp.zeros(inputs.shape)
        return at_internal(rng, params, defense_params, grads, inputs + d, targets, fac)

    def at_internal_region(rng, params, defense_params, grads, inputs, targets, fac, n_samples):
        @jax.jit
        def compute_one_sample(sample_rng):
            d = jnp.zeros(inputs.shape)
            return at_internal(rng, params, defense_params, grads, inputs + d, targets, fac)
        rngs = jax.random.split(rng, n_samples + 1)
        rng, sample_rngs = rngs[0], rngs[1:]
        losses = jax.vmap(compute_one_sample)(sample_rngs)
        return losses.mean()

    @jax.jit
    def at_update(rng, params, defense_params, opt_state, inputs, targets, grads, fac, inv_mean, inv_std):
        att_grad_fn = jax.value_and_grad(at_internal, 4)
        att_loss, att_grad = att_grad_fn(rng, params, defense_params, grads, inputs, targets, fac, inv_mean, inv_std)
        updates, opt_state = opt.update(att_grad, opt_state)
        new_att_value = optax.apply_updates(inputs, updates)
        return new_att_value, opt_state, att_loss

    @jax.jit
    def at_update_train(rng, params, defense_params, opt_state, inputs, targets, grads, fac):
        att_grad_fn = jax.value_and_grad(at_internal, 4)
        att_loss, att_grad = att_grad_fn(rng, params, defense_params, grads, inputs, targets, fac)
        updates, opt_state = opt.update(att_grad, opt_state)
        new_att_value = optax.apply_updates(inputs, updates)
        return new_att_value, opt_state, att_loss

    jit_at_internal_region = jax.jit(at_internal_region, static_argnums=(7,))

    if is_train:
        return opt, at_internal_train, jit_at_internal_region, at_update_train

    return opt, at_internal, jit_at_internal_region, at_update
