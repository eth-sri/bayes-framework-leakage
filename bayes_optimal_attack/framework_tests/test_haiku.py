import haiku as hk
import jax
import jax.numpy as jnp

def softmax_cross_entropy(logits, labels):
  one_hot = jax.nn.one_hot(labels, logits.shape[-1])
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)

def loss_fn(images, labels):
  mlp = hk.Sequential([
      hk.Linear(300), jax.nn.relu,
      hk.Linear(100), jax.nn.relu,
      hk.Linear(10),
  ])
  logits = mlp(images)
  return jnp.mean(softmax_cross_entropy(logits, labels))

# There are two transforms in Haiku, hk.transform and hk.transform_with_state.
# If our network updated state during the forward pass (e.g. like the moving
# averages in hk.BatchNorm) we would need hk.transform_with_state, but for our
# simple MLP we can just use hk.transform.
loss_fn_t = hk.transform(loss_fn)

# MLP is deterministic once we have our parameters, as such we will not need to
# pass an RNG key to apply. without_apply_rng is a convenience wrapper that will
# make the rng argument to `loss_fn_t.apply` default to `None`.
loss_fn_t = hk.without_apply_rng(loss_fn_t)

print("Done2")