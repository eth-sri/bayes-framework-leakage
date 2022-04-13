import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state


def reparameterize(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, logvar.shape)
    return mean + eps * std


class Encoder(nn.Module):
    latents: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(500)(x)
        x = nn.relu(x)
        mean_x = nn.Dense(self.latents)(x)
        logvar_x = nn.Dense(self.latents)(x)
        return mean_x, logvar_x


class Decoder(nn.Module):

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(500)(z)
        z = nn.relu(z)
        z = nn.Dense(784)(z)
        return z


@jax.vmap
def kl_divergence(mean, logvar):
    return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))


@jax.vmap
def binary_cross_entropy_with_logits(logits, labels):
    logits = nn.log_sigmoid(logits)
    return -jnp.sum(labels * logits + (1. - labels) * jnp.log(-jnp.expm1(logits)))


class VAE(nn.Module):
    latents: int = 20

    def setup(self):
        self.encoder = Encoder(self.latents)
        self.decoder = Decoder()

    def __call__(self, x, z_rng):
        mean, logvar = self.encoder(x)
        z = reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def generate(self, z):
        return nn.sigmoid(self.decoder(z))


def model():
    return VAE(latents=20)


@jax.jit
def vae_log_prob(params, x, z_rng):
    recon_x, mean, logvar = model().apply({'params': params}, x, z_rng)
    bce_loss = binary_cross_entropy_with_logits(recon_x, x)
    kld_loss = kl_divergence(mean, logvar)
    loss = bce_loss + kld_loss
    return -loss


def compute_metrics(recon_x, x, mean, logvar):
    bce_loss = binary_cross_entropy_with_logits(recon_x, x).mean()
    kld_loss = kl_divergence(mean, logvar).mean()
    return {'bce': bce_loss, 'kld': kld_loss, 'loss': bce_loss + kld_loss}


@jax.jit
def compute_l2_loss(params):
    l2_loss = sum(jax.tree_leaves(jax.tree_map(lambda x: jnp.sum(x**2), params)))
    return l2_loss


@jax.jit
def train_step(state, batch, z_rng):
    def loss_fn(params):
        recon_x, mean, logvar = model().apply({'params': params}, batch, z_rng)
        bce_loss = binary_cross_entropy_with_logits(recon_x, batch).mean()
        kld_loss = kl_divergence(mean, logvar).mean()
        l2_loss = compute_l2_loss(params)
        loss = bce_loss + kld_loss + 1e-1 * l2_loss
        return loss
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


def eval(vae, params, images, z, z_rng):
    def eval_model(vae):
        recon_images, mean, logvar = vae(images, z_rng)
        comparison = jnp.concatenate([images[:8].reshape(-1, 28, 28, 1), recon_images[:8].reshape(-1, 28, 28, 1)])
        generate_images = vae.generate(z)
        generate_images = generate_images.reshape(-1, 28, 28, 1)
        metrics = compute_metrics(recon_images, images, mean, logvar)
        return metrics, comparison, generate_images
    return nn.apply(eval_model, vae)({'params': params})


