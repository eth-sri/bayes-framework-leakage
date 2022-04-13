import argparse
import cloudpickle
import pickle
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets.common import get_dataset_by_name
from flax import serialization
from flax.training import train_state
from tqdm import tqdm
from vae import VAE, train_step, eval
from sklearn.mixture import GaussianMixture

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


def model(args):
    return VAE(latents=args.latents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_model', type=str, required=True, choices=['vae', 'gmm'], help='which generative model to use')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset to use')
    parser.add_argument('--batch_size', type=int, default=32, required=False, help='Batch size to use')
    parser.add_argument('--epochs', type=int, default=30, required=False, help='Batch size to use')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate to use')
    parser.add_argument('--latents', type=int, default=20)
    parser.add_argument('--gmm_components', type=int, default=1)
    args = parser.parse_args()

    train_loader, test_loader, n_targets, dummy_input = get_dataset_by_name(args.dataset, args.batch_size, args.batch_size, normalize=True)

    rng = jax.random.PRNGKey(0)
    rng, rng_init = jax.random.split(rng)

    if args.gen_model == 'vae':
        vae = model(args)
        init_data = jnp.ones((args.batch_size, 784), jnp.float32)
        state = train_state.TrainState.create(
            apply_fn=vae.apply,
            params=vae.init(rng_init, init_data, rng)['params'],
            tx=optax.adam(args.learning_rate),
        )

        rng, z_key, eval_rng = jax.random.split(rng, 3)
        z = jax.random.normal(z_key, (64, args.latents))

        for test_inputs, test_targets in test_loader:
            test_inputs = np.einsum("bijk -> bjki", test_inputs)
            test_inputs = jnp.reshape(test_inputs, (test_inputs.shape[0], -1))
            break

        for epoch in range(args.epochs):
            for inputs, targets in tqdm(train_loader):
                rng, batch_rng = jax.random.split(rng)
                inputs = np.einsum("bijk -> bjki", inputs)
                inputs = jnp.reshape(inputs, (inputs.shape[0], -1))
                state = train_step(state, inputs, batch_rng)
            metrics, comparison, sample = eval(vae, state.params, test_inputs, z, eval_rng)
            print(epoch, metrics)

        path = 'generative/saved_models/vae.pickle'
        with open(path, "wb") as out_file:
            vae_dict_out = serialization.to_state_dict(state)
            cloudpickle.dump(vae_dict_out, out_file)
    else:
        gmm = {}
        all_inputs = []
        for inputs, targets in train_loader:
            inputs = np.einsum("bijk -> bjki", inputs)
            inputs = jnp.reshape(inputs, (inputs.shape[0], -1))
            all_inputs += [inputs]
        x = jnp.concatenate(all_inputs, axis=0)
        
        gmm = GaussianMixture(n_components=args.gmm_components, covariance_type='full', verbose=10)
        gmm.fit(x)

        print(gmm.score_samples(x))
        path = f'generative/saved_models/gmm_{args.dataset}.pickle'
        with open(path, 'wb') as out_file:
            pickle.dump(gmm, out_file)





if __name__ == '__main__':
    main()
