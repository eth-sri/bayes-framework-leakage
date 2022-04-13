import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pickle
import torch
from flax import serialization
from flax.training import train_state
from torch.utils.data import DataLoader, Dataset
from tensorflow_probability.substrates import jax as tfp
from generative.vae import VAE, vae_log_prob
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt

tfd = tfp.distributions


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(DataLoader):

    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn)


class FlattenAndCast(object):

    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


class MyDataset(Dataset):

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)


class VAE_Prior:

    def __init__(self, vae, vae_state):
        self.vae = VAE
        self.vae_state = vae_state

    def log_prob(self, x, z_rng):
        return vae_log_prob(self.vae_state.params, x, z_rng)


class GMM_Prior:

    def __init__(self, gmm_sk):
        means, covs = [], []
        for c in range(gmm_sk.n_components):
            means.append(gmm_sk.means_[c])
            covs.append(gmm_sk.covariances_[c])
        means = np.stack(means, axis=0).astype(np.float32)
        covs = np.stack(covs, axis=0).astype(np.float32)
        self.gmm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=gmm_sk.weights_.astype(np.float32)),
            components_distribution=tfd.MultivariateNormalFullCovariance(
                loc=means,
                covariance_matrix=covs,
            ))

    def log_prob(self, x, c):
        return self.gmm.log_prob(x)

    def sample(self, shape, rng):
        return self.gmm.sample(shape, seed=rng)


class ToJNP(object):
    def __call__(self, pic):
        return np.array(pic, dtype=jnp.float32)


def get_prior(args, dataset, rng):
    if dataset.startswith('dist_gmm'):
        gmm = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.25, 0.25, 0.25, 0.25]),
            components_distribution=tfd.MultivariateNormalDiag(
                loc=[[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [0.0, -2.0]],
                scale_identity_multiplier=[1.0, 1.0, 1.0, 1.0]
            ))
        return gmm
    elif dataset.startswith('dist_gaussian'):
        d = int(dataset.split('_')[2])
        mean = jnp.zeros(d)
        cov = jnp.eye(d)
        # cov_delta = jax.random.normal(rng, (d, d))
        # cov = 0.01*np.dot(cov_delta, cov_delta.T)
        return tfd.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov)
    elif dataset == 'BinaryMNIST':
        path = 'generative/saved_models/gmm_BinaryMNIST.pickle'
        with open(path, 'rb') as in_file:
            gmm_sk = pickle.load(in_file)
        gmm_prior = GMM_Prior(gmm_sk)
        return gmm_prior
    elif dataset == 'MNIST' or dataset == 'SmallMNIST':
        return None
        # path = 'generative/saved_models/gmm_SmallMNIST.pickle'  # TODO: Change this
        # with open(path, 'rb') as in_file:
        #     gmm_sk = pickle.load(in_file)
        # gmm_prior = GMM_Prior(gmm_sk)
        # return gmm_prior

        # vae = VAE(latents=20)
        # init_data = jnp.ones((args.batch_size, 784), jnp.float32)
        # rng, rng_init = jax.random.split(rng)
        # vae_state = train_state.TrainState.create(
        #     apply_fn=vae.apply,
        #     params=vae.init(rng_init, init_data, rng)['params'],
        #     tx=optax.adam(args.learning_rate),
        # )
        # path = 'generative/saved_models/vae.pickle'
        # with open(path, "rb") as in_file:
        #     try:
        #         vae_state = serialization.from_state_dict(vae_state, cloudpickle.load(in_file))
        #         print('Loaded VAE')
        #     except Exception as e:
        #         print(f"Couldn't load specified model: {e}")
        # return VAE_Prior(vae, vae_state)
    else:
        return None


def get_distribution(dataset, train_batch_size, test_batch_size, k_batches=1, rng=None, n_clients=None):
    if dataset.startswith('dist_gmm'):
        n_targets = 2
        n_train, n_test = 1000, 1000

        gaussians = [
            tfd.MultivariateNormalDiag(loc=[2, 0], scale_identity_multiplier=1.0),
            tfd.MultivariateNormalDiag(loc=[0, 2], scale_identity_multiplier=1.0),
            tfd.MultivariateNormalDiag(loc=[-2, 0], scale_identity_multiplier=1.0),
            tfd.MultivariateNormalDiag(loc=[0, -2], scale_identity_multiplier=1.0),
        ]

        rng, train_rng = jax.random.split(rng)
        train_targets = jax.random.randint(train_rng, (n_train,), 0, 4)
        train_samples = []
        for i in range(n_train):
            rng, train_rng = jax.random.split(rng)
            train_samples += [gaussians[train_targets[i]].sample(seed=train_rng)]
        train_samples = jax.device_get(jnp.stack(train_samples, axis=0))

        rng, test_rng = jax.random.split(rng)
        test_targets = jax.random.randint(test_rng, (n_test,), 0, 4)
        test_samples = []
        for i in range(n_test):
            rng, test_rng = jax.random.split(rng)
            test_samples += [gaussians[test_targets[i]].sample(seed=test_rng)]
        test_samples = jax.device_get(jnp.stack(test_samples, axis=0))
    elif dataset.startswith('dist_gaussian'):
        d = int(dataset.split('_')[2])
        n_targets = int(dataset.split('_')[3])
        n_train, n_test = 1000, 1000
        # n_train, n_test = 100, 100
        gauss = get_prior(None, dataset, rng)

        rng, train_rng = jax.random.split(rng)
        train_samples = jax.device_get(gauss.sample(sample_shape=(n_train,), seed=train_rng))
        rng, test_rng = jax.random.split(rng)
        test_samples = jax.device_get(gauss.sample(sample_shape=(n_test,), seed=test_rng))

        rng, w_rng = jax.random.split(rng)
        w = jax.random.normal(rng, (d, n_targets))

        train_targets = jnp.argmax(jnp.dot(train_samples, w), axis=1).astype(np.long)
        test_targets = jnp.argmax(jnp.dot(test_samples, w), axis=1).astype(np.long)

        # for i in range(n_targets):
        #     plt.scatter(train_samples[train_targets==i, 0], train_samples[train_targets==i, 1], s=4)
        # plt.show()
        # exit(0)
    else:
        assert False

    if n_clients is None:
        train_loader_net = NumpyLoader(MyDataset(train_samples, train_targets), batch_size=train_batch_size, drop_last=True, shuffle=True)
        test_loader_net = NumpyLoader(MyDataset(test_samples, test_targets), batch_size=test_batch_size, drop_last=True, shuffle=False)
        train_loader_def = NumpyLoader(MyDataset(train_samples, train_targets), batch_size=train_batch_size*k_batches, drop_last=True, shuffle=True)
        test_loader_def = NumpyLoader(MyDataset(test_samples, test_targets), batch_size=test_batch_size*k_batches, drop_last=True, shuffle=False)
        return (train_loader_net, test_loader_net), (train_loader_def, test_loader_def), n_targets, (train_samples[:1], train_targets[:1])
    else:
        train_loaders, test_loaders, train_loaders_def, test_loaders_def = [], [], [], []
        k = n_train // n_clients
        for i in range(n_clients):
            train_loaders += [NumpyLoader(MyDataset(train_samples[i*k:(i+1)*k], train_targets[i*k:(i+1)*k]), batch_size=train_batch_size, drop_last=True, shuffle=True)]
            test_loaders += [NumpyLoader(MyDataset(test_samples[i*k:(i+1)*k], test_targets[i*k:(i+1)*k]), batch_size=test_batch_size, drop_last=True, shuffle=False)]
            train_loaders_def += [NumpyLoader(MyDataset(train_samples[i*k:(i+1)*k], train_targets[i*k:(i+1)*k]), batch_size=train_batch_size*k_batches, drop_last=True, shuffle=True)]
            test_loaders_def += [NumpyLoader(MyDataset(test_samples[i*k:(i+1)*k], test_targets[i*k:(i+1)*k]), batch_size=test_batch_size*k_batches, drop_last=True, shuffle=False)]
        return (train_loaders, test_loaders), (train_loaders_def, test_loaders_def), n_targets, (train_samples[:1], train_targets[:1])
