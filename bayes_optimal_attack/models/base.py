from typing import List
from jax._src.nn.functions import sigmoid

import numpy as np
import itertools
import jax
import jax.numpy as jnp



##### To be moved into losses.py #####
def softmax_cross_entropy(logits, labels):
    one_hot = jax.nn.one_hot(labels, logits.shape[-1])
    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)


# Simple example
def get_network(name: str, is_training: bool = True):
    split = name.split("_")
    net_name = split[0]
    layer_list = split[1:]
    layer_list = list(map(lambda x: int(x), layer_list))

    if net_name == "mlp":
        def func(x):
            x = x.reshape(x.shape[0], -1)
            f = hk.nets.MLP(layer_list)  # , hk.initializers.UniformScaling, hk.initializers.UniformScaling, activation=jax.nn.sigmoid)
            return f(x)
        return hk.transform_with_state(func)  # For consistent interface with ResNet

    elif net_name == "resnet":
        assert len(layer_list) == 2, f"ResNet with multiple size arguments: {layer_list}"

        def func(x):
            if layer_list[0] == 18:
                res_net = hk.nets.ResNet18(layer_list[1], resnet_v2=True)
            elif layer_list[0] == 34:
                res_net = hk.nets.ResNet34(layer_list[1], resnet_v2=True)
            elif layer_list[0] == 50:
                res_net = hk.nets.ResNet50(layer_list[1], resnet_v2=True)
            elif layer_list[0] == 101:
                res_net = hk.nets.ResNet101(layer_list[1], resnet_v2=True)
            elif layer_list[0] == 152:
                res_net = hk.nets.ResNet152(layer_list[1], resnet_v2=True)
            elif layer_list[0] == 200:
                res_net = hk.nets.ResNet200(layer_list[1], resnet_v2=True)

            f = res_net
            # TODO move this to the function call itself
            return f(x, is_training=is_training)

        return hk.transform_with_state(func)

    elif net_name == "conv":
        def mlp_fun(input: np.ndarray) -> jnp.ndarray:
            mlp = hk.Sequential([
                hk.Conv2D(12, kernel_shape=(5,5), padding="SAME", stride=2), jax.nn.sigmoid,
                hk.Conv2D(12, kernel_shape=(5,5), padding="SAME", stride=2), jax.nn.sigmoid,
                #hk.Conv2D(12, kernel_shape=(5,5), padding="SAME", stride=2), jax.nn.sigmoid,
                hk.Flatten(),
                #hk.Linear(588), jax.nn.sigmoid, 
                #hk.Linear(100), jax.nn.sigmoid,
                hk.Linear(10),
            ])

            return mlp(input)

        mlp_map = hk.transform_with_state(mlp_fun)

    #  mlp = list(itertools.chain.from_iterable([ [hk.Linear(j), jax.nn.relu] for j in layers[:-1]] + [[hk.Linear(layers[-1])]])))
    return mlp_map
