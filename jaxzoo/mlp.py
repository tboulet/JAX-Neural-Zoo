from abc import ABC, abstractmethod
from typing import Dict, List, Union
import numpy as np

import jax
from jax import random
import jax.numpy as jnp
from flax import struct
import flax.linen as nn

from jaxzoo.base_model import JaxzooBaseModel
from jaxzoo.spaces import Space, DiscreteSpace, ContinuousSpace, TupleSpace, DictSpace


class JaxzooMLP(JaxzooBaseModel):
    """A model that use MLP networks to process observations and generate actions.
    It does the following :
    - flatten and concatenate the observation components to obtain a single vector : input_space structure -> (n_features,)
    - process the concatenated output with an MLP : n_features, -> hidden_dims[0], -> hidden_dims[1], -> ... -> hidden_dims[-1]
    - generate the outputs for each output space through finals MLPs : (hidden_dims[-1],) -> output_space structure

    Args:
        hidden_dims (List[int]): the number of hidden units in each hidden layer. It also defines the number of hidden layers.
        name_activation_fn (str): the name of the activation function to use. Default is "swish".
    """

    hidden_dims: List[int]
    name_activation_fn: str

    def process_input(
        self,
        x: jax.Array,
        key_random: jnp.ndarray = None,
    ) -> jnp.ndarray:
        """Converts the input x to a vector encoding that can be processed by the MLP."""

        # Flatten and concatenate inputs
        list_spaces_and_values = self.space_input.get_list_spaces_and_values(x)
        list_vectors = []
        for space, x in list_spaces_and_values:
            if isinstance(space, ContinuousSpace):
                x = x.reshape((-1,))
                list_vectors.append(x)
            elif isinstance(space, DiscreteSpace):
                one_hot_encoded = jax.nn.one_hot(x, space.n)
                list_vectors.append(one_hot_encoded)
            else:
                raise ValueError(f"Unknown space type for input: {type(space)}")
        x = jnp.concatenate(list_vectors, axis=-1)

        # Process the concatenated output with a final MLP
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(
                features=hidden_dim,
            )(x)
            x = self.activation_fn(name_activation_fn=self.name_activation_fn, x=x)

        return x
