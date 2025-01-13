from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type

import jax
import numpy as np

import flax.linen as nn
from jax import random
import jax.numpy as jnp

from jaxzoo.spaces import DiscreteSpace, ContinuousSpace, ProbabilitySpace, Space, TupleSpace, DictSpace
from jaxzoo.utils import warning_message

name_activation_fn_to_fn = {
    "relu": nn.relu,
    "sigmoid": nn.sigmoid,
    "tanh": nn.tanh,
    "leaky_relu": nn.leaky_relu,
    "elu": nn.elu,
    "selu": nn.selu,
    "gelu": nn.gelu,
    "swish": nn.swish,
    "identity": lambda x: x,
    "linear": lambda x: x,
    None: lambda x: x,
}


class JaxzooBaseModel(nn.Module, ABC):
    """The base class for all models. A model is a way to map input from a certain space to an output in another space.
    This abstract class subclasses nn.Module, which is the base class for all Flax models.

    For subclassing this class, users need to add the dataclass parameters required for their model and implement the process_input method.
    The process_input method should map the input to either :
    - an encoding of shape (d_encoding,) with d_encoding >= 2 that will be transformed to the output space
    - a pytree that belongs to the output space
    
    Args:
        space_input (EcojaxSpace): the input space of the model
        space_output (EcojaxSpace): the output space of the model
    """

    space_input: Space
    space_output: Space

    # ==== Abstract methods ====
    @abstractmethod
    def process_input(self, x, key_random: jnp.ndarray = None) -> jnp.ndarray:
        """The call function of the model. It maps the input to an encoding.
        If that encoding is not in the output space, the JaxzooBaseModel will transform it, eventually adding parameters, to make it fit the output space.
        """
        raise NotImplementedError

    # ==== Main methods ====
    
    def get_initialized_variables(
        self,
        key_random: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """Initializes the model's variables and returns them as a dictionary.
        This is a wrapper around the init method of nn.Module, which creates an observation for initializing the model.
        """
        # Sample the observation from the different spaces
        key_random, subkey = random.split(key_random)
        x = self.space_input.sample(subkey)

        # Initialize the model
        key_random, subkey = random.split(key_random)
        variables = nn.Module.init(
            self,
            key_random,
            x=x,
            key_random=subkey,
        )
                
        # Return the variables
        return variables

    def process_encoding(
        self, x: jnp.ndarray, key_random: jnp.ndarray = None
    ) -> jnp.ndarray:
        """Processes the encoding obtained through the submodel (method process_input) to obtain the output in the right format.

        It does the following :
        - if the encoding x is already in the output space, it returns it
        - if not, it checks that x is an encoding (i.e. of shape (d_encoding,) with d_encoding >= 2) and raise an error if not
        - process the encoding to make it fit the output space. 
            If x have the same dimension as certain output space, they will be directly returned.
            Else, a dense layer will be applied to obtain the right shape.
        """
        # If x is already in the output space, return it
        if self.space_output.accept(x):
            return x
        
        # Assert that x is an array and not a more complex structure
        assert isinstance(x, jnp.ndarray), (
            f"The output of the process_input method must either be:\n"
            "   - a encoding (a jnp.ndarray of shape (d_encoding,)), in this case it will be transformed to the output space\n"
            "   - a pytree that belong to the output space, in this case it will be returned as is\n"
            "Pytree are not supported for the output of the process_input method unless they are in the output space."
        )
        # Assert that x is not of shape () or (1,)
        if x.shape == () or x.shape == (1,):
            if self.space_output.get_dimension() == 1:
                raise ValueError(
                    f"The output of the process_input method is of shape {x.shape} which is not in the output space. Please make it belong to the output space {self.space_output}."
                )
            else:
                raise ValueError(
                    f"The output of the process_input method is of shape {x.shape}, but expected a shape of (d_encoding,) with d_encoding >= 2."
                )
        # Assert that x is not of dimension 2 or more
        assert (
            len(x.shape) == 1
        ), f"The output of the process_input method is of shape {x.shape}, but expected a shape of (d_encoding,) with d_encoding >= 2."

        # Process the encoding to obtain the output
        d_encoding = x.shape[0]
        
        # For discrete space, apply a dense layer if shape-needed and then sample
        if isinstance(self.space_output, DiscreteSpace):
            assert (
                key_random is not None
            ), "key_random must be provided for random operations"
            if d_encoding == self.space_output.n and ProbabilitySpace(shape = (self.space_output.n,)).contains(x):
                raise ValueError("The output of the process_input is of same length as the number of classes, and is a probability distribution in this space, but it should output logits. We assume this is an error, please provide logits.")
            elif d_encoding == self.space_output.n:
                warning_message(message_warning=f"The output of the process_input of shape {x.shape} is of same length as the number of classes {self.space_output.n}. This will be given as logits. We assume this is the expected behavior.")
            else:
                warning_message(message_warning=f"The output of the process_input of shape {x.shape} is not of same length as the number of classes {self.space_output.n}. A dense layer will be applied to obtain logits.")
                x = nn.Dense(
                    features=self.space_output.n,
                )(x)
            key_random, subkey = random.split(key_random)
            output = random.categorical(
                subkey, x
            )  # crashes here because non supported operation
                    
        # For continuous space, apply a dense layer if shape-needed
        elif isinstance(self.space_output, ContinuousSpace):
            shape_output = self.space_output.shape
            # (d_encoding,) -> ?
            if len(shape_output) == 1:
                d_output = shape_output[0]
                # (d_encoding,) -> (n,)
                if d_output == d_encoding:
                    # (d_encoding,) -> (d_encoding,) : just return the value
                    output = x
                else:
                    # (d_encoding,) -> (d_output,) : apply dense layer
                    warning_message(message_warning="The output of the process_input is not of same length as the output space. A dense layer will be applied to obtain the right shape.")
                    output = nn.Dense(
                        features=d_output,
                        # bias_init=nn.initializers.ones_init(), # TODO : optionalize this, and find a way so that zeros doesnt create NaNs
                        bias_init=nn.initializers.zeros_init(),
                    )(x)
            elif len(shape_output) == 0:
                # (d_encoding,) -> ()
                if len(x.shape) == 1:
                    # (1,) -> () : just extract the value
                    output = x[0]
                else:
                    # (d_encoding,) -> () : apply dense layer and extract the value
                    output = nn.Dense(
                        features=1,
                        bias_init=nn.initializers.ones_init(),
                    )(x)[0]
            else:
                raise NotImplementedError(
                    f"Processing of continuous space of shape {shape_output} is not implemented."
                )
            
            # For in particular probability space, apply a softmax
            if isinstance(self.space_output, ProbabilitySpace):
                output = nn.softmax(output)
                
        # For tuple and dict space, apply the process_encoding to each element
        elif isinstance(self.space_output, TupleSpace):
            if key_random is not None:
                subkeys = random.split(key_random, len(self.space_output.tuple_spaces))
            else:
                subkeys = [None] * len(self.space_output.tuple_spaces)
            output = tuple(
                self.process_encoding(x, subkeys[i])
                for i in range(len(self.space_output.tuple_spaces))
            )
        elif isinstance(self.space_output, DictSpace):
            if key_random is not None:
                subkeys = random.split(key_random, len(self.space_output.dict_space))
            else:
                subkeys = [None] * len(self.space_output.dict_space)
            output = {
                key: self.process_encoding(x, subkeys[i])
                for i, key in enumerate(self.space_output.dict_space.keys())
            }
        else:
            raise ValueError(
                f"Unknown space type for output: {type(self.space_output)}"
            )
        
        return output

    @nn.compact
    def __call__(
        self,
        x: Any,
        key_random: jnp.ndarray = None,
    ) -> Tuple[jnp.ndarray]:
        """The forward pass of the model. It maps the observation to the output in the right format.

        Args:
            x (Any) : input observation
            key_random (jnp.ndarray): the random key used for any random operation in the forward pass

        Returns:
            Tuple[jnp.ndarray]: a tuple of the requested outputs
        """
        # Convert the input to adapted structure
        encoding = self.process_input(x, key_random)

        # Return the output in the desired output space
        output = self.process_encoding(encoding, key_random)
        
        # Assert that the output is in the output space
        assert self.space_output.accept(output), (
            f"The output of the model {output} is not in the format of the output space {self.space_output}."
        )
        
        # Return the output
        return output
    
    # ==== Helper functions ====
    def apply_batched(self, variables: Dict[str, jnp.ndarray], x: jnp.ndarray, key_random: jnp.ndarray = None) -> jnp.ndarray:
        """Apply the model to a batch of inputs, batching key_random for unique randomness per sample."""
        batch_size = x.shape[0]
        
        # Split key_random into unique keys for each sample in the batch
        keys = jax.random.split(key_random, batch_size) if key_random is not None else [None] * batch_size

        return jax.vmap(
            lambda x_i, key_i: self.apply(variables=variables, x=x_i, key_random=key_i),
            in_axes=0,  # Vectorize over both 'x' and 'key_random'
        )(x, keys)
    
    def activation_fn(self, name_activation_fn, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the activation function to the input."""
        return name_activation_fn_to_fn[name_activation_fn](x)

    def get_table_summary(self) -> Dict[str, Any]:
        """Returns a table that summarizes the model's parameters and shapes."""
        key_random = jax.random.PRNGKey(0)
        x = self.space_input.sample(key_random)
        return nn.tabulate(self, rngs=key_random)(x, key_random)