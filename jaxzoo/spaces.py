from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import random


class Space(ABC):
    """The base class for any space. A space describes the valid domain of a variable."""

    @abstractmethod
    def sample(self, key_random: jnp.ndarray) -> Any:
        """Sample a value from the space.

        Args:
            key_random (jnp.ndarray): the random key_random

        Returns:
            Any: the sampled value
        """
        pass

    @abstractmethod
    def contains(self, x: Any) -> bool:
        """Check if a value is in the space.

        Args:
            x (Any): the value to check

        Returns:
            bool: whether the value is in the space
        """
        pass
    
    @abstractmethod
    def accept(self, x : jax.Array) -> bool:
        """Check if the value has the same structure as the space.
        This doesn't mean that the value is in the space, but that it has the same structure.

        Args:
            x (jax.Array): the value to check

        Returns:
            bool: whether the value has the same structure as the space
        """
        pass
    
    @abstractmethod
    def get_list_spaces_and_values(
        self, x: Any
    ) -> List[Tuple["Space", jnp.ndarray]]:
        """Flatten the input x to a list of spaces and values."""
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimension of the space. The dimension is the number of values needed to represent the space."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass


class DiscreteSpace(Space):

    def __init__(self, n: int):
        """A discrete space with n possible values.

        Args:
            n (int): the number of possible values
        """
        assert n > 0, "The number of possible values must be positive."
        assert type(n) == int, "The number of possible values must be an integer."
        self.n = n

    def sample(self, key_random: jnp.ndarray) -> int:
        """Sample a value from the space.

        Args:
            key_random (jnp.ndarray): the random key_random

        Returns:
            int: the sampled value
        """
        return random.randint(key_random, (), 0, self.n)

    def contains(self, x: jax.Array) -> bool:
        """Check if a value is in the space.

        Args:
            x (jax.Array): the value to check

        Returns:
            bool: whether the value is in the space
        """
        assert isinstance(x, jax.Array), "The value must be a JAX array."
        return jnp.logical_and(0 <= x < self.n, jnp.equal(x, jnp.floor(x)))

    def accept(self, x : jax.Array) -> bool:
        return jnp.isscalar(x)
        
    def get_list_spaces_and_values(self, x: int) -> List[Tuple["Space", int]]:
        return [(self, x)]

    def get_dimension(self) -> int:
        return 1

    def __repr__(self) -> str:
        return f"Discrete({self.n})"


class ContinuousSpace(Space):

    def __init__(
        self,
        shape: Union[int, Tuple[int]],
        low: Union[float, jax.Array, None] = None,
        high: Union[float, jax.Array, None] = None,
    ):
        """A continuous space with a shape and bounds. 

        Args:
            shape (Union[int, Tuple[int]]): the shape of the space, as a tuple of non-negative integers (or a single integer for 1D spaces)
            low (float): the lower bound of the space, either as :
            - a float for a common value for all dimensions
            - a jax.Array for a specific value for each dimension. jnp.nan/-jnp.inf values will be interpreted as no lower bound
            - None for no lower bound
            high (float): the upper bound of the space, with the same format as low
        """
        if isinstance(shape, int):
            self.shape = (shape,)
        elif isinstance(shape, tuple):
            self.shape = shape
        else:
            raise ValueError("The shape must be an integer or a tuple of integers.")
        self.low = self.get_bound(low)
        self.high = self.get_bound(high, is_high=True)
        assert jnp.all(self.low <= self.high), "The low bound must be lower than the high bound."

    def get_bound(self, bound: Union[float, jax.Array, None], is_high: bool = False):
        direction = 1 if is_high else -1
        if bound is None:
            return  jnp.full(self.shape, direction * jnp.inf)
        elif isinstance(bound, (float, int)):
            # Assert there is no jnp.inf (if low bound) or -jnp.inf (if high bound)
            if is_high:
                assert bound != -jnp.inf, "The high bound cannot be -inf."
            else:
                assert bound != jnp.inf, "The low bound cannot be inf."
            return jnp.full(self.shape, bound)
        elif isinstance(bound, jax.Array):
            # Assert the shape is correct
            assert jnp.shape(bound) == self.shape, f"The shape of the bound is {jnp.shape(bound)}, but it should be {self.shape}."
            # Assert there is no jnp.inf (if low bound) or -jnp.inf (if high bound)
            if is_high:
                assert jnp.all(bound != -jnp.inf), "The high bound cannot be -inf."
            else:
                assert jnp.all(bound != jnp.inf), "The low bound cannot be inf."
            # Replace all jnp.nan values by +/- jnp.inf
            bound = jnp.where(jnp.isnan(bound), direction * jnp.inf, bound)
            return bound
        raise ValueError(
            f"The bound must be a float, a JAX array or None, not a {type(bound)}."
        )
        
    def sample(self, key_random: jnp.ndarray) -> float:
        """Sample a value from the space.

        Args:
            key_random (jnp.ndarray): the random key_random

        Returns:
            float: the sampled value
        """
        res = jnp.empty(self.shape)
        # On bounds [a, b], sample uniformly in [a, b]
        where_ab = jnp.logical_and(self.low != -jnp.inf, self.high != jnp.inf)
        res = jnp.where(
            where_ab,
            random.uniform(key_random, self.shape, minval=self.low, maxval=self.high),
            res,
        )
        # On bounds [a, inf], sample with exponential distribution
        where_a = jnp.logical_and(self.low != -jnp.inf, self.high == jnp.inf)
        res = jnp.where(
            where_a,
            random.exponential(key_random, self.shape) + self.low,
            res,
        )
        # On bounds [-inf, b], sample with exponential distribution
        where_b = jnp.logical_and(self.low == -jnp.inf, self.high != jnp.inf)
        res = jnp.where(
            where_b,
            -random.exponential(key_random, self.shape) + self.high,
            res,
        )
        # On bounds [-inf, inf], sample with standard normal distribution
        where_inf = jnp.logical_and(self.low == -jnp.inf, self.high == jnp.inf)
        res = jnp.where(
            where_inf,
            random.normal(key_random, self.shape),
            res,
        )
        # On bounds [-inf, b], sample uniformly in [-inf, b]
        return res

    def contains(self, x: float) -> bool:
        """Check if a value is in the space.

        Args:
            x (float): the value to check

        Returns:
            bool: whether the value is in the space
        """
        # Assert x is a JAX array
        assert isinstance(x, jax.Array), "The value must be a JAX array."
        # Check shape
        if not jnp.shape(x) == self.shape:
            return False
        # Check bounds
        return jnp.all(jnp.logical_and(self.low <= x, x <= self.high))
        
    def accept(self, x : jax.Array) -> bool:
        # x must be a JAX array
        if not isinstance(x, jax.Array):
            return False
        # Check shape
        if not jnp.shape(x) == self.shape:
            return False
        return True
    
    def get_list_spaces_and_values(self, x: float) -> List[Tuple["Space", float]]:
        return [(self, x)]

    def get_dimension(self) -> int:
        return jnp.prod(jnp.array(self.shape))

    def __repr__(self) -> str:
        minval = self.low if jnp.any(self.low != -jnp.inf) else "-inf"
        maxval = self.high if jnp.any(self.high != jnp.inf) else "inf"
        return f"Continuous({self.shape} in [{minval}, {maxval}])"


class TupleSpace(Space):

    def __init__(self, tuple_space: Tuple[Space]):
        """A space that is the cartesian product of multiple spaces.

        Args:
            tuple_spaces (Tuple[EcojaxSpace]): the spaces to combine
        """
        self.tuple_spaces = tuple_space

    def sample(self, key_random: jnp.ndarray) -> Tuple[Any]:
        return tuple(space.sample(key_random) for space in self.tuple_spaces)

    def contains(self, x: Tuple[Any]) -> bool:
        if not isinstance(x, tuple):
            return False
        return all(space.contains(x[i]) for i, space in enumerate(self.tuple_spaces))

    def accept(self, x : Tuple[jax.Array]) -> bool:
        # x must be a tuple
        if not isinstance(x, tuple):
            return False
        # Check the length
        if not len(x) == len(self.tuple_spaces):
            return False
        # Check each element
        return all(space.accept(x[i]) for i, space in enumerate(self.tuple_spaces))
    
    def get_list_spaces_and_values(
        self, x: Tuple[Any]
    ) -> List[Tuple["Space", Any]]:
        list_spaces_and_values = []
        for i, space in enumerate(self.tuple_spaces):
            list_spaces_and_values += space.get_list_spaces_and_values(x[i])
        return list_spaces_and_values

    def get_dimension(self) -> int:
        return sum([space.get_dimension() for space in self.tuple_spaces])
    
    def __repr__(self) -> str:
        return f"TupleSpace({', '.join([str(space) for space in self.tuple_spaces])})"


class DictSpace(Space):

    def __init__(self, dict_space: Dict[str, Space]):
        """A space that is the dictionary of multiple spaces.

        Args:
            dict_space (Dict[str, EcojaxSpace]): the spaces to combine
        """
        self.dict_space = dict_space

    def sample(self, key_random: jnp.ndarray) -> Dict[str, Any]:
        return {key: space.sample(key_random) for key, space in self.dict_space.items()}

    def contains(self, x: Dict[str, Any]) -> bool:
        if not isinstance(x, dict):
            return False
        return all(space.contains(x[key]) for key, space in self.dict_space.items())

    def accept(self, x : Dict[str, jax.Array]) -> bool:
        # x must be a dict
        if not isinstance(x, dict):
            return False
        # Check the keys
        if not set(x.keys()) == set(self.dict_space.keys()):
            return False
        # Check each element
        return all(space.accept(x[key]) for key, space in self.dict_space.items())
    
    def get_list_spaces_and_values(
        self, x: Dict[str, Any]
    ) -> List[Tuple["Space", Any]]:
        list_spaces_and_values = []
        for key, space in self.dict_space.items():
            list_spaces_and_values += space.get_list_spaces_and_values(x[key])
        return list_spaces_and_values

    def get_dimension(self) -> int:
        return sum([space.get_dimension() for space in self.dict_space.values()])
    
    def __repr__(self) -> str:
        return f"DictSpace({', '.join([f'{key}: {str(space)}' for key, space in self.dict_space.items()])})"


class ProbabilitySpace(ContinuousSpace):

    def __init__(self, shape: Union[int, Tuple[int]]):
        """A probability space, i.e. a continuous space with values in [0, 1] and summing to 1."""
        assert (
            isinstance(shape, int) or len(shape) == 1
        ), "The shape of the probability space must be an integer or a tuple of length 1."
        super().__init__(shape, 0, 1)

    def sample(self, key_random: jnp.ndarray) -> jnp.ndarray:
        x = super().sample(key_random)
        return jax.nn.softmax(x)

    def contains(self, x: jnp.ndarray) -> bool:
        return super().contains(x) and jnp.allclose(jnp.sum(x), 1)

    def __repr__(self) -> str:
        return f"ProbabilitySpace({self.shape})"
    
    
if __name__ == "__main__":
    # Test the spaces
    key_random = random.PRNGKey(0)
    c = ContinuousSpace((2, 3), jnp.inf, jnp.array([[1, 1, 1], [1, 1, jnp.inf]]))
    assert c.contains(c.sample(key_random))
