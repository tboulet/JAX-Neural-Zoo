from jaxzoo.spaces import DiscreteSpace, ContinuousSpace, TupleSpace, DictSpace
from jax import random
import jax.numpy as jnp

def test_spaces():
    # Test the spaces
    key_random = random.PRNGKey(0)
    discrete = DiscreteSpace(5)
    assert discrete.contains(jnp.array(0))
    assert discrete.contains(jnp.array(1))
    assert discrete.contains(jnp.array(2))
    assert discrete.contains(jnp.array(3))
    assert discrete.contains(jnp.array(4))
    assert not discrete.contains(jnp.array(5))
    assert not discrete.contains(jnp.array(0.5))
    assert not discrete.contains(jnp.array(-1))
    assert discrete.sample(key_random) in [0, 1, 2, 3, 4]

    continuous = ContinuousSpace((), -1, 1)
    assert continuous.contains(jnp.array(0))
    assert continuous.contains(jnp.array(0.5))
    assert continuous.contains(jnp.array(-0.5))
    assert not continuous.contains(jnp.array(1.5))
    assert not continuous.contains(jnp.array(-1.5))
    assert continuous.sample(key_random) >= -1
    assert continuous.sample(key_random) <= 1
    assert continuous.contains(continuous.sample(key_random))
    c = ContinuousSpace((2, 3), jnp.array([[0, jnp.nan, 0], [0, 0, 0]]), jnp.array([[0, 1, 1], [1, 1, jnp.inf]]))
    assert c.contains(c.sample(key_random))
    c = ContinuousSpace((2, 3), -jnp.inf, jnp.array([[1, 1, 1], [-1, 1, jnp.inf]]))
    assert c.contains(c.sample(key_random))
    
    tuple_space = TupleSpace((discrete, continuous))
    assert tuple_space.contains((jnp.array(0), jnp.array(0)))
    assert tuple_space.contains((jnp.array(4), jnp.array(1)))
    assert not tuple_space.contains((jnp.array(5), jnp.array(0)))
    assert not tuple_space.contains((jnp.array(0), jnp.array(2)))
    assert tuple_space.contains(tuple_space.sample(key_random))

    dict_space = DictSpace({"discrete": discrete, "continuous": continuous})
    assert dict_space.contains({"discrete": jnp.array(0), "continuous": jnp.array(0)})
    assert dict_space.contains({"discrete": jnp.array(4), "continuous": jnp.array(1)})
    assert not dict_space.contains({"discrete": jnp.array(5), "continuous": jnp.array(0)})
    assert not dict_space.contains({"discrete": jnp.array(0), "continuous": jnp.array(2)})
    assert dict_space.contains(dict_space.sample(key_random))