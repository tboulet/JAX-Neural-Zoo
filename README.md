# JaxZoo

```jaxzoo``` is a package built on top of JAX and Flax (the library for neural network in JAX) that provides a zoo of neural network models. It is designed to be easy to use and to be easily extensible. The models of JaxZoo adapt to the input and output space of the model, and can be easily created with one or two lines of code.

```python
model = JaxzooMLP(
    space_input=DictSpace({
        "figures" : TupleSpace([DiscreteSpace(10), DiscreteSpace(10)]),
        "embedding" : ContinuousSpace(64),
        "image" : ContinuousSpace((28, 28), low=0.0, high=1.0),
    }),
    space_output=ProbabilitySpace(10),
    hidden_dims=[32],
    name_activation_fn="swish",
    )
variables = model.get_initialized_variables(key_random=subkey)

print(f"Model table summary : {model.get_table_summary()}")
```

The 2 main feature of JaxZoo is its simplicity : you can easily create a simple model with one or two lines of code, and its flexibility : JaxZoo support any kind of input and output space, including discrete, continuous, hierarchical and automatically adapt to the input/output structure of the model.

## Installation

You will need to install numpy, JAX and Flax before installing JaxZoo.
```bash
pip install jaxzoo
```

## Quickstart

To create a simple MLP model that receives images of shape (32, 32, 3) and output a probability vector of shape (10,), you can do :
```python
from jaxzoo.mlp import JaxzooMLP

model = JaxzooMLP(
    space_input=ContinuousSpace((28, 28, 3)),
    space_output=ProbabilitySpace(10),
    hidden_dims=[32],
    name_activation_fn="swish",
    )
variables = model.get_initialized_variables(key_random=subkey)
print(f"Model table summary : {model.get_table_summary()}")
```

## Features

### Simple basic models

JaxZoo provides a zoo of simple models that can be easily created with one or two lines of code. The models are :
- ```JaxzooMLP``` : a simple MLP model
- ```JaxzooCNN``` : a simple CNN model

### Input space agnosticism

You can give any kind of input space to the model, including continuous, discrete, hierarchical, etc. The model will automatically adapt to the input space provided this one stays constant for the duration of the model use. 

For example, if your input is a dictionnary containing one tuple of 2 figures, one embedding vector and one image between 0 and 1, you can do :

```python
from jaxzoo.spaces import Space, ContinuousSpace, DiscreteSpace, TupleSpace, DictSpace

# Define the input space
space_input = DictSpace({
    "figures" : TupleSpace([DiscreteSpace(10), DiscreteSpace(10)]),
    "embedding" : ContinuousSpace(64),
    "image" : ContinuousSpace((28, 28), low=0.0, high=1.0),
})
```

How it work is that the model will treat hierarchically the input space, applying model-wise functions to each sub-input. For example, the JaxzooMLP model will flatten and concatenate each input components before applying the MLP layers, while the JaxzooCNN model will apply a CNN to the images and concatenate the embedding with non-image inputs before applying an MLP.

### Output space agnosticism

Similarly to the input space, you can give any kind of output space to the model, except 2D spaces and above which are not yet supported. 

For example, if your output is a dictionnary containing one probability vector of size 10 and one embedding of size 2, you can do :

```python
from jaxzoo.spaces import Space, ContinuousSpace, DiscreteSpace, TupleSpace, DictSpace, ProbabilitySpace

# Define the output space
space_output = DictSpace({
    "probability" : ProbabilitySpace(10),
    "embedding" : ContinuousSpace(2, low=jnp.array([-1.0, jnp.nan]), high=jnp.inf),
})
```

How it work is that each model treat the input to produce an embedding vector. If this embedding is already adapted to the output space, the model will directly output the embedding. Otherwise, the model will apply for each sub-output space a space-wise operation to the embedding to produce the output. For example, shape constraints will lead to a dense layer, probability space will lead to applying a softmax, etc.

### Stochastic models

JaxZoo support stochastic models, ie models that take a random key as input, as this may be the case for some models. The key is passed as an additional argument to the model, and is used to generate random numbers in the model.

```python
key_random = jax.random.PRNGKey(0)
key_random, subkey = jax.random.split(key_random)
pred = model.apply(variables=variables, x=x, key_random=subkey)
```

### Model summary

You can get a summary of the model by calling the ```get_table_summary``` method. This will give you a table with the input and output spaces, the number of parameters, the number of layers, etc.

```python
print(f"Model table summary : {model.get_table_summary()}")
```

### Inference for single data and batched data

You can apply the model to a single data point with method ```apply``` or to a batch of data points with method ```apply_batched```.

```python
# Single data point
pred = model.apply(variables=variables, x=x_batch[0], key_random=subkey)

# Batch of data points
pred = model.apply_batched(variables=variables, x=x_batch, key_random=subkey)
```