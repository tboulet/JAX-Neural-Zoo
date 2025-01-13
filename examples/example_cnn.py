import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax


from jaxzoo.cnn import JaxzooCNN
from jaxzoo.spaces import ContinuousSpace, ProbabilitySpace

# Create a random key
rng = jax.random.PRNGKey(0)
rng, subkey = jax.random.split(rng)

# Define the model
shape_x = (28, 28, 1)
model = JaxzooCNN(
    space_input=ContinuousSpace(shape_x),
    space_output=ProbabilitySpace(10),
    cnn_config={
        "hidden_dims": [32],
        "kernel_size": 3,
        "strides": 1,
        "name_activation_fn": "swish",
        "name_activation_output_fn": "linear",
    },
    dim_cnn_output=32,
    mlp_config={
        "hidden_dims": [32],
        "n_output_features": 10,
        "name_activation_fn": "swish",
        "name_activation_output_fn": "linear",
    },
)
variables = model.get_initialized_variables(key_random=subkey)
print(f"Model table summary : {model.get_table_summary()}")


# Define a loss function
def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=10)
    return -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))


# Create training state with Adam optimizer
optimizer = optax.adam(learning_rate=1e-3)
state = train_state.TrainState.create(
    apply_fn=model.apply_batched, params=variables["params"], tx=optimizer
)

# Dummy data batch
B = 32
batch = {
    "inputs": jax.random.normal(
        rng, (B, *shape_x)
    ),  # Batch of 32 samples, input size 784
    "labels": jax.random.randint(rng, (B,), 0, 10),  # Random integer labels
}

# Perform inference
pred = model.apply(variables=variables, x=batch["inputs"][0])
print(f"Performed inference. pred : {pred.shape}")

# Perform inference on batch
batch_preds = model.apply_batched(
    variables=variables, x=batch["inputs"], key_random=subkey
)
print(f"Performed batch inference. batch_preds : {batch_preds.shape}")


# Perform one gradient update step
@jax.jit
def train_step(state: train_state.TrainState, batch):
    def loss_fn(params):
        logits = state.apply_fn(variables={"params": params}, x=batch["inputs"])
        loss = cross_entropy_loss(logits, batch["labels"])
        return loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


state = train_step(state, batch)
print("Performed one gradient step.")
