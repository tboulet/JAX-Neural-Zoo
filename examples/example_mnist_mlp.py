import os
import random
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
import optax
from flax.training import train_state

from jaxzoo.base_model import JaxzooBaseModel
from jaxzoo.mlp import JaxzooMLP
from jaxzoo.spaces import ContinuousSpace, ProbabilitySpace

# Create a random key
seed = random.randint(0, 1000)
rng = jax.random.PRNGKey(seed=seed)
rng, subkey = jax.random.split(rng)

# Define the model
model = JaxzooMLP(
    space_input=ContinuousSpace((28, 28)),
    space_output=ProbabilitySpace(10),
    hidden_dims=[32],
    name_activation_fn="swish",
)
variables = model.get_initialized_variables(key_random=subkey)
print(f"Model table summary : {model.get_table_summary()}")

# Load the MNIST dataset using PyTorch DataLoader (still using torchvision)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_data = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

# Convert to DataLoader
train_loader = DataLoader(train_data, batch_size=5, shuffle=True)
test_loader = DataLoader(test_data, batch_size=5, shuffle=False)


# Define loss and accuracy functions
def compute_loss(probs, labels):
    labels_one_hot = jax.nn.one_hot(labels, num_classes=10)
    loss = -jnp.mean(jnp.sum(labels_one_hot * jax.nn.log_softmax(probs), axis=-1))
    return loss

def compute_accuracy(preds, labels):
    preds = jnp.argmax(preds, axis=-1)
    return jnp.mean(preds == labels)


# Using Adam optimizer from JAX
optimizer = optax.adam(learning_rate=1e-3)
opt_state = train_state.TrainState.create(
    apply_fn=model.apply_batched, params=variables["params"], tx=optimizer
)

# TensorBoardX for logging
import datetime

tb_writer = SummaryWriter(
    f"tensorboard/MNIST_MLP_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_seed{seed}"
)


# Training step
@jax.jit
def train_step(opt_state: train_state.TrainState, params, batch):
    images, labels = batch
    # Define the loss function wrt params
    def loss_fn(params):
        preds = opt_state.apply_fn(variables={"params": params}, x=images)
        loss = compute_loss(preds, labels)
        return loss

    # Compute the loss and gradients
    loss, grads = jax.value_and_grad(loss_fn)(opt_state.params)
    opt_state_new = opt_state.apply_gradients(grads=grads)

    # Compute the accuracy
    preds = opt_state_new.apply_fn(variables={"params": params}, x=images)
    accuracy = compute_accuracy(preds, labels)
    return (
        opt_state_new,
        opt_state.params,
        loss,
        accuracy,
    )


# Training loop
def train(model, train_loader, params, opt_state, num_epochs=5):
    for epoch in range(num_epochs):
        for step, (images, labels) in enumerate(train_loader):
            # Train step
            images = jnp.array(images.numpy())
            labels = jnp.array(labels.numpy())
            opt_state, params, loss, accuracy = train_step(
                opt_state, params, (images, labels)
            )

            # Log the loss and accuracy using TensorBoardX
            if step % 100 == 0:  # Log every 100 steps
                tb_writer.add_scalar(
                    "Loss/train", loss, epoch * len(train_loader) + step
                )
                tb_writer.add_scalar(
                    "Accuracy/train", accuracy, epoch * len(train_loader) + step
                )
                print(
                    f"Epoch {epoch} Step {step} Loss {loss:.4f} Accuracy {accuracy:.4f}"
                )
        
        # Validation after each epoch
        validate(model, test_loader, params, epoch)


# Validation function to evaluate the model on the test dataset
def validate(model: JaxzooBaseModel, test_loader, params, epoch):
    all_preds = []
    all_labels = []
    for images, labels in test_loader:
        images = jnp.array(images.numpy())
        labels = jnp.array(labels.numpy())
        preds = model.apply_batched(variables={"params": params}, x=images)
        all_preds.append(jnp.argmax(preds, axis=-1))
        all_labels.append(jnp.argmax(labels, axis=-1))
    preds = jnp.concatenate(all_preds)
    labels = jnp.array(all_labels)
    accuracy = compute_accuracy(preds, labels)
    tb_writer.add_scalar("Accuracy/val", accuracy, epoch)
    print(f"Validation Accuracy: {accuracy:.4f}")


# Run the training
params = variables["params"]
train(model, train_loader, params, opt_state)

# Save the trained model (optional)
# jax.experimental.serialization.save_npz('model_params.npz', params)
