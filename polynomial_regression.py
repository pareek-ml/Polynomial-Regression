"""
Template for Assignment 1
"""

import numpy as np  # Is it version 2.1 the one you are running? Its 2.1.1 which supports polyval
import matplotlib.pyplot as plt
import torch  # Is it version 2.4 the one you are running?
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data


def plot_polynomial(coeffs, z_range, color="b"):
    x = np.linspace(start=z_range[0], stop=z_range[1], num=100)
    y = np.polynomial.polynomial.polyval(x=x, c=coeffs)
    plt.plot(x, y, color=color)
    plt.title("Polynomial plot")
    plt.xlabel("x")
    plt.ylabel("y")


def create_dataset(
    coeffs, z_range=(-1, 1), sample_size=100, sigma=0.1, seed=42, debug=False
):
    degree = coeffs.shape[0]  # Should be 5 in this case
    torch.manual_seed(seed)  # For reproducibility

    # Generate random z values within the specified range
    z = (
        torch.rand(sample_size, dtype=torch.float32) * (z_range[1] - z_range[0])
        + z_range[0]
    )

    # Initialize and populate the feature matrix X
    X = torch.zeros(
        size=(sample_size, degree), dtype=torch.float32
    )  # Shape: (sample_size, 5)
    for i in range(degree):
        X[:, i] = z**i  # Each column is z raised to a power from 0 to 4

    # Generate Gaussian noise
    noise = torch.normal(0, sigma, size=(sample_size, 1))  # Shape: (sample_size, 1)

    # Compute target values with added noise
    y = X @ coeffs + noise  # Matrix multiplication followed by addition of noise

    # Debugging shapes
    if debug:
        print("z shape:", z.shape)
        print("X shape:", X.shape)
        print("coeffs shape:", coeffs.shape)
        print("noise shape:", noise.shape)
        print("y shape:", y.shape)
    return X, y


def visualize_data(X, y, coeffs, z_range, title=""):

    z = X[:, 1].numpy()

    # Plot the true polynomial
    plot_polynomial(coeffs, z_range, color="b")

    # Plot the dataset as a scatter plot
    plt.scatter(z, y.numpy(), color="r", label="Data Points")

    # Add title, legend, and show plot
    plt.title(f"Plotting Polynomial vs {title}")
    plt.legend()
    plt.show()


class LinearRegressionModel:
    def __init__(self, input_dim, lr):
        """
        Initialize the linear regression model with given parameters.

        Args:
            input_dim: Number of input features
            lr: Learning rate for the optimizer
        """
        self.model = nn.Linear(input_dim, 1)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.SGD(params=self.model.parameters(), lr=lr)

    def train(self, X_train, X_val, y_train, y_val, steps):
        """
        Train the model using the provided training and validation data.

        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training targets
            y_val: Validation targets
            steps: Number of training steps
        """
        loss_train = []
        loss_val = []
        params_history = {"weights": [], "biases": []}
        for step in range(steps):
            self.model.train()  # Set the model in training mode
            # Set the optim gradient to 0
            self.optimizer.zero_grad()  # Reset the gradients of the model parameters
            y_hat = self.model(X_train)  # Perform a forward pass to compute predictions
            loss = self.loss_fn(y_hat, y_train)  # Compute the training loss
            loss.backward()  # Backpropagate the loss to compute gradients
            self.optimizer.step()  # Update the model parameters using the computed gradients

            params_history["weights"].append(self.model.weight.detach().clone())
            params_history["biases"].append(self.model.bias.detach().clone())

            self.model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Disable gradient computation for validation
                y_hat_val = self.model(X_val)
                val_loss = self.loss_fn(y_hat_val, y_val)  # Compute the validation loss
                # Store the loss
                loss_val.append(val_loss.item())
                loss_train.append(loss.item())

            if step % 20 == 0:  # Print the loss every 20 steps
                print("Step:", step, "- Loss eval:", val_loss.item())

        self.plot_loss(loss_train, loss_val)

        return self.model.weight, self.model.bias, params_history

    def plot_loss(self, loss_train, loss_val):
        plt.plot(range(len(loss_train)), loss_train)
        plt.plot(range(len(loss_val)), loss_val)
        plt.legend(["Training loss", "Validation loss"])
        plt.xlabel("Steps")
        plt.ylabel("Loss value")
        plt.show()


def main():
    """
    Code for Q1
    """
    assert (
        np.version.version >= "2.1"
    )  # I have optimise this as == sign caused an assertion error and numpy ver > 2.1 also have same capabilities

    """
    Code for Q2
    """
    coeffs = np.array([1, -1, 5, -0.1, 1 / 30], dtype=np.float32)
    z_range = (-4, 4)
    plot_polynomial(coeffs, z_range)
    plt.show()

    """
    Code for Q4: Create dataset
    """
    coeffs = torch.from_numpy(coeffs).view(
        -1, 1
    )  # Convert to column vector, as y = X[i].dot(c), and X[i] is row vector and c should be column vector
    X_train, y_train = create_dataset(
        coeffs, z_range=(-2, 2), sample_size=500, sigma=0.5, seed=0, debug=True
    )
    X_val, y_val = create_dataset(
        coeffs, z_range=(-2, 2), sample_size=500, sigma=0.5, seed=1
    )

    """
    Code for Q5: Visualize Training and Validation data
    """
    visualize_data(
        X_train,
        y_train.reshape(-1),
        coeffs.numpy().reshape(-1),
        z_range=(-2, 2),
        title="Training Data",
    )
    visualize_data(
        X_train,
        y_train.reshape(-1),
        coeffs.numpy().reshape(-1),
        z_range=(-2, 2),
        title="Validation Data",
    )

    """
    Code for Q6 and Q7: Train the model and plot the loss
    """
    lin_regression = LinearRegressionModel(input_dim=X_train.shape[1], lr=0.003)
    weights, bias, params = lin_regression.train(
        X_train, X_val, y_train, y_val, steps=100
    )
    print(weights, bias)

    """
    Code for Q8: plot the polynomial generated by the model and the true polynomial
    """
    weights.detach_()
    plot_polynomial(weights.numpy().reshape(-1), z_range, color="b")
    plot_polynomial(coeffs.numpy().reshape(-1), z_range, color="r")
    plt.show()

    """
    Code for Q9: plot the evolution of the weights and biases during training
    """
    weights_history = [weight.numpy().flatten() for weight in params["weights"]]
    biases_history = [bias.item() for bias in params["biases"]]

    plt.plot(weights_history)
    plt.xlabel("Steps")
    plt.ylabel("Weight values")
    plt.title("Evolution of Weight Parameters During Training")
    plt.show()

    plt.plot(biases_history)
    plt.xlabel("Steps")
    plt.ylabel("Bias value")
    plt.title("Evolution of Bias During Training")
    plt.show()



if __name__ == "__main__":
    main()

# Bonus Question

# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# from torch import nn
# from torch import optim

# def generate_data(a, n_samples=100, noise_std=0.01):
#     np.random.seed(0)
#     # Generate x_i in [-0.05, a]
#     x = np.random.uniform(-0.05, a, n_samples)
#     # Ensure x + 1 > 0 to avoid log of non-positive number
#     x = np.clip(x, -0.999, a)
#     # Compute f(x) = 2 * log(x + 1) + 3
#     y = 2 * np.log(x + 1) + 3
#     # Add Gaussian noise
#     noise = np.random.normal(0, noise_std, size=y.shape)
#     y_noisy = y + noise
#     return x.reshape(-1, 1), y_noisy

# def linear_regression(x_np, y_np, num_epochs=1000, learning_rate=0.005):
#     # Convert numpy arrays to torch tensors
#     x = torch.from_numpy(x_np.astype(np.float32))
#     y = torch.from_numpy(y_np.astype(np.float32)).view(-1, 1)
    
#     # Normalize the data
#     x_mean = x.mean()
#     x_std = x.std()
#     y_mean = y.mean()
#     y_std = y.std()
#     x_norm = (x - x_mean) / x_std
#     y_norm = (y - y_mean) / y_std

#     model = nn.Linear(1, 1)
#     loss_fn = nn.MSELoss()
#     optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
#     # Store losses for plotting
#     losses = []
    
#     # Training loop
#     for epoch in range(num_epochs):
#         # Zero the gradients
#         optimizer.zero_grad()
        
#         # Forward pass
#         outputs = model(x_norm)
        
#         # Compute loss
#         loss = loss_fn(outputs, y_norm)
        
#         # Backward pass
#         loss.backward()
        
#         # Update weights
#         optimizer.step()
        
#         # Record loss
#         losses.append(loss.item())
    
#     # After training, compute predictions
#     with torch.no_grad():
#         y_pred_norm = model(x_norm)
#         mse = loss_fn(y_pred_norm, y_norm).item()
#         # Convert predictions back to original scale
#         y_pred = y_pred_norm * y_std + y_mean
    
#     return model, mse, losses, y_pred.numpy().flatten()

# # Case 1: a = 0.01
# a1 = 0.01
# x1_np, y1_np = generate_data(a1, noise_std=0.01)
# model1, mse1, losses1, y1_pred = linear_regression(x1_np, y1_np)
# print(f"Case 1 (a = {a1}): MSE = {mse1:.6f}")

# # Plotting
# plt.figure(figsize=(15, 5))
# # Plot loss curve
# plt.subplot(1, 2, 1)
# plt.plot(losses1)
# plt.title(f'Loss Curve (a = {a1})')
# plt.xlabel('Epoch')
# plt.ylabel('MSE Loss')

# # Plot data and fit
# plt.subplot(1, 2, 2)
# plt.scatter(x1_np, y1_np, label='Noisy Data', color='blue')
# plt.plot(x1_np, y1_pred, color='red', label='Linear Fit')
# plt.title(f'Linear Regression Fit (a = {a1})')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()

# plt.tight_layout()
# plt.show()

# # Case 2: a = 10
# a2 = 10
# x1_np, y1_np = generate_data(a2, noise_std=0.01)
# model1, mse1, losses1, y1_pred = linear_regression(x1_np, y1_np)
# print(f"Case 1 (a = {a2}): MSE = {mse1:.6f}")

# # Plotting
# plt.figure(figsize=(15, 5))
# # Plot loss curve
# plt.subplot(1, 2, 1)
# plt.plot(losses1)
# plt.title(f'Loss Curve (a = {a1})')
# plt.xlabel('Epoch')
# plt.ylabel('MSE Loss')

# # Plot data and fit
# plt.subplot(1, 2, 2)
# plt.scatter(x1_np, y1_np, label='Noisy Data', color='blue')
# plt.plot(x1_np, y1_pred, color='red', label='Linear Fit')
# plt.title(f'Linear Regression Fit (a = {a1})')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()

# plt.tight_layout()
# plt.show()

