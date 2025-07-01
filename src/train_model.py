# train_model.py

import torch
import matplotlib.pyplot as plt
from ecosystem import Grid, Species, Ecosystem
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math


def train_species_model(ecosystem, hidden_size=10, epochs=500, lr=0.01):
    if ecosystem.tensor is None:
        raise ValueError("Ecosystem must be sampled before training.")

    env_vals = []
    species_outputs = [[] for _ in ecosystem.species]

    for y in range(ecosystem.grid.height):
        for x in range(ecosystem.grid.width):
            cell = ecosystem.grid.get_cell(x, y)
            env_vals.append([cell.env_value])
            for i in range(len(ecosystem.species)):
                species_outputs[i].append(ecosystem.tensor[i, y, x])

    X = torch.tensor(env_vals, dtype=torch.float32)
    models = []

    for i, species in enumerate(ecosystem.species):
        y = torch.tensor(species_outputs[i], dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        model = nn.Sequential(
            nn.Linear(X.shape[1], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus()  # ensures output > 0
        )

        optimizer = optim.Adam(model.parameters(), lr=lr)
        # loss_fn = nn.MSELoss()
        loss_fn = torch.nn.PoissonNLLLoss(log_input=False)


        for epoch in range(epochs):
            for batch_X, batch_y in loader:
                pred = model(batch_X)
                loss = loss_fn(pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        models.append(model)

    return models


def plot_prediction_scatter(ecosystem, models):
    import matplotlib.pyplot as plt

    if ecosystem.tensor is None:
        raise ValueError("Ecosystem must be sampled first.")

    env_vals = []
    true_vals = [[] for _ in ecosystem.species]
    pred_vals = [[] for _ in ecosystem.species]

    for y in range(ecosystem.grid.height):
        for x in range(ecosystem.grid.width):
            cell = ecosystem.grid.get_cell(x, y)
            env_input = torch.tensor([[cell.env_value]], dtype=torch.float32)
            env_vals.append(cell.env_value)

            for i, model in enumerate(models):
                true = ecosystem.lambda_tensor[i, y, x]
                lam = model(env_input).item()
                true_vals[i].append(true)
                pred_vals[i].append(lam)

    num_species = len(ecosystem.species)
    fig, axes = plt.subplots(1, num_species, figsize=(5 * num_species, 4))

    if num_species == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.scatter(true_vals[i], pred_vals[i], alpha=0.7)
        ax.plot([0, max(true_vals[i])], [0, max(true_vals[i])], 'r--', label='Ideal')
        ax.set_xlabel("True Count")
        ax.set_ylabel("Predicted λ")
        ax.set_title(f"{ecosystem.species[i].name}")
        ax.legend()
        ax.grid(True)

    plt.suptitle("Predicted λ vs. True Count")
    plt.tight_layout()
    plt.show()




def main():
    # Re-create ecosystem
    grid = Grid(20, 20)

    grass = Species("Grass", lambda cell: 5 * cell.env_value)
    rabbit = Species("Rabbit", lambda cell: .3 * np.ceil(cell.env_value - 0.5) + .5*cell.env_value + cell.env_value**0.6)
    fox = Species("Fox", lambda cell: (cell.env_value - 0.5) ** 2 + 1.2)

    ecosystem = Ecosystem(grid, [grass, rabbit, fox])
    ecosystem.sample()



    # Train models
    models = train_species_model(ecosystem)

    plot_prediction_scatter(ecosystem, models)


if __name__ == "__main__":
    main()
