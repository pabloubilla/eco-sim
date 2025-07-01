import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class Cell:
    def __init__(self, env_value=1.0):
        self.env_value = env_value
        self.populations = {}

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.cells = [[Cell(random.uniform(0.5, 2.0)) for _ in range(width)] for _ in range(height)]

    def get_cell(self, x, y):
        return self.cells[y][x]

class Species:  
    def __init__(self, name, lambda_fn):
        self.name = name
        self.lambda_fn = lambda_fn

class Ecosystem:
    def __init__(self, grid, species_list):
        self.grid = grid
        self.species = species_list
        self.tensor = None  # shape: (num_species, height, width)

    def sample(self):
        height, width = self.grid.height, self.grid.width
        num_species = len(self.species)
        self.tensor = np.zeros((num_species, height, width), dtype=int)
        self.lambda_tensor = np.zeros((num_species, height, width), dtype=float)


        for y in range(height):
            for x in range(width):
                cell = self.grid.get_cell(x, y)
                for s_idx, species in enumerate(self.species):
                    lam = species.lambda_fn(cell)
                    pop = np.random.poisson(lam)
                    cell.populations[species.name] = pop
                    self.tensor[s_idx, y, x] = pop # store the population
                    self.lambda_tensor[s_idx, y, x] = lam  # store the true lambda


    def print_grid(self):
        for row in self.grid.cells:
            line = []
            for cell in row:
                pops = ','.join(f"{k}:{v}" for k, v in cell.populations.items())
                line.append(f"[{pops}]")
            print(' '.join(line))
        print()

    def plot_annotated_grid(self):
        if self.tensor is None:
            print("No data to plot. Run `sample()` first.")
            return

        height, width = self.grid.height, self.grid.width
        num_species = len(self.species)
        side = math.ceil(math.sqrt(num_species))  # subgrid dimension (e.g., 2x2, 3x3, etc.)

        fig, ax = plt.subplots(figsize=(width, height))
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xticks(np.arange(width + 1))
        ax.set_yticks(np.arange(height + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True)

        colors = list(mcolors.TABLEAU_COLORS.values())[:num_species]

        for y in range(height):
            for x in range(width):
                for s_idx, species in enumerate(self.species):
                    val = self.tensor[s_idx, y, x]
                    # Compute subcell coordinates
                    sub_x = s_idx % side
                    sub_y = s_idx // side
                    if sub_y >= side:
                        continue  # avoid overfilling the cell

                    # Coordinates in plot units
                    x0 = x + sub_x / side
                    y0 = y + sub_y / side
                    xc = x0 + 0.5 / side
                    yc = y0 + 0.5 / side

                    # Draw background rect (optional for visual aid)
                    ax.add_patch(plt.Rectangle((x0, y0), 1/side, 1/side,
                                            edgecolor='lightgray', facecolor='white', linewidth=0.5))

                    # Add number
                    ax.text(xc, yc, str(val),
                            ha='center', va='center',
                            color=colors[s_idx], fontsize=9, weight='bold')

        # Legend
        handles = [plt.Line2D([0], [0], marker='s', color='w', label=s.name,
                            markerfacecolor=colors[i], markeredgecolor='gray', markersize=10)
                for i, s in enumerate(self.species)]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    grid = Grid(10, 10)

    grass = Species("Grass", lambda cell: 5 * cell.env_value)
    rabbit = Species("Rabbit", lambda cell: 2 * (cell.env_value - 0.5))
    fox = Species("Fox", lambda cell: (cell.env_value - 0.5)**2)

    ecosystem = Ecosystem(grid, [grass, rabbit, fox])
    ecosystem.sample()
    ecosystem.print_grid()
    ecosystem.plot_annotated_grid()
