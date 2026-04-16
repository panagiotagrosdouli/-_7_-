"""
CASim — 3D Cellular Automaton Simulator (General)
===================================================
Author  : Panagiota Grosdouli (58523)
Course  : Analysis and Synthesis of Complex Electronic Systems (2024-2025)
Dept.   : Electrical & Computer Engineering, DUTH
Version : 1.0

Description:
    General-purpose 3D Cellular Automaton engine with configurable:
      - Grid dimensions (X × Y × Z)
      - Number of cell states (k ≥ 2)
      - Activation threshold (τ)
      - Number of generations
    Neighbourhood: 26-cell Moore neighbourhood (full 3×3×3 cube minus centre).
    Boundary conditions: periodic (toroidal) on all three axes.

    Visualisation: 2D slice animation through the Z-axis, one frame per
    generation, rendered with matplotlib's 3D scatter and/or imshow.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 — registers 3D projection
import matplotlib.animation as animation


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def initialise_grid_3d(size: int, num_states: int,
                        density: float = 0.15) -> np.ndarray:
    """
    Create a sparse random initial 3D grid.

    Args:
        size       : Edge length of the cubic grid.
        num_states : Number of distinct cell states.
        density    : Fraction of cells initialised to a non-zero state.

    Returns:
        3D numpy array of shape (size, size, size) with dtype int.
    """
    grid = np.zeros((size, size, size), dtype=int)
    num_active = int(density * size ** 3)
    indices = np.random.choice(size ** 3, size=num_active, replace=False)
    xs, ys, zs = np.unravel_index(indices, (size, size, size))
    for x, y, z in zip(xs, ys, zs):
        grid[x, y, z] = np.random.randint(1, num_states)
    return grid


def count_active_neighbours_3d(grid: np.ndarray, x: int, y: int, z: int) -> int:
    """
    Count non-zero cells in the 26-cell Moore neighbourhood of (x, y, z).

    Uses periodic (toroidal) boundary conditions on all axes.

    Args:
        grid : 3D numpy array representing the current generation.
        x, y, z : Coordinates of the target cell.

    Returns:
        Integer count of active (non-zero) neighbours.
    """
    size = grid.shape[0]
    count = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                nx = (x + dx) % size
                ny = (y + dy) % size
                nz = (z + dz) % size
                if grid[nx, ny, nz] > 0:
                    count += 1
    return count


def evolve_3d(grid: np.ndarray, threshold: int, num_states: int) -> np.ndarray:
    """
    Apply one synchronous update step to the 3D CA grid.

    Transition rule (threshold-based):
      - A dead cell (state 0) becomes active (state 1) if its active
        neighbour count equals exactly `threshold`.
      - An active cell ages: state → state + 1.
      - A cell reaching state `num_states` returns to state 0 (dead).

    Args:
        grid       : Current generation (3D numpy array).
        threshold  : Exact active-neighbour count required for birth.
        num_states : Total number of distinct states.

    Returns:
        New generation as a 3D numpy array.
    """
    size     = grid.shape[0]
    new_grid = np.zeros_like(grid)

    for x in range(size):
        for y in range(size):
            for z in range(size):
                state     = grid[x, y, z]
                neighbours = count_active_neighbours_3d(grid, x, y, z)

                if state == 0:
                    # Birth rule: dead cell activates when neighbours == threshold
                    if neighbours == threshold:
                        new_grid[x, y, z] = 1
                else:
                    # Ageing rule: active cell advances state, wraps to 0
                    new_grid[x, y, z] = (state + 1) % num_states

    return new_grid


def run_simulation_3d(size: int, generations: int,
                      threshold: int, num_states: int) -> list:
    """
    Execute the full 3D CA simulation.

    Args:
        size        : Edge length of the cubic grid.
        generations : Number of time steps.
        threshold   : Birth threshold.
        num_states  : Number of distinct states.

    Returns:
        List of 3D numpy arrays — one per generation.
    """
    grid    = initialise_grid_3d(size, num_states)
    history = [grid.copy()]

    for t in range(1, generations):
        grid = evolve_3d(grid, threshold, num_states)
        history.append(grid.copy())
        active = np.sum(grid > 0)
        print(f"  Generation {t:>4d} | Active cells: {active:>6d}", end="\r")

    print()  # newline after progress output
    return history


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def visualise_3d_scatter(history: list, interval_ms: int = 200) -> None:
    """
    Animate the 3D CA evolution as a scatter plot of active cells.

    Args:
        history     : List of 3D grids from run_simulation_3d().
        interval_ms : Milliseconds between animation frames.
    """
    fig  = plt.figure(figsize=(9, 7))
    ax   = fig.add_subplot(111, projection="3d")
    size = history[0].shape[0]

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_zlim(0, size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    scatter_ref = [None]

    def update(frame):
        if scatter_ref[0] is not None:
            scatter_ref[0].remove()

        grid = history[frame]
        xs, ys, zs = np.where(grid > 0)
        states      = grid[xs, ys, zs]

        scatter_ref[0] = ax.scatter(
            xs, ys, zs,
            c=states, cmap="viridis",
            vmin=1, vmax=grid.max() or 1,
            s=18, alpha=0.7, depthshade=True
        )
        ax.set_title(
            f"3D CA  |  Generation {frame}  |  Active: {len(xs)}",
            fontsize=11, fontweight="bold"
        )
        return scatter_ref[0],

    ani = animation.FuncAnimation(
        fig, update,
        frames=len(history),
        interval=interval_ms,
        blit=False,
        repeat=False
    )

    plt.tight_layout()
    plt.show()


def visualise_3d_slices(history: list, generation: int = -1) -> None:
    """
    Display three orthogonal cross-section slices (XY, XZ, YZ) of a
    selected generation as 2D heatmaps.

    Args:
        history    : List of 3D grids.
        generation : Index of the generation to inspect (-1 = last).
    """
    grid = history[generation]
    size = grid.shape[0]
    mid  = size // 2
    gen_label = len(history) - 1 if generation == -1 else generation

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    slices = [
        (grid[:, :, mid], f"XY plane  (z={mid})",  "X", "Y"),
        (grid[:, mid, :], f"XZ plane  (y={mid})",  "X", "Z"),
        (grid[mid, :, :], f"YZ plane  (x={mid})",  "Y", "Z"),
    ]

    for ax, (data, title, xlabel, ylabel) in zip(axes, slices):
        im = ax.imshow(data, cmap="viridis", interpolation="nearest",
                       vmin=0, vmax=max(grid.max(), 1))
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04).set_label("State")

    fig.suptitle(
        f"3D CA Cross-Section Slices  |  Generation {gen_label}",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class CA3DApp(tk.Tk):
    """
    GUI for the general 3D Cellular Automaton Simulator.
    """

    def __init__(self):
        super().__init__()
        self.title("CASim — 3D Cellular Automaton")
        self.resizable(False, False)
        self._history = None
        self._build_widgets()

    def _build_widgets(self):
        pad = {"padx": 12, "pady": 6}

        # Header
        header = tk.Frame(self, bg="#16213e", pady=10)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(header, text="🧬  CASim  3D",
                 font=("Helvetica", 16, "bold"),
                 fg="white", bg="#16213e").pack()
        tk.Label(header, text="Parametric 3D Cellular Automaton Simulator",
                 font=("Helvetica", 9),
                 fg="#aaaacc", bg="#16213e").pack()

        fields = [
            ("Grid Size (N×N×N):",        "size",        "20"),
            ("Number of Generations:",    "generations", "30"),
            ("Activation Threshold (τ):", "threshold",   "4"),
            ("Number of States (k):",     "states",      "3"),
        ]

        self._vars = {}
        for row, (label_text, key, default) in enumerate(fields, start=1):
            tk.Label(self, text=label_text, anchor="w",
                     font=("Helvetica", 10)).grid(row=row, column=0, sticky="w", **pad)
            var = tk.StringVar(value=default)
            self._vars[key] = var
            ttk.Entry(self, textvariable=var, width=12,
                      font=("Helvetica", 10)).grid(row=row, column=1, sticky="w", **pad)

        # Visualisation mode
        tk.Label(self, text="Visualisation:", anchor="w",
                 font=("Helvetica", 10)).grid(row=5, column=0, sticky="w", **pad)
        self._viz_var = tk.StringVar(value="scatter")
        viz_frame = tk.Frame(self)
        viz_frame.grid(row=5, column=1, sticky="w", **pad)
        for text, value in [("3D Scatter Animation", "scatter"),
                             ("Cross-Section Slices", "slices")]:
            ttk.Radiobutton(viz_frame, text=text,
                            variable=self._viz_var, value=value).pack(anchor="w")

        # Buttons
        btn_frame = tk.Frame(self)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=14)
        ttk.Button(btn_frame, text="▶  Run Simulation",
                   command=self._run).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="✕  Quit",
                   command=self.destroy).pack(side="left", padx=6)

        self._status = tk.StringVar(value="Configure parameters and press Run.")
        tk.Label(self, textvariable=self._status, fg="#555555",
                 font=("Helvetica", 9), wraplength=320).grid(
            row=7, column=0, columnspan=2, pady=(0, 10))

    def _run(self):
        try:
            size        = int(self._vars["size"].get())
            generations = int(self._vars["generations"].get())
            threshold   = int(self._vars["threshold"].get())
            states      = int(self._vars["states"].get())
            viz_mode    = self._viz_var.get()

            assert size        >= 4,  "Grid size must be at least 4."
            assert generations >= 1,  "Generations must be at least 1."
            assert states      >= 2,  "Number of states must be at least 2."
            assert 1 <= threshold <= 26, "Threshold must be between 1 and 26."

        except (ValueError, AssertionError) as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        self._status.set("⏳ Running simulation (this may take a moment)…")
        self.update_idletasks()

        self._history = run_simulation_3d(size, generations, threshold, states)

        self._status.set(
            f"✅ Done — {size}³ grid, τ={threshold}, k={states}, "
            f"{generations} generations."
        )

        if viz_mode == "scatter":
            visualise_3d_scatter(self._history)
        else:
            visualise_3d_slices(self._history)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = CA3DApp()
    app.mainloop()
