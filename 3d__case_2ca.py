"""
CASim — 3D CA Case Study 2: Multi-State Ageing with Entropy Analysis
=====================================================================
Author  : Panagiota Grosdouli (58523)
Course  : Analysis and Synthesis of Complex Electronic Systems (2024-2025)
Dept.   : Electrical & Computer Engineering, DUTH
Version : 1.0

Description:
    Case Study 2 explores k-ary (multi-state) 3D Cellular Automata where
    each live cell progresses through an ageing cycle:

        0 (dead) → 1 (young) → 2 (mature) → … → k-1 (old) → 0 (dead)

    A dead cell becomes alive only when its total neighbour activity
    (sum of neighbour states) falls within a configurable range [τ_min, τ_max].

    This models systems such as:
      - Excitable media (cardiac/neural signal propagation)
      - Ecological succession (species colonisation and senescence)
      - Reaction-diffusion chemical patterns

    Additionally, this module computes per-generation Shannon entropy of
    the state distribution to quantify the complexity of the system's evolution.

    Neighbourhood: 26-cell Moore (3×3×3 cube minus centre).
    Boundary: periodic (toroidal) on all three axes.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from scipy.stats import entropy as scipy_entropy


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def initialise_grid_multistate(size: int, num_states: int,
                                density: float = 0.1) -> np.ndarray:
    """
    Create a sparse initial grid with random states 1…k-1.

    Args:
        size       : Cubic grid edge length.
        num_states : Number of distinct states (k).
        density    : Fraction of initially active cells.

    Returns:
        3D integer numpy array of shape (size, size, size).
    """
    grid = np.zeros((size, size, size), dtype=int)
    mask = np.random.random(grid.shape) < density
    grid[mask] = np.random.randint(1, num_states,
                                    size=int(mask.sum()))
    return grid


def neighbour_activity_sum(grid: np.ndarray) -> np.ndarray:
    """
    Compute the sum of neighbour states (not just count) for every cell.

    This captures not just *how many* neighbours are active but also
    *how active* they are — important for multi-state CA.

    Args:
        grid : Current 3D grid of integer states.

    Returns:
        3D array of neighbour activity sums.
    """
    sums = np.zeros_like(grid, dtype=int)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                sums += np.roll(np.roll(np.roll(grid, dx, axis=0),
                                         dy, axis=1),
                                 dz, axis=2)
    return sums


def evolve_multistate(grid: np.ndarray,
                      num_states: int,
                      tau_min: int,
                      tau_max: int) -> np.ndarray:
    """
    Apply one ageing-rule update step.

    Rules:
      - Dead cells (state 0): born (→ state 1) if τ_min ≤ activity_sum ≤ τ_max.
      - Live cells (state > 0): age by one step; wrap to 0 at state k.

    Args:
        grid       : Current multi-state 3D grid.
        num_states : Number of distinct states k.
        tau_min    : Minimum activity sum for birth.
        tau_max    : Maximum activity sum for birth.

    Returns:
        New grid after one update.
    """
    activity   = neighbour_activity_sum(grid)
    new_grid   = np.zeros_like(grid)

    dead_mask  = grid == 0
    birth_mask = dead_mask & (activity >= tau_min) & (activity <= tau_max)
    age_mask   = grid > 0

    new_grid[birth_mask] = 1
    new_grid[age_mask]   = (grid[age_mask] + 1) % num_states

    return new_grid


def shannon_entropy(grid: np.ndarray, num_states: int) -> float:
    """
    Compute the Shannon entropy of the cell-state distribution.

    H = -Σ p_i log2(p_i)

    Maximum entropy = log2(k) — all states equally probable.
    Entropy = 0 — all cells in the same state (completely ordered).

    Args:
        grid       : Current 3D grid.
        num_states : Number of states (used for bin count).

    Returns:
        Shannon entropy in bits.
    """
    counts = np.bincount(grid.flatten(), minlength=num_states).astype(float)
    probs  = counts / counts.sum()
    probs  = probs[probs > 0]   # avoid log(0)
    return float(scipy_entropy(probs, base=2))


def run_simulation_multistate(size: int, generations: int,
                               num_states: int,
                               tau_min: int, tau_max: int
                               ) -> tuple:
    """
    Execute the multi-state 3D CA simulation.

    Returns:
        Tuple (history, entropy_series) where:
          - history      : list of 3D grids per generation
          - entropy_series : list of floats, one per generation
    """
    grid           = initialise_grid_multistate(size, num_states)
    history        = [grid.copy()]
    entropy_series = [shannon_entropy(grid, num_states)]

    for t in range(1, generations):
        grid = evolve_multistate(grid, num_states, tau_min, tau_max)
        history.append(grid.copy())
        entropy_series.append(shannon_entropy(grid, num_states))
        print(f"  Gen {t:>4d} | H = {entropy_series[-1]:.3f} bits", end="\r")

    print()
    return history, entropy_series


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_entropy(entropy_series: list, num_states: int) -> None:
    """Plot Shannon entropy over generations with maximum-entropy reference."""
    max_h = np.log2(num_states)
    gens  = range(len(entropy_series))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(gens, entropy_series, color="#e74c3c", linewidth=2,
            label="H(t)  — simulation")
    ax.axhline(max_h, color="#2ecc71", linewidth=1.5,
               linestyle="--", label=f"H_max = log₂({num_states}) = {max_h:.2f} bits")
    ax.fill_between(gens, entropy_series, alpha=0.15, color="#e74c3c")

    ax.set_title("Case Study 2 — Shannon Entropy over Generations",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Entropy (bits)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_state_distribution(history: list, num_states: int) -> None:
    """
    Stacked area chart of the fraction of cells in each state per generation.
    """
    fractions = np.zeros((len(history), num_states))
    total     = history[0].size

    for t, grid in enumerate(history):
        counts = np.bincount(grid.flatten(), minlength=num_states)
        fractions[t] = counts / total

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap    = plt.cm.get_cmap("tab10", num_states)
    xs      = np.arange(len(history))
    bottom  = np.zeros(len(history))

    for s in range(num_states):
        label = "Dead (0)" if s == 0 else f"State {s}"
        ax.fill_between(xs, bottom, bottom + fractions[:, s],
                        alpha=0.85, color=cmap(s), label=label)
        bottom += fractions[:, s]

    ax.set_title("Case Study 2 — State Distribution over Generations",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fraction of Cells")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def animate_scatter_multistate(history: list,
                                num_states: int,
                                interval_ms: int = 150) -> None:
    """Animate the 3D evolution with colour-coded cell states."""
    fig  = plt.figure(figsize=(9, 7))
    ax   = fig.add_subplot(111, projection="3d")
    size = history[0].shape[0]
    ax.set_xlim(0, size); ax.set_ylim(0, size); ax.set_zlim(0, size)
    cmap = plt.cm.get_cmap("plasma", num_states)
    scat_ref = [None]

    def update(frame):
        if scat_ref[0]:
            scat_ref[0].remove()
        grid = history[frame]
        xs, ys, zs = np.where(grid > 0)
        states = grid[xs, ys, zs]
        scat_ref[0] = ax.scatter(xs, ys, zs, c=states, cmap=cmap,
                                  vmin=1, vmax=num_states - 1,
                                  s=14, alpha=0.7, depthshade=True)
        active = len(xs)
        ax.set_title(f"Case Study 2 — Gen {frame} | Active: {active}",
                     fontsize=10, fontweight="bold")
        return scat_ref[0],

    ani = animation.FuncAnimation(fig, update, frames=len(history),
                                   interval=interval_ms, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class CaseStudy2App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CASim — 3D CA Case Study 2: Multi-State Ageing")
        self.resizable(False, False)
        self._history = None
        self._entropy = None
        self._build_widgets()

    def _build_widgets(self):
        pad = {"padx": 12, "pady": 5}

        header = tk.Frame(self, bg="#2c003e", pady=10)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(header, text="🧬  CASim  3D — Case Study 2",
                 font=("Helvetica", 14, "bold"),
                 fg="white", bg="#2c003e").pack()
        tk.Label(header, text="Multi-State Ageing with Entropy Analysis",
                 font=("Helvetica", 9), fg="#ddaaff", bg="#2c003e").pack()

        fields = [
            ("Grid Size (N×N×N):",        "size",        "20"),
            ("Number of Generations:",    "generations", "50"),
            ("Number of States (k):",     "states",      "5"),
            ("Min Activity Threshold:",   "tau_min",     "3"),
            ("Max Activity Threshold:",   "tau_max",     "10"),
        ]
        self._vars = {}
        for row, (lbl, key, default) in enumerate(fields, start=1):
            tk.Label(self, text=lbl, anchor="w",
                     font=("Helvetica", 10)).grid(row=row, column=0, sticky="w", **pad)
            var = tk.StringVar(value=default)
            self._vars[key] = var
            ttk.Entry(self, textvariable=var, width=12).grid(
                row=row, column=1, sticky="w", **pad)

        btn = tk.Frame(self)
        btn.grid(row=6, column=0, columnspan=2, pady=12)
        ttk.Button(btn, text="▶  Run",          command=self._run).pack(side="left", padx=5)
        ttk.Button(btn, text="📊 Entropy",       command=self._plot_entropy).pack(
            side="left", padx=5)
        ttk.Button(btn, text="📈 State Dist.",   command=self._plot_dist).pack(
            side="left", padx=5)
        ttk.Button(btn, text="✕  Quit",          command=self.destroy).pack(side="left", padx=5)

        self._status = tk.StringVar(value="Configure parameters and press Run.")
        tk.Label(self, textvariable=self._status, fg="#555",
                 font=("Helvetica", 9), wraplength=340).grid(
            row=7, column=0, columnspan=2, pady=(0, 8))

    def _run(self):
        try:
            size        = int(self._vars["size"].get())
            generations = int(self._vars["generations"].get())
            states      = int(self._vars["states"].get())
            tau_min     = int(self._vars["tau_min"].get())
            tau_max     = int(self._vars["tau_max"].get())
            assert size >= 4 and generations >= 1 and states >= 2
            assert tau_min <= tau_max
        except Exception as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        self._status.set("⏳ Simulating…")
        self.update_idletasks()
        self._history, self._entropy = run_simulation_multistate(
            size, generations, states, tau_min, tau_max)
        self._num_states = states
        self._status.set(
            f"✅ Done — k={states}, τ=[{tau_min},{tau_max}], "
            f"{size}³, {len(self._history)} generations.")
        animate_scatter_multistate(self._history, states)

    def _plot_entropy(self):
        if self._entropy is None:
            messagebox.showinfo("No Data", "Run a simulation first.")
            return
        plot_entropy(self._entropy, self._num_states)

    def _plot_dist(self):
        if self._history is None:
            messagebox.showinfo("No Data", "Run a simulation first.")
            return
        plot_state_distribution(self._history, self._num_states)


if __name__ == "__main__":
    app = CaseStudy2App()
    app.mainloop()
