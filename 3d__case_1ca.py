"""
CASim — 3D CA Case Study 1: Birth / Survival Threshold Rules
=============================================================
Author  : Panagiota Grosdouli (58523)
Course  : Analysis and Synthesis of Complex Electronic Systems (2024-2025)
Dept.   : Electrical & Computer Engineering, DUTH
Version : 1.0

Description:
    Case Study 1 extends the general 3D engine with a richer
    Birth/Survival (B/S) rule set, analogous to the notation used in
    Conway's Game of Life but generalised to 3D.

    Rule notation  B{b1,b2,…} / S{s1,s2,…}:
      - A DEAD cell is Born  (→ state 1) when its active-neighbour count ∈ B.
      - A LIVE cell Survives (→ state 1) when its active-neighbour count ∈ S.
      - All other cells die  (→ state 0).

    Classic 3D analogue of Conway's Life  →  B5,6,7 / S4,5,6,7
    Amoeba rule (stable blobs)            →  B3,5,6,7,8 / S5,6,7,8
    Crystals (dendritic growth)           →  B1,3 / S2

    Neighbourhood: 26-cell Moore (3×3×3 cube minus centre).
    Boundary: periodic (toroidal) on all three axes.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# PREDEFINED RULE PRESETS
# ─────────────────────────────────────────────────────────────────────────────

PRESETS = {
    "Conway 3D  (B5-7 / S4-7)":      {"birth": {5, 6, 7},         "survival": {4, 5, 6, 7}},
    "Amoeba     (B3,5-8 / S5-8)":    {"birth": {3, 5, 6, 7, 8},   "survival": {5, 6, 7, 8}},
    "Crystals   (B1,3 / S2)":        {"birth": {1, 3},             "survival": {2}},
    "Stable Fog (B4,6,8 / S3,6,9)":  {"birth": {4, 6, 8},         "survival": {3, 6, 9}},
    "Custom":                         {"birth": set(),              "survival": set()},
}


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def initialise_grid(size: int, density: float = 0.12) -> np.ndarray:
    """
    Create a sparse binary initial grid.

    Args:
        size    : Edge length of the cubic grid.
        density : Fraction of initially live cells.

    Returns:
        3D boolean numpy array of shape (size, size, size).
    """
    grid = np.zeros((size, size, size), dtype=int)
    mask = np.random.random(grid.shape) < density
    grid[mask] = 1
    return grid


def count_neighbours_vectorised(grid: np.ndarray) -> np.ndarray:
    """
    Compute the active-neighbour count for every cell simultaneously
    using numpy roll operations — avoids explicit Python loops over cells.

    Args:
        grid : 3D binary numpy array.

    Returns:
        3D numpy array of neighbour counts, same shape as grid.
    """
    counts = np.zeros_like(grid, dtype=int)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                counts += np.roll(np.roll(np.roll(grid, dx, axis=0),
                                          dy, axis=1),
                                  dz, axis=2)
    return counts


def evolve_bs(grid: np.ndarray,
              birth_set: set,
              survival_set: set) -> np.ndarray:
    """
    Apply one B/S update step to the 3D binary grid.

    Args:
        grid         : Current binary grid (0 = dead, 1 = live).
        birth_set    : Neighbour counts that cause dead cells to be born.
        survival_set : Neighbour counts that allow live cells to survive.

    Returns:
        New binary grid after the update.
    """
    counts    = count_neighbours_vectorised(grid)
    new_grid  = np.zeros_like(grid)

    # Vectorised birth
    birth_mask = np.isin(counts, list(birth_set)) & (grid == 0)
    # Vectorised survival
    surv_mask  = np.isin(counts, list(survival_set)) & (grid == 1)

    new_grid[birth_mask] = 1
    new_grid[surv_mask]  = 1
    return new_grid


def run_simulation(size: int, generations: int,
                   birth_set: set, survival_set: set) -> list:
    """
    Execute the full B/S 3D CA simulation.

    Args:
        size        : Edge length of the cubic grid.
        generations : Number of time steps.
        birth_set   : Birth condition.
        survival_set: Survival condition.

    Returns:
        List of 3D binary numpy arrays, one per generation.
    """
    grid    = initialise_grid(size)
    history = [grid.copy()]

    for t in range(1, generations):
        grid = evolve_bs(grid, birth_set, survival_set)
        history.append(grid.copy())
        active = int(grid.sum())
        print(f"  Gen {t:>4d} | Live cells: {active:>6d}", end="\r")
        if active == 0:
            print(f"\n  Population extinct at generation {t}.")
            break

    print()
    return history


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_population_curve(history: list) -> None:
    """Plot the total live-cell count across all generations."""
    counts = [int(g.sum()) for g in history]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(counts, color="#5e81f4", linewidth=2)
    ax.fill_between(range(len(counts)), counts, alpha=0.2, color="#5e81f4")
    ax.set_title("3D CA Case Study 1 — Population Over Time",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Live Cell Count")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def animate_scatter(history: list, interval_ms: int = 150) -> None:
    """Animate the 3D evolution as a scatter plot of live cells."""
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    size = history[0].shape[0]
    ax.set_xlim(0, size); ax.set_ylim(0, size); ax.set_zlim(0, size)

    scat_ref = [None]

    def update(frame):
        if scat_ref[0]:
            scat_ref[0].remove()
        xs, ys, zs = np.where(history[frame] == 1)
        scat_ref[0] = ax.scatter(xs, ys, zs, c="#5e81f4",
                                 s=14, alpha=0.65, depthshade=True)
        ax.set_title(f"Case Study 1 — Gen {frame} | Live: {len(xs)}",
                     fontsize=10, fontweight="bold")
        return scat_ref[0],

    ani = animation.FuncAnimation(fig, update,
                                  frames=len(history),
                                  interval=interval_ms,
                                  blit=False, repeat=False)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class CaseStudy1App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CASim — 3D CA Case Study 1: B/S Rules")
        self.resizable(False, False)
        self._build_widgets()

    def _build_widgets(self):
        pad = {"padx": 12, "pady": 5}

        header = tk.Frame(self, bg="#0f3460", pady=10)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(header, text="🧬  CASim  3D — Case Study 1",
                 font=("Helvetica", 14, "bold"),
                 fg="white", bg="#0f3460").pack()
        tk.Label(header, text="Birth / Survival Rule System",
                 font=("Helvetica", 9), fg="#aaddff", bg="#0f3460").pack()

        # Numeric parameters
        fields = [
            ("Grid Size (N×N×N):",     "size",        "25"),
            ("Number of Generations:", "generations", "40"),
        ]
        self._vars = {}
        for row, (lbl, key, default) in enumerate(fields, start=1):
            tk.Label(self, text=lbl, anchor="w",
                     font=("Helvetica", 10)).grid(row=row, column=0, sticky="w", **pad)
            var = tk.StringVar(value=default)
            self._vars[key] = var
            ttk.Entry(self, textvariable=var, width=12).grid(
                row=row, column=1, sticky="w", **pad)

        # Preset selector
        tk.Label(self, text="Rule Preset:", anchor="w",
                 font=("Helvetica", 10)).grid(row=3, column=0, sticky="w", **pad)
        self._preset_var = tk.StringVar(value=list(PRESETS.keys())[0])
        ttk.Combobox(self, textvariable=self._preset_var,
                     values=list(PRESETS.keys()), state="readonly",
                     width=26).grid(row=3, column=1, sticky="w", **pad)

        # Custom rule fields
        tk.Label(self, text="Birth counts (e.g. 5,6,7):", anchor="w",
                 font=("Helvetica", 9), fg="#666").grid(
            row=4, column=0, sticky="w", **pad)
        self._birth_var = tk.StringVar(value="5,6,7")
        ttk.Entry(self, textvariable=self._birth_var, width=20).grid(
            row=4, column=1, sticky="w", **pad)

        tk.Label(self, text="Survival counts (e.g. 4,5,6,7):", anchor="w",
                 font=("Helvetica", 9), fg="#666").grid(
            row=5, column=0, sticky="w", **pad)
        self._surv_var = tk.StringVar(value="4,5,6,7")
        ttk.Entry(self, textvariable=self._surv_var, width=20).grid(
            row=5, column=1, sticky="w", **pad)

        # Buttons
        btn = tk.Frame(self)
        btn.grid(row=6, column=0, columnspan=2, pady=12)
        ttk.Button(btn, text="▶  Run", command=self._run).pack(side="left", padx=5)
        ttk.Button(btn, text="📈 Population", command=self._plot_pop).pack(
            side="left", padx=5)
        ttk.Button(btn, text="✕  Quit", command=self.destroy).pack(side="left", padx=5)

        self._status = tk.StringVar(value="Select a preset or enter custom rules.")
        tk.Label(self, textvariable=self._status, fg="#555",
                 font=("Helvetica", 9), wraplength=340).grid(
            row=7, column=0, columnspan=2, pady=(0, 8))

        self._history = None

    def _parse_set(self, raw: str) -> set:
        return {int(x.strip()) for x in raw.split(",") if x.strip()}

    def _run(self):
        try:
            size        = int(self._vars["size"].get())
            generations = int(self._vars["generations"].get())
            preset_name = self._preset_var.get()

            if preset_name == "Custom":
                birth_set = self._parse_set(self._birth_var.get())
                surv_set  = self._parse_set(self._surv_var.get())
            else:
                birth_set = PRESETS[preset_name]["birth"]
                surv_set  = PRESETS[preset_name]["survival"]

            assert size >= 4 and generations >= 1
        except Exception as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        self._status.set("⏳ Simulating…")
        self.update_idletasks()
        self._history = run_simulation(size, generations, birth_set, surv_set)
        self._status.set(
            f"✅ Done — B{sorted(birth_set)}/S{sorted(surv_set)}, "
            f"{size}³, {len(self._history)} generations."
        )
        animate_scatter(self._history)

    def _plot_pop(self):
        if self._history is None:
            messagebox.showinfo("No Data", "Run a simulation first.")
            return
        plot_population_curve(self._history)


if __name__ == "__main__":
    app = CaseStudy1App()
    app.mainloop()
