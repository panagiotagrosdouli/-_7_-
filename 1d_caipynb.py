"""
CASim — 1D Cellular Automaton Simulator
========================================
Author  : Panagiota Grosdouli (58523)
Course  : Analysis and Synthesis of Complex Electronic Systems (2024-2025)
Dept.   : Electrical & Computer Engineering, DUTH
Version : 1.0

Description:
    Implements a fully parametric 1D Cellular Automaton using Wolfram-style
    elementary rules. The user configures all parameters via a tkinter GUI.
    The space-time evolution is rendered as a 2D heatmap with matplotlib.

Wolfram Classes observed:
    Rule  30 → Class III (chaotic / pseudo-random)
    Rule  90 → Class II  (fractal / Sierpiński triangle)
    Rule 110 → Class IV  (complex / Turing-complete)
    Rule 184 → Class II  (particle motion / traffic model)
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def build_rule_table(rule_number: int, num_states: int = 2) -> dict:
    """
    Decode an integer rule number into a neighbourhood → next-state lookup table.

    For binary (k=2) CA this covers Wolfram's 256 elementary rules.
    For k-ary CA the same decimal encoding is generalised.

    Args:
        rule_number : Integer encoding of the transition rule.
        num_states  : Number of distinct cell states (default 2).

    Returns:
        Dictionary mapping 3-tuple neighbourhood patterns to next states.
    """
    rule_table = {}
    patterns = [(i, j, k)
                for i in range(num_states)
                for j in range(num_states)
                for k in range(num_states)]
    # Patterns are ordered from (0,0,0) → (k-1,k-1,k-1)
    for idx, pattern in enumerate(patterns):
        rule_table[pattern] = (rule_number // (num_states ** idx)) % num_states
    return rule_table


def initialise_grid(size: int, num_states: int, seed_mode: str = "single") -> np.ndarray:
    """
    Create the initial generation of the 1D CA.

    Args:
        size       : Number of cells in the grid.
        num_states : Number of distinct cell states.
        seed_mode  : 'single'  → single active cell in the centre
                     'random'  → uniformly random initial states

    Returns:
        1D numpy array of shape (size,) with dtype int.
    """
    grid = np.zeros(size, dtype=int)
    if seed_mode == "single":
        grid[size // 2] = 1
    elif seed_mode == "random":
        grid = np.random.randint(0, num_states, size=size)
    return grid


def evolve_1d(grid: np.ndarray, rule_table: dict) -> np.ndarray:
    """
    Apply one synchronous update step to a 1D CA grid.

    Boundary conditions: periodic (toroidal wrap-around).

    Args:
        grid       : Current generation as a 1D numpy array.
        rule_table : Neighbourhood → next-state lookup dictionary.

    Returns:
        New generation as a 1D numpy array.
    """
    size = len(grid)
    new_grid = np.zeros(size, dtype=int)
    for i in range(size):
        left   = grid[(i - 1) % size]
        centre = grid[i]
        right  = grid[(i + 1) % size]
        new_grid[i] = rule_table[(left, centre, right)]
    return new_grid


def run_simulation(size: int,
                   generations: int,
                   rule_number: int,
                   num_states: int,
                   seed_mode: str) -> np.ndarray:
    """
    Run the full 1D CA simulation and collect generational history.

    Args:
        size        : Grid width (number of cells).
        generations : Number of time steps to simulate.
        rule_number : Wolfram-style rule encoding.
        num_states  : Number of distinct cell states.
        seed_mode   : Initial condition type ('single' or 'random').

    Returns:
        2D numpy array of shape (generations, size) — the full space-time diagram.
    """
    rule_table = build_rule_table(rule_number, num_states)
    history    = np.zeros((generations, size), dtype=int)
    grid       = initialise_grid(size, num_states, seed_mode)
    history[0] = grid

    for t in range(1, generations):
        grid      = evolve_1d(grid, rule_table)
        history[t] = grid

    return history


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def visualise_spacetime(history: np.ndarray,
                        rule_number: int,
                        num_states: int) -> None:
    """
    Render the full space-time evolution diagram as a 2D heatmap.

    Rows    = generations (time, top → bottom)
    Columns = cell positions (space)

    Args:
        history     : 2D array (generations × size) from run_simulation().
        rule_number : Rule used (displayed in title).
        num_states  : State count (used to set colour scale).
    """
    generations, size = history.shape

    fig, ax = plt.subplots(figsize=(min(size / 5, 16), min(generations / 5, 10)))

    cmap = plt.cm.get_cmap("viridis", num_states)
    im   = ax.imshow(history,
                     cmap=cmap,
                     vmin=0,
                     vmax=num_states - 1,
                     interpolation="nearest",
                     aspect="auto")

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label("Cell State", fontsize=10)
    cbar.set_ticks(np.arange(num_states))

    ax.set_title(
        f"1D Cellular Automaton — Rule {rule_number}  "
        f"(k={num_states}, grid={size}, gen={generations})",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("Cell Position", fontsize=11)
    ax.set_ylabel("Generation (↓ time)", fontsize=11)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class CA1DApp(tk.Tk):
    """
    Graphical User Interface for the 1D Cellular Automaton Simulator.

    Provides labelled entry fields for all simulation parameters and
    launches the simulation + visualisation on button press.
    """

    def __init__(self):
        super().__init__()
        self.title("CASim — 1D Cellular Automaton")
        self.resizable(False, False)
        self._build_widgets()

    # ── Widget Construction ──────────────────────────────────────────────────

    def _build_widgets(self):
        pad = {"padx": 12, "pady": 6}

        # ── Header ──────────────────────────────────────────────────────────
        header = tk.Frame(self, bg="#1a1a2e", pady=10)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(header, text="🧬  CASim  1D",
                 font=("Helvetica", 16, "bold"),
                 fg="white", bg="#1a1a2e").pack()
        tk.Label(header, text="Parametric Cellular Automaton Simulator",
                 font=("Helvetica", 9),
                 fg="#aaaacc", bg="#1a1a2e").pack()

        # ── Parameter Fields ─────────────────────────────────────────────────
        fields = [
            ("Grid Size (cells):",       "size",        "101"),
            ("Number of Generations:",   "generations", "80"),
            ("Rule Number (0–255):",      "rule",        "30"),
            ("Number of States (k):",    "states",      "2"),
        ]

        self._vars = {}
        for row, (label_text, key, default) in enumerate(fields, start=1):
            tk.Label(self, text=label_text, anchor="w",
                     font=("Helvetica", 10)).grid(row=row, column=0, sticky="w", **pad)
            var = tk.StringVar(value=default)
            self._vars[key] = var
            ttk.Entry(self, textvariable=var, width=14,
                      font=("Helvetica", 10)).grid(row=row, column=1, sticky="w", **pad)

        # ── Seed Mode ─────────────────────────────────────────────────────────
        tk.Label(self, text="Initial Condition:", anchor="w",
                 font=("Helvetica", 10)).grid(row=5, column=0, sticky="w", **pad)
        self._seed_var = tk.StringVar(value="single")
        seed_frame = tk.Frame(self)
        seed_frame.grid(row=5, column=1, sticky="w", **pad)
        for text, value in [("Single centre cell", "single"), ("Random", "random")]:
            ttk.Radiobutton(seed_frame, text=text,
                            variable=self._seed_var, value=value).pack(anchor="w")

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_frame = tk.Frame(self)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=14)

        ttk.Button(btn_frame, text="▶  Run Simulation",
                   command=self._run).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="✕  Quit",
                   command=self.destroy).pack(side="left", padx=6)

        # ── Info label ────────────────────────────────────────────────────────
        self._status = tk.StringVar(value="Configure parameters and press Run.")
        tk.Label(self, textvariable=self._status,
                 fg="#555555", font=("Helvetica", 9),
                 wraplength=320).grid(row=7, column=0, columnspan=2, pady=(0, 10))

    # ── Event Handlers ───────────────────────────────────────────────────────

    def _run(self):
        """Validate inputs, run simulation, and display results."""
        try:
            size        = int(self._vars["size"].get())
            generations = int(self._vars["generations"].get())
            rule        = int(self._vars["rule"].get())
            states      = int(self._vars["states"].get())
            seed_mode   = self._seed_var.get()

            assert size        >= 3,   "Grid size must be at least 3."
            assert generations >= 1,   "Generations must be at least 1."
            assert states      >= 2,   "Number of states must be at least 2."
            assert 0 <= rule,          "Rule number must be non-negative."

        except (ValueError, AssertionError) as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        self._status.set("⏳ Running simulation …")
        self.update_idletasks()

        history = run_simulation(size, generations, rule, states, seed_mode)

        self._status.set(
            f"✅ Done — Rule {rule}, k={states}, "
            f"{size} cells × {generations} generations."
        )
        visualise_spacetime(history, rule, states)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = CA1DApp()
    app.mainloop()
