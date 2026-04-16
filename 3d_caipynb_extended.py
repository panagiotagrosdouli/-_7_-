"""
CASim — 3D Cellular Automaton (Extended Variant)
=================================================
Author  : Panagiota Grosdouli (58523)
Course  : Analysis and Synthesis of Complex Electronic Systems (2024-2025)
Dept.   : Electrical & Computer Engineering, DUTH
Version : 1.0

Description:
    Extended 3D CA module that adds two capabilities beyond the general engine:

    1. CONFIGURABLE NEIGHBOURHOOD RADIUS (r):
       Instead of the fixed 26-cell Moore neighbourhood (r=1), users can
       select any integer radius r ≥ 1. The neighbourhood then consists of
       all cells within the L-inf (Chebyshev) ball of radius r:
         N_r(x,y,z) = { (x+dx, y+dy, z+dz) : |dx|<=r, |dy|<=r, |dz|<=r } minus {cell}
       Total neighbourhood size = (2r+1)³ − 1.

    2. AUTOMATED RULE SWEEP:
       Systematically iterates over a grid of threshold values and records
       the final active-cell fraction and Shannon entropy for each, producing
       a phase-diagram heatmap. This enables identification of phase
       transitions and edge-of-chaos regions in the rule space.

    Neighbourhood: generalised Moore (Chebyshev ball of radius r).
    Boundary: periodic (toroidal) on all three axes.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from itertools import product as iproduct


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE — GENERALISED RADIUS
# ─────────────────────────────────────────────────────────────────────────────

def neighbourhood_offsets(radius: int) -> np.ndarray:
    """
    Pre-compute the set of (dx, dy, dz) offsets for a Moore neighbourhood
    of the given radius, excluding (0, 0, 0).

    Args:
        radius : Chebyshev radius r ≥ 1.

    Returns:
        Array of shape (N, 3) where N = (2r+1)³ − 1.
    """
    offsets = [
        (dx, dy, dz)
        for dx, dy, dz in iproduct(range(-radius, radius + 1), repeat=3)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    return np.array(offsets, dtype=int)


def count_neighbours_radius(grid: np.ndarray,
                              offsets: np.ndarray) -> np.ndarray:
    """
    Vectorised neighbour count for an arbitrary neighbourhood defined by
    pre-computed offsets.

    Args:
        grid    : Current binary 3D grid.
        offsets : Array of (dx, dy, dz) offsets from neighbourhood_offsets().

    Returns:
        3D integer array of neighbour counts.
    """
    counts = np.zeros_like(grid, dtype=int)
    for dx, dy, dz in offsets:
        counts += np.roll(np.roll(np.roll(grid, int(dx), 0),
                                   int(dy), 1),
                           int(dz), 2)
    return counts


def evolve_extended(grid: np.ndarray,
                    offsets: np.ndarray,
                    threshold: int,
                    max_neighbours: int) -> np.ndarray:
    """
    Threshold-based binary evolution with generalised neighbourhood.

    A cell is active in the next generation if its neighbour count
    is within [threshold, max_neighbours - threshold] — a symmetric
    survival window around the midpoint.

    Args:
        grid          : Current binary 3D grid.
        offsets       : Pre-computed neighbourhood offsets.
        threshold     : Minimum / (max_n − maximum) boundary of the window.
        max_neighbours: Total neighbourhood size (number of offsets).

    Returns:
        New binary 3D grid.
    """
    counts = count_neighbours_radius(grid, offsets)
    lo     = threshold
    hi     = max_neighbours - threshold
    if lo > hi:
        lo, hi = hi, lo
    return ((counts >= lo) & (counts <= hi)).astype(int)


def run_simulation_extended(size: int, generations: int,
                              radius: int, threshold: int) -> dict:
    """
    Execute the extended 3D CA simulation.

    Args:
        size        : Cubic grid edge length.
        generations : Time steps.
        radius      : Neighbourhood radius.
        threshold   : Symmetric survival window boundary.

    Returns:
        Dictionary with 'history', 'population', 'entropy'.
    """
    offsets     = neighbourhood_offsets(radius)
    max_n       = len(offsets)
    grid        = (np.random.random((size, size, size)) < 0.25).astype(int)
    history     = [grid.copy()]
    population  = [int(grid.sum())]
    entropy     = [_entropy(grid)]

    for t in range(1, generations):
        grid = evolve_extended(grid, offsets, threshold, max_n)
        history.append(grid.copy())
        population.append(int(grid.sum()))
        entropy.append(_entropy(grid))
        print(f"  Gen {t:>4d} | r={radius} | τ={threshold} | "
              f"Pop: {population[-1]:>6d}", end="\r")

    print()
    return {"history": history, "population": population, "entropy": entropy}


def _entropy(grid: np.ndarray) -> float:
    p = grid.mean()
    if p == 0.0 or p == 1.0:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


# ─────────────────────────────────────────────────────────────────────────────
# PHASE DIAGRAM / RULE SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def rule_sweep(size: int = 14,
               generations: int = 25,
               radius: int = 1) -> tuple:
    """
    Sweep the threshold parameter and measure steady-state statistics.

    For each threshold value τ in 0…max_n//2:
      - Run a short simulation.
      - Record the mean active-cell fraction and entropy over the last 5 generations.

    Args:
        size        : Grid edge length (kept small for speed).
        generations : Steps per configuration.
        radius      : Neighbourhood radius.

    Returns:
        Tuple (thresholds, fractions, entropies).
    """
    offsets   = neighbourhood_offsets(radius)
    max_n     = len(offsets)
    thresholds = list(range(0, max_n // 2 + 1))
    fractions  = []
    entropies  = []

    total = size ** 3
    print(f"\n  Rule sweep: r={radius}, max_n={max_n}, {len(thresholds)} configs …")

    for tau in thresholds:
        grid = (np.random.random((size, size, size)) < 0.3).astype(int)
        frac_buf = []
        ent_buf  = []
        for _ in range(generations):
            grid = evolve_extended(grid, offsets, tau, max_n)
            frac_buf.append(grid.sum() / total)
            ent_buf.append(_entropy(grid))
        fractions.append(np.mean(frac_buf[-5:]))
        entropies.append(np.mean(ent_buf[-5:]))
        print(f"    τ={tau:>3d} | frac={fractions[-1]:.3f} | "
              f"H={entropies[-1]:.3f}", end="\r")

    print()
    return thresholds, fractions, entropies


def plot_phase_diagram(thresholds: list,
                        fractions: list,
                        entropies: list,
                        radius: int) -> None:
    """Plot phase diagram: active fraction and entropy vs threshold."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(thresholds, fractions, "o-", color="#2980b9",
             linewidth=2, markersize=5)
    ax1.set_ylabel("Mean Active Fraction")
    ax1.set_title(f"Phase Diagram — Extended 3D CA (r={radius})",
                  fontsize=12, fontweight="bold")
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2.plot(thresholds, entropies, "s-", color="#e74c3c",
             linewidth=2, markersize=5)
    ax2.set_ylabel("Mean Shannon Entropy (bits)")
    ax2.set_xlabel("Threshold τ")
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.05)

    # Mark the edge-of-chaos region (near-peak entropy)
    peak_idx = int(np.argmax(entropies))
    ax2.axvline(thresholds[peak_idx], color="green", linestyle="--",
                linewidth=1.5, label=f"Peak entropy at τ={thresholds[peak_idx]}")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def animate_scatter(history: list, radius: int,
                    threshold: int, interval_ms: int = 150) -> None:
    """Scatter animation with neighbourhood-radius label."""
    fig  = plt.figure(figsize=(9, 7))
    ax   = fig.add_subplot(111, projection="3d")
    size = history[0].shape[0]
    ax.set_xlim(0, size); ax.set_ylim(0, size); ax.set_zlim(0, size)
    scat_ref = [None]

    def update(frame):
        if scat_ref[0]:
            scat_ref[0].remove()
        xs, ys, zs = np.where(history[frame] == 1)
        scat_ref[0] = ax.scatter(xs, ys, zs, c="#1abc9c",
                                  s=12, alpha=0.65, depthshade=True)
        ax.set_title(f"Extended 3D CA  r={radius}  τ={threshold}  "
                     f"Gen {frame}  Active: {len(xs)}",
                     fontsize=9, fontweight="bold")
        return scat_ref[0],

    ani = animation.FuncAnimation(fig, update, frames=len(history),
                                   interval=interval_ms, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()


def plot_population_and_entropy(results: dict,
                                  radius: int, threshold: int) -> None:
    """Side-by-side population and entropy curves."""
    pop = results["population"]
    ent = results["entropy"]
    gens = range(len(pop))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(gens, pop, color="#1abc9c", linewidth=2)
    ax1.fill_between(gens, pop, alpha=0.15, color="#1abc9c")
    ax1.set_title(f"Population  (r={radius}, τ={threshold})", fontweight="bold")
    ax1.set_xlabel("Generation"); ax1.set_ylabel("Active Cells")
    ax1.grid(alpha=0.3)

    ax2.plot(gens, ent, color="#9b59b6", linewidth=2)
    ax2.set_title(f"Shannon Entropy  (r={radius}, τ={threshold})",
                  fontweight="bold")
    ax2.set_xlabel("Generation"); ax2.set_ylabel("Entropy (bits)")
    ax2.set_ylim(0, 1.05); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class ExtendedCA3DApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CASim — Extended 3D CA (Radius + Rule Sweep)")
        self.resizable(False, False)
        self._results = None
        self._build_widgets()

    def _build_widgets(self):
        pad = {"padx": 12, "pady": 5}

        header = tk.Frame(self, bg="#1a3a1a", pady=10)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(header, text="🧬  CASim  3D  Extended",
                 font=("Helvetica", 14, "bold"),
                 fg="white", bg="#1a3a1a").pack()
        tk.Label(header, text="Configurable Radius  ·  Rule Space Sweep",
                 font=("Helvetica", 9), fg="#aaffaa", bg="#1a3a1a").pack()

        fields = [
            ("Grid Size (N×N×N):",        "size",        "18"),
            ("Number of Generations:",    "generations", "40"),
            ("Neighbourhood Radius (r):", "radius",      "1"),
            ("Threshold (τ):",            "threshold",   "5"),
        ]
        self._vars = {}
        for row, (lbl, key, default) in enumerate(fields, start=1):
            tk.Label(self, text=lbl, anchor="w",
                     font=("Helvetica", 10)).grid(row=row, column=0, sticky="w", **pad)
            var = tk.StringVar(value=default)
            self._vars[key] = var
            ttk.Entry(self, textvariable=var, width=12).grid(
                row=row, column=1, sticky="w", **pad)

        # Info label showing max neighbours
        self._info_var = tk.StringVar(value="r=1 → 26 neighbours")
        tk.Label(self, textvariable=self._info_var,
                 fg="#888", font=("Helvetica", 8, "italic")).grid(
            row=5, column=0, columnspan=2, pady=2)
        self._vars["radius"].trace_add("write", self._update_info)

        btn = tk.Frame(self)
        btn.grid(row=6, column=0, columnspan=2, pady=12)
        ttk.Button(btn, text="▶  Run",        command=self._run).pack(side="left", padx=5)
        ttk.Button(btn, text="📊 Stats",       command=self._stats).pack(side="left", padx=5)
        ttk.Button(btn, text="🔬 Rule Sweep",  command=self._sweep).pack(side="left", padx=5)
        ttk.Button(btn, text="✕  Quit",        command=self.destroy).pack(side="left", padx=5)

        self._status = tk.StringVar(value="Configure and press Run.")
        tk.Label(self, textvariable=self._status, fg="#555",
                 font=("Helvetica", 9), wraplength=360).grid(
            row=7, column=0, columnspan=2, pady=(0, 8))

    def _update_info(self, *_):
        try:
            r = int(self._vars["radius"].get())
            n = (2 * r + 1) ** 3 - 1
            self._info_var.set(f"r={r} → {n} neighbours in Moore cube")
        except ValueError:
            pass

    def _run(self):
        try:
            size        = int(self._vars["size"].get())
            generations = int(self._vars["generations"].get())
            radius      = int(self._vars["radius"].get())
            threshold   = int(self._vars["threshold"].get())
            assert size >= 4 and generations >= 1
            assert 1 <= radius <= 4, "Radius must be 1–4 for reasonable speed."
        except Exception as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        self._status.set("⏳ Simulating…")
        self.update_idletasks()
        self._results  = run_simulation_extended(size, generations, radius, threshold)
        self._radius    = radius
        self._threshold = threshold
        self._status.set(
            f"✅ Done — r={radius}, τ={threshold}, {size}³, "
            f"{len(self._results['history'])} generations."
        )
        animate_scatter(self._results["history"], radius, threshold)

    def _stats(self):
        if not self._results:
            messagebox.showinfo("No Data", "Run a simulation first.")
            return
        plot_population_and_entropy(self._results, self._radius, self._threshold)

    def _sweep(self):
        try:
            size   = min(int(self._vars["size"].get()), 15)
            radius = int(self._vars["radius"].get())
            assert 1 <= radius <= 2, "Keep r ≤ 2 for the sweep (speed)."
        except Exception as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        self._status.set("⏳ Running rule sweep (may take a minute)…")
        self.update_idletasks()
        thresholds, fractions, entropies = rule_sweep(size=size,
                                                       generations=20,
                                                       radius=radius)
        self._status.set("✅ Rule sweep complete.")
        plot_phase_diagram(thresholds, fractions, entropies, radius)


if __name__ == "__main__":
    app = ExtendedCA3DApp()
    app.mainloop()
