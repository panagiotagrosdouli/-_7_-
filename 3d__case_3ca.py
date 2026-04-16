"""
CASim — 3D CA Case Study 3: Periodic Emergence & Pattern Classification
========================================================================
Author  : Panagiota Grosdouli (58523)
Course  : Analysis and Synthesis of Complex Electronic Systems (2024-2025)
Dept.   : Electrical & Computer Engineering, DUTH
Version : 1.0

Description:
    Case Study 3 investigates the emergence of periodic, self-organising
    structures in 3D CA. The simulation:

      1. Runs a configurable 3D CA with threshold-based rules.
      2. Automatically detects oscillators — configurations that repeat
         with a measurable period.
      3. Classifies each run according to Wolfram's four complexity classes
         based on quantitative metrics (entropy, variance, autocorrelation).
      4. Generates a full diagnostic report with heatmaps and charts.

    The system evolves under a birth/death rule based on neighbour count:
      - A cell is active in the next generation if its active-neighbour
        count falls within the interval [τ_low, τ_high].
      - All cells outside this interval die or remain dead.

    This is analogous to the "range rule" formulation of Game of Life:
        B3 / S2,3  →  Conway's Life (2D canonical form)
    but parameterised freely for 3D exploration.

    Boundary: periodic (toroidal).
    Neighbourhood: 26-cell Moore (3×3×3 cube minus centre).
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def initialise_grid(size: int, density: float = 0.3) -> np.ndarray:
    return (np.random.random((size, size, size)) < density).astype(int)


def neighbour_count(grid: np.ndarray) -> np.ndarray:
    """Fast vectorised 26-neighbour count using numpy roll."""
    cnt = np.zeros_like(grid, dtype=int)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                cnt += np.roll(np.roll(np.roll(grid, dx, 0), dy, 1), dz, 2)
    return cnt


def evolve_range_rule(grid: np.ndarray,
                      tau_low: int, tau_high: int) -> np.ndarray:
    """
    Apply the range rule: a cell is alive next step iff its
    active-neighbour count n satisfies  τ_low ≤ n ≤ τ_high.

    This unifies birth and survival in a single interval condition,
    equivalent to outer totalistic rules.

    Args:
        grid     : Current binary grid.
        tau_low  : Lower bound of survival/birth interval.
        tau_high : Upper bound of survival/birth interval.

    Returns:
        New binary grid.
    """
    cnt = neighbour_count(grid)
    return ((cnt >= tau_low) & (cnt <= tau_high)).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# PERIOD DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_period(history: list, window: int = 30) -> int:
    """
    Search for the smallest repeating period in the last `window` generations.

    Compares grid states pairwise using element-wise equality.
    Returns the detected period (1 = fixed point, >1 = oscillator, 0 = none).

    Args:
        history : List of 3D binary numpy arrays.
        window  : Number of recent generations to inspect.

    Returns:
        Detected period as integer (0 if no period found).
    """
    recent = history[-window:] if len(history) >= window else history
    for period in range(1, len(recent) // 2 + 1):
        matches = all(
            np.array_equal(recent[-(p + 1)], recent[-(p + 1 + period)])
            for p in range(period)
        )
        if matches:
            return period
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# COMPLEXITY CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_behaviour(population_series: list, entropy_series: list) -> str:
    """
    Classify observed CA behaviour into Wolfram's four classes.

    Heuristics (based on population and entropy dynamics):
      Class I   — population rapidly collapses to 0 or saturates (low entropy)
      Class II  — low variance in population; periodic entropy oscillations
      Class III — high entropy variance; no detectable period
      Class IV  — intermediate entropy; structured aperiodic patterns

    Args:
        population_series : List of active cell counts per generation.
        entropy_series    : List of Shannon entropy values per generation.

    Returns:
        Classification string.
    """
    pop      = np.array(population_series, dtype=float)
    ent      = np.array(entropy_series, dtype=float)
    pop_var  = np.var(pop[-20:]) if len(pop) >= 20 else np.var(pop)
    ent_mean = np.mean(ent[-20:]) if len(ent) >= 20 else np.mean(ent)
    pop_last = pop[-1] if len(pop) > 0 else 0

    if pop_last == 0:
        return "Class I — Fixed Point (extinction)"
    if pop_var < 5 and ent_mean < 0.4:
        return "Class I — Fixed Point (stable)"
    if pop_var < 50 and ent_mean < 0.8:
        return "Class II — Periodic / Oscillator"
    if ent_mean > 0.85:
        return "Class III — Chaotic / Pseudo-random"
    return "Class IV — Complex / Self-Organising"


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(size: int, generations: int,
                   tau_low: int, tau_high: int) -> dict:
    """
    Execute the range-rule 3D CA and collect diagnostics.

    Returns:
        Dictionary with keys:
          'history'     : list of 3D grids
          'population'  : list of active cell counts
          'entropy'     : list of per-generation Shannon entropy values
          'period'      : detected oscillation period (0 = none)
          'class'       : Wolfram complexity class string
    """
    grid       = initialise_grid(size)
    history    = [grid.copy()]
    population = [int(grid.sum())]
    entropy    = [float(_entropy(grid))]

    for t in range(1, generations):
        grid = evolve_range_rule(grid, tau_low, tau_high)
        history.append(grid.copy())
        population.append(int(grid.sum()))
        entropy.append(float(_entropy(grid)))
        print(f"  Gen {t:>4d} | Pop: {population[-1]:>6d} | H: {entropy[-1]:.3f}", end="\r")
        if population[-1] == 0:
            print(f"\n  Extinct at generation {t}.")
            break

    print()
    period    = detect_period(history)
    behaviour = classify_behaviour(population, entropy)

    return {
        "history":    history,
        "population": population,
        "entropy":    entropy,
        "period":     period,
        "class":      behaviour,
    }


def _entropy(grid: np.ndarray) -> float:
    """Shannon entropy (bits) of a binary grid's state distribution."""
    p1 = grid.mean()
    p0 = 1.0 - p1
    if p1 == 0.0 or p1 == 1.0:
        return 0.0
    return -(p0 * np.log2(p0) + p1 * np.log2(p1))


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def diagnostic_dashboard(results: dict,
                          size: int,
                          tau_low: int, tau_high: int) -> None:
    """
    Render a multi-panel diagnostic dashboard:
      - Panel A : Population over time
      - Panel B : Shannon entropy over time
      - Panel C : XY mid-slice at the final generation
      - Panel D : XZ mid-slice at the final generation
    """
    history    = results["history"]
    population = results["population"]
    entropy_s  = results["entropy"]
    period     = results["period"]
    behaviour  = results["class"]
    final_grid = history[-1]
    mid        = size // 2
    gens       = range(len(history))

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"CASim  3D CA Case Study 3 — Diagnostic Dashboard\n"
        f"Rule: τ ∈ [{tau_low}, {tau_high}]  |  {size}³  |  "
        f"Period: {'none' if period == 0 else period}  |  {behaviour}",
        fontsize=11, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Panel A: Population
    ax_pop = fig.add_subplot(gs[0, 0:2])
    ax_pop.plot(gens, population, color="#3498db", linewidth=2)
    ax_pop.fill_between(gens, population, alpha=0.15, color="#3498db")
    ax_pop.set_title("A — Population over Time", fontweight="bold")
    ax_pop.set_xlabel("Generation"); ax_pop.set_ylabel("Active Cells")
    ax_pop.grid(alpha=0.3)

    # Panel B: Entropy
    ax_ent = fig.add_subplot(gs[1, 0:2])
    ax_ent.plot(gens, entropy_s, color="#e67e22", linewidth=2)
    ax_ent.axhline(1.0, color="grey", linestyle="--", linewidth=1,
                   label="H_max (binary)")
    ax_ent.set_title("B — Shannon Entropy over Time", fontweight="bold")
    ax_ent.set_xlabel("Generation"); ax_ent.set_ylabel("Entropy (bits)")
    ax_ent.set_ylim(0, 1.05)
    ax_ent.legend(fontsize=8); ax_ent.grid(alpha=0.3)

    # Panel C: XY slice
    ax_xy = fig.add_subplot(gs[0, 2])
    ax_xy.imshow(final_grid[:, :, mid], cmap="Blues",
                 vmin=0, vmax=1, interpolation="nearest")
    ax_xy.set_title(f"C — XY slice (z={mid})", fontweight="bold")
    ax_xy.set_xlabel("X"); ax_xy.set_ylabel("Y")

    # Panel D: XZ slice
    ax_xz = fig.add_subplot(gs[1, 2])
    ax_xz.imshow(final_grid[:, mid, :], cmap="Blues",
                 vmin=0, vmax=1, interpolation="nearest")
    ax_xz.set_title(f"D — XZ slice (y={mid})", fontweight="bold")
    ax_xz.set_xlabel("X"); ax_xz.set_ylabel("Z")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def animate_evolution(history: list, interval_ms: int = 120) -> None:
    """Scatter animation of the 3D CA evolution."""
    fig  = plt.figure(figsize=(9, 7))
    ax   = fig.add_subplot(111, projection="3d")
    size = history[0].shape[0]
    ax.set_xlim(0, size); ax.set_ylim(0, size); ax.set_zlim(0, size)
    scat_ref = [None]

    def update(frame):
        if scat_ref[0]:
            scat_ref[0].remove()
        xs, ys, zs = np.where(history[frame] == 1)
        scat_ref[0] = ax.scatter(xs, ys, zs, c="#2980b9",
                                  s=12, alpha=0.6, depthshade=True)
        ax.set_title(f"Case Study 3 — Gen {frame} | Active: {len(xs)}",
                     fontsize=10, fontweight="bold")
        return scat_ref[0],

    ani = animation.FuncAnimation(fig, update, frames=len(history),
                                   interval=interval_ms, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class CaseStudy3App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CASim — 3D CA Case Study 3: Periodic Emergence")
        self.resizable(False, False)
        self._results = None
        self._build_widgets()

    def _build_widgets(self):
        pad = {"padx": 12, "pady": 5}

        header = tk.Frame(self, bg="#003333", pady=10)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        tk.Label(header, text="🧬  CASim  3D — Case Study 3",
                 font=("Helvetica", 14, "bold"),
                 fg="white", bg="#003333").pack()
        tk.Label(header, text="Periodic Emergence & Complexity Classification",
                 font=("Helvetica", 9), fg="#aaffcc", bg="#003333").pack()

        fields = [
            ("Grid Size (N×N×N):",        "size",        "20"),
            ("Number of Generations:",    "generations", "60"),
            ("Threshold Low  (τ_low):",   "tau_low",     "4"),
            ("Threshold High (τ_high):",  "tau_high",    "7"),
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
        btn.grid(row=5, column=0, columnspan=2, pady=12)
        ttk.Button(btn, text="▶  Run",        command=self._run).pack(side="left", padx=5)
        ttk.Button(btn, text="📊 Dashboard",  command=self._show_dashboard).pack(
            side="left", padx=5)
        ttk.Button(btn, text="🎬 Animate",    command=self._animate).pack(
            side="left", padx=5)
        ttk.Button(btn, text="✕  Quit",       command=self.destroy).pack(side="left", padx=5)

        self._status = tk.StringVar(value="Configure and press Run.")
        tk.Label(self, textvariable=self._status, fg="#555",
                 font=("Helvetica", 9), wraplength=360).grid(
            row=6, column=0, columnspan=2, pady=(0, 8))

    def _run(self):
        try:
            size        = int(self._vars["size"].get())
            generations = int(self._vars["generations"].get())
            tau_low     = int(self._vars["tau_low"].get())
            tau_high    = int(self._vars["tau_high"].get())
            assert size >= 4 and generations >= 1
            assert 0 <= tau_low <= tau_high <= 26
        except Exception as exc:
            messagebox.showerror("Input Error", str(exc))
            return

        self._status.set("⏳ Simulating…")
        self.update_idletasks()
        self._params = (size, generations, tau_low, tau_high)
        self._results = run_simulation(size, generations, tau_low, tau_high)

        period    = self._results["period"]
        behaviour = self._results["class"]
        period_str = f"Period = {period}" if period > 0 else "No period detected"
        self._status.set(
            f"✅ Done — τ=[{tau_low},{tau_high}], {size}³, "
            f"{len(self._results['history'])} gen.\n"
            f"{period_str} | {behaviour}"
        )

    def _show_dashboard(self):
        if not self._results:
            messagebox.showinfo("No Data", "Run a simulation first.")
            return
        size, _, tau_low, tau_high = self._params
        diagnostic_dashboard(self._results, size, tau_low, tau_high)

    def _animate(self):
        if not self._results:
            messagebox.showinfo("No Data", "Run a simulation first.")
            return
        animate_evolution(self._results["history"])


if __name__ == "__main__":
    app = CaseStudy3App()
    app.mainloop()
