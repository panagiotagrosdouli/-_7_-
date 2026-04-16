<div align="center">

<h1>🧬 CASim</h1>
<h3>A Parametric Framework for Multi-Dimensional Cellular Automata Simulation</h3>

<p>
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Dimensions-1D%20%7C%202D%20%7C%203D-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Visualization-matplotlib-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/GUI-tkinter-green?style=flat-square" />
  <img src="https://img.shields.io/badge/License-Academic-lightgrey?style=flat-square" />
</p>

<p>
  <em>
    A fully parametric, interactive simulation engine for exploring emergent complexity<br>
    in one-, two-, and three-dimensional cellular automata.
  </em>
</p>

</div>

---

## 📖 Overview

**CASim** is a research-oriented software framework for the design, execution, and visual analysis of **Cellular Automata (CA)** across multiple spatial dimensions. The system supports user-defined parameters — including state count, grid size, evolution rules, thresholds, and neighbourhood topology — allowing systematic experimentation with complex self-organising systems.

The project was developed as part of the course **"Analysis and Synthesis of Complex Electronic Systems"** (2024–2025) at the **Department of Electrical and Computer Engineering, Democritus University of Thrace (DUTH)**.

> *"Simple local rules. Global emergent complexity."*
> — The fundamental principle of Cellular Automata theory.

---

## 🔬 Research Motivation

Cellular Automata have been a cornerstone of computational science since the pioneering work of von Neumann and Wolfram, demonstrating that discrete, deterministic systems with minimal rule sets can generate extraordinarily rich and unpredictable behaviour. From modelling biological pattern formation to simulating fluid dynamics and cryptographic applications, CA remain a powerful tool across disciplines.

CASim addresses the need for a **generalised, accessible simulation platform** that allows researchers and students to:

- Rapidly prototype CA configurations without hardware or language barriers
- Visually explore the phase space of rule sets and dimensional configurations
- Compare 1D Wolfram-style elementary automata with higher-dimensional threshold systems
- Build intuition for **emergence, periodicity, and chaotic dynamics** in discrete systems

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Multi-dimensional support** | Simulate CA in 1D, 2D, and 3D on a unified parametric engine |
| **Configurable state spaces** | Define any number of discrete cell states (binary, ternary, k-ary) |
| **Rule flexibility** | Wolfram-style decimal rule codes (1D) or threshold-based activation (2D/3D) |
| **Neighbourhood topology** | Choose Moore or Von Neumann neighbourhood (2D); configurable radius in 3D |
| **Periodic boundary conditions** | Toroidal (wrap-around) grid topology for finite-size artefact elimination |
| **Interactive GUI** | Parameter input via `tkinter` — no command-line expertise required |
| **Real-time visualisation** | Generation-by-generation animation powered by `matplotlib` |
| **Generational history** | Full evolution history stored in memory for post-hoc analysis |

---

## 🗂️ Repository Structure

```
CASim/
│
├── 1d_caipynb.py                  # 1D Cellular Automaton engine (Wolfram-style rules)
├── 3d_caipynb.py                  # 3D Cellular Automaton — general implementation
├── 3d_caipynb (1).py              # 3D CA — extended variant
├── 3d__περιπτωση_1ca.py           # 3D CA — Case Study 1 (threshold-based)
├── 3d__περιπτωση_2ca.py           # 3D CA — Case Study 2 (multi-state)
├── 3d__περιπτωση_3ca.py           # 3D CA — Case Study 3 (periodic emergence)
├── report.pdf                     # Full technical report with results and analysis
└── README.md                      # This file
```

---

## ⚙️ Technical Specifications

### Dependencies

```bash
pip install numpy matplotlib
```

> `tkinter` is included in the Python standard library and requires no separate installation.

| Library | Role |
|---|---|
| `numpy` | Grid state representation and vectorised evolution computation |
| `matplotlib` | 2D/3D visualisation and animation of CA generations |
| `tkinter` | Cross-platform graphical user interface for parameter input |

### System Requirements

- Python 3.7 or higher
- Works on Windows, macOS, and Linux
- Compatible with Google Colab (headless mode, matplotlib inline)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/PanagiotaGr/CASim-A-Parametric-Framework-for-Multi-Dimensional-Cellular-Automata-Simulation.git
cd CASim-A-Parametric-Framework-for-Multi-Dimensional-Cellular-Automata-Simulation
```

### 2. Install dependencies

```bash
pip install numpy matplotlib
```

### 3. Run a simulation

**1D Cellular Automaton:**
```bash
python 1d_caipynb.py
```

**3D Cellular Automaton:**
```bash
python 3d_caipynb.py
```

### 4. Configure via GUI

Upon launch, a graphical interface will prompt you to specify:

- Number of **cell states** (e.g., 2 for binary, 3 for ternary)
- **Grid size** (number of cells per dimension)
- **Number of generations** to simulate
- **Evolution rule** (decimal code for 1D) or **threshold** (for 2D/3D)
- **Neighbourhood type** (Moore / Von Neumann — 2D only)

---

## 🧪 Simulation Design

### 1D — Wolfram Elementary Automata

The 1D engine implements the classical Wolfram rule system. A rule number (0–255 for binary states) is decoded into a lookup table mapping every 3-cell neighbourhood pattern to a next-generation state. The full space-time evolution is rendered as a 2D heatmap using `plt.imshow()`.

```
Rule 30  →  Chaotic, pseudo-random patterns (used in Mathematica's RNG)
Rule 90  →  Sierpiński triangle fractal
Rule 110 →  Turing-complete behaviour (proven by Matthew Cook, 2004)
```

### 2D & 3D — Threshold-Based Activation

For higher-dimensional CA, cell state transitions are governed by a configurable **threshold `τ`**:

```
next_state(cell) = f( Σ active_neighbours, τ )
```

Boundary conditions are **periodic (toroidal)**, ensuring that every cell has a symmetric neighbourhood without edge effects. Visualisation proceeds frame-by-frame, rendering each generation in real time.

---

## 📊 Experimental Results

### Experiment 1 — 1D Binary CA, Rule 30

| Parameter | Value |
|---|---|
| States | 2 |
| Grid size | 20 |
| Generations | 20 |
| Rule | 30 |

**Result:** Highly non-linear, chaotic space-time pattern. No periodicity detected within the observed window. Consistent with Rule 30's Class III (chaotic) Wolfram classification.

---

### Experiment 2 — 3D Ternary CA, Threshold 5

| Parameter | Value |
|---|---|
| States | 3 |
| Grid size | 30 × 30 × 30 |
| Generations | 100 |
| Threshold | 5 |

**Result:** Self-organised spatial clustering emerges within the first 20 generations, followed by periodic structural oscillation. Demonstrates spontaneous symmetry breaking in a homogeneous initial condition.

---

## 🔭 Theoretical Background

### What are Cellular Automata?

A Cellular Automaton is a discrete computational model defined by:

- A **regular grid** of cells, each in one of *k* finite states
- A **neighbourhood** function mapping each cell to its local context
- A **transition rule** applied synchronously to all cells at each time step

Despite their simplicity, CA can exhibit the full range of Wolfram's four complexity classes:

| Class | Behaviour | Example Rule |
|---|---|---|
| I | Fixed-point convergence | Rule 0 |
| II | Periodic / stable structures | Rule 4 |
| III | Chaotic, aperiodic patterns | Rule 30 |
| IV | Complex, localised structures | Rule 110 |

### Why Multi-Dimensional CA?

Extending CA beyond 1D dramatically expands the emergent behaviour space. In 2D, John Conway's *Game of Life* demonstrated that threshold rules over Moore neighbourhoods can produce self-replicating structures and universal computation. In 3D, CA serve as models for crystalline growth, tumour dynamics, and reaction-diffusion systems.

---

## 🛣️ Future Work

The following extensions are planned or proposed for future research iterations:

- **Object-Oriented Refactoring** — Redesign the CA modules using OOP principles (`CAEngine`, `Grid`, `Rule`, `Visualiser` classes) for modularity and extensibility
- **4D Cellular Automata** — Extend the framework to support temporal CA with a fourth spatial dimension
- **Complexity Metrics** — Integrate quantitative analysis tools: Shannon entropy per generation, Lyapunov exponent estimation, and fractal dimension measurement
- **Machine Learning Integration** — Apply unsupervised learning (clustering, autoencoders) to identify and classify emergent patterns automatically
- **Rule Space Exploration** — Implement automated sweeping of rule/threshold parameter spaces with behaviour classification
- **GPU Acceleration** — Leverage CUDA or JAX for large-scale 3D simulations beyond CPU feasibility

---

## 📄 References

1. Wolfram, S. (2002). *A New Kind of Science*. Wolfram Media.
2. von Neumann, J. (1966). *Theory of Self-Reproducing Automata*. University of Illinois Press.
3. Gardner, M. (1970). Mathematical Games: The fantastic combinations of John Conway's new solitaire game "life". *Scientific American*, 223(4), 120–123.
4. Cook, M. (2004). Universality in elementary cellular automata. *Complex Systems*, 15(1), 1–40.
5. Sarkar, P. (2000). A brief history of cellular automata. *ACM Computing Surveys*, 32(1), 80–107.
6. IEEE Xplore — *Programmable Cellular Automata Based Encryption Algorithm* (course reference).

---

## 👩‍💻 Author

**Panagiota Grosdouli**
Department of Electrical and Computer Engineering
Democritus University of Thrace (DUTH), Greece
Student ID: 58523

*Course: Analysis and Synthesis of Complex Electronic Systems — Assignment 7, June 2025*

---

## 📜 License

This project was developed for academic purposes as part of a university course assignment. All rights reserved by the author. For use, citation, or adaptation, please contact the author directly.

---

<div align="center">
<sub>Built with 🧠 curiosity and ⚡ Python · Democritus University of Thrace · 2025</sub>
</div>
