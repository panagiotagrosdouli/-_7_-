<div align="center">

<h1>🧬 CASim</h1>
<h3>A Parametric Framework for Multi-Dimensional Cellular Automata Simulation</h3>

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dimensions-1D · 2D · 3D-7B2FBE?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/GUI-tkinter-2E8B57?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Viz-matplotlib-E07B00?style=for-the-badge"/>
</p>

<p><em>
A fully parametric, interactive simulation engine for exploring emergent complexity<br>
across one-, two-, and three-dimensional Cellular Automata.
</em></p>

</div>

---

## Overview

**CASim** is a research-oriented Python framework for designing, executing, and visually analysing **Cellular Automata (CA)** across multiple spatial dimensions. Every aspect of the simulation — state count, grid size, evolution rule, neighbourhood topology, and boundary conditions — is exposed as a user-configurable parameter, enabling systematic exploration of complex self-organising systems.

> *"Simple local rules. Global emergent complexity."*

The project was developed as **Assignment 7** for the course *Analysis and Synthesis of Complex Electronic Systems* (2024–2025), Department of Electrical and Computer Engineering, Democritus University of Thrace.

---

## Repository Structure

```
CASim/
│
├── 1d_caipynb.py            # 1D CA — Wolfram elementary rules, GUI, space-time heatmap
├── 3d_caipynb.py            # 3D CA — general threshold-based engine
├── 3d__case_1ca.py          # Case Study 1 — Birth/Survival (B/S) rule system
├── 3d__case_2ca.py          # Case Study 2 — Multi-state ageing + Shannon entropy
├── 3d__case_3ca.py          # Case Study 3 — Period detection + Wolfram classification
├── 3d_caipynb_extended.py   # Extended — configurable radius + rule-space phase diagram
├── report.pdf               # Full technical report with results and analysis
└── README.md
```

---

## Modules

### `1d_caipynb.py` — 1D Wolfram Automaton

Implements the classical Wolfram elementary CA. A decimal rule number is decoded into a neighbourhood → next-state lookup table; the full space-time evolution is rendered as a 2D heatmap.

| Parameter | Description |
|---|---|
| Grid size | Number of cells |
| Generations | Time steps to simulate |
| Rule number | Wolfram rule (e.g. 30, 90, 110) |
| States (k) | Binary, ternary, or k-ary |
| Initial condition | Single seed or random |

Notable rules: **Rule 30** (chaos), **Rule 90** (Sierpiński fractal), **Rule 110** (Turing-complete).

---

### `3d_caipynb.py` — 3D General Engine

Threshold-based 3D CA where live cells age through states `1 → 2 → … → k → 0`. A dead cell is born when its active-neighbour count equals the threshold exactly. Visualised as a real-time 3D scatter animation or orthogonal cross-section slices (XY / XZ / YZ).

---

### `3d__case_1ca.py` — Case Study 1: Birth/Survival Rules

Extends the engine with the **B/S notation** familiar from Conway's Game of Life, generalised to 3D:

- A **dead** cell is **born** when its active-neighbour count ∈ B  
- A **live** cell **survives** when its active-neighbour count ∈ S

Includes four ready-to-run presets:

| Preset | Rule | Behaviour |
|---|---|---|
| Conway 3D | B5,6,7 / S4,5,6,7 | Self-sustaining blobs |
| Amoeba | B3,5,6,7,8 / S5,6,7,8 | Expanding clusters |
| Crystals | B1,3 / S2 | Dendritic growth |
| Stable Fog | B4,6,8 / S3,6,9 | Persistent diffuse cloud |

---

### `3d__case_2ca.py` — Case Study 2: Multi-State Ageing

k-ary CA modelling **excitable media** (e.g. cardiac signal propagation, reaction-diffusion). Birth is controlled by an activity-sum window `[τ_min, τ_max]`. Includes:

- Per-generation **Shannon entropy** H(t) = −Σ pᵢ log₂ pᵢ
- Stacked area chart of the state distribution over time
- Entropy vs. maximum-entropy reference plot

---

### `3d__case_3ca.py` — Case Study 3: Periodic Emergence

Range-rule CA (`τ_low ≤ n ≤ τ_high`) with automated diagnostics:

- **Period detection** — identifies oscillators by comparing grid snapshots
- **Wolfram classification** (Classes I–IV) inferred from entropy and population variance
- **Diagnostic dashboard** — four-panel figure (population, entropy, two cross-section slices)

---

### `3d_caipynb_extended.py` — Extended: Radius & Rule Sweep

Two capabilities beyond the standard engine:

1. **Configurable neighbourhood radius r** — Moore neighbourhood expands from 26 cells (r=1) to 124 (r=2), 342 (r=3)
2. **Automated rule sweep** — sweeps all threshold values, records steady-state active fraction and entropy, and plots the **phase diagram** to locate order–chaos phase transitions

---

## Installation

```bash
pip install numpy matplotlib scipy
```

`tkinter` is included in the Python standard library. **Python 3.7+** · Windows / macOS / Linux / Google Colab.

---

## Quick Start

```bash
python 1d_caipynb.py           # 1D Wolfram automaton
python 3d_caipynb.py           # 3D general engine
python 3d__case_1ca.py         # Case Study 1 — B/S rules
python 3d__case_2ca.py         # Case Study 2 — entropy analysis
python 3d__case_3ca.py         # Case Study 3 — period detection
python 3d_caipynb_extended.py  # Extended — radius + phase diagram
```

Each script opens a GUI — configure parameters and press **Run**.

---

## Theoretical Background

A Cellular Automaton is defined by a regular grid of cells updated synchronously via a local transition rule. Despite this simplicity, CA exhibit Wolfram's four complexity classes:

| Class | Behaviour | Example |
|---|---|---|
| I | Converges to fixed point | Rule 0 |
| II | Periodic / stable structures | Rule 4 |
| III | Chaotic, aperiodic | Rule 30 |
| IV | Complex, localised structures | Rule 110 |

Extending to 3D enables modelling of physical phenomena: crystal growth, neural excitation waves, tumour dynamics, and reaction-diffusion chemistry.

---

## References

1. Wolfram, S. (2002). *A New Kind of Science*. Wolfram Media.
2. von Neumann, J. (1966). *Theory of Self-Reproducing Automata*. University of Illinois Press.
3. Gardner, M. (1970). Mathematical Games: Conway's Game of Life. *Scientific American*, 223(4).
4. Cook, M. (2004). Universality in elementary cellular automata. *Complex Systems*, 15(1).
5. Sarkar, P. (2000). A brief history of cellular automata. *ACM Computing Surveys*, 32(1).

---

## Author

**Panagiota Grosdouli** · Student ID 58523  
Department of Electrical and Computer Engineering, Democritus University of Thrace (DUTH)  
*Analysis and Synthesis of Complex Electronic Systems — Assignment 7, June 2025*

---

<div align="center">
<sub>Built with Python · Democritus University of Thrace · 2025</sub>
</div>
