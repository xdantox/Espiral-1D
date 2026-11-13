# Espiral-1D

Numerical simulations and analysis of one-dimensional magnetic chains, focusing on spin spirals, antiferromagnetic states, solitons, and magnon transport.

> **Repository status:** Research in progress. The root contains scripts from different stages of development, along with auxiliary data and results. The scripts do not form a single end-to-end pipeline, and some require local input files that are not included in the repository.

## Overview

This project explores spin dynamics and modulated states, including ansatz fitting, spatial and frequency spectra, thermal stability, and excitation transmission.

## Getting started

1. Read the [repository map](docs/REPO_MAP.md) to identify the main modules and a suggested reading order.
2. Use Python 3.10 or later and install the dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   python -m pip install -r requirements.txt
   ```

3. Before running an analysis, check which input files and paths the script expects. Several scripts load specific `.npy` histories that are not included in the repository. Scripts importing `relax3D.py` also require that module and, in some cases, its input data.

Do not run `chain.py`, `stab_termica.py`, or other long calculations without first checking their parameters, expected output size, and data requirements.

## Modules at a glance

- `Heff.py`, `E.py`, `dot.py`: effective field, energy, and model operations.
- `chain.py`, `rot.py`, `rot_methods.py`: dynamics and integration routines.
- `cadena0.py`, `direct3*.py`: initial-state construction and ansatz optimization.
- `relax.py`, `relax3D.py`, `history.py`: state and time-history diagnostics.
- `fft.py`, `fftBEoptimized.py`, `transmision.py`: spectral and transport analyses.
- `stab_termica.py`, `flopproxy.py`: thermal/noise stability tests and related workflows.
- `plot_*.py`, `fig*.py`, `gap*.py`, `satelite*.py`: specialized analyses and visualizations.

This list is not exhaustive; consult [docs/REPO_MAP.md](docs/REPO_MAP.md) before choosing a script.

## Data and results

Simulation histories can be large and are often stored as `.npy` files. The `.gitignore` excludes common binary data formats to help avoid accidentally committing large outputs. Files already tracked in the repository are not removed or relocated automatically.

## Reproducibility and current limitations

- Some scripts define input paths directly in the source; update them to match your local data files.
- Set and record random seeds when comparing runs. Not all scripts accept seeds through command-line arguments.
- Results may depend on system size, physical parameters, library versions, and hardware. Record these settings alongside each figure or history.
- This README does not declare a license. If no license is published, contact the author before reusing the code.

## Documentation

- [Repository map and script relationships](docs/REPO_MAP.md)
