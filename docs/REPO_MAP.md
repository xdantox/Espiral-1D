# Repository map — Espiral-1D

The repository root contains prototypes and tools from different stages of research. The descriptions below are based on the code inspected; check each script and its inputs before running it. There is no single entry point that runs every analysis.

## Model and dynamics

- `Heff.py`: parameters and routines for calculating the effective field.
- `E.py`: energy calculation.
- `dot.py`: helper operations for dot products and vectorized calculations.
- `chain.py`: orchestration of a chain simulation; check input paths first.
- `rot.py`, `rot_methods.py`: evolution and integration routines.
- `B_ext.py`: external magnetic field.
- `cadena0.py`, `1spin.py`, `rand.py`: initial states and spin utilities.

## States, ansatz, and optimization

- `direct3.py`, `direct3_PBC.py`, `direct3_PBC_folded.py`: variants for constructing/fitting states with boundary conditions and ansatz functions.
- `JDS.py`, `Jdiscreta.py`, `Jmodified.py`: model/coupling variants.
- `alpha.py`, `alpha2.py`, `m2.py`: auxiliary calculations/models; review their context before treating them as part of the spin model.

## State and history analysis

- `relax.py`, `relax3D.py`, `relax3Dexact.py`: relaxation and state analysis.
- `history.py`: analysis of a saved time sequence.
- `check.py`, `fitq.py`, `perfil_banda.py`, `plot_ansatz_theta.py`, `plot_q_local.py`: checks and specialized analyses.
- `stab_termica.py`, `flop.py`, `flopproxy.py`: thermal-stability and transition protocols.

## Spectra, transport, and visualization

- `fft.py`, `fftBEoptimized.py`: FFT analysis.
- `transmision.py`: magnon transmission analysis.
- `animation.py`, `gif.py`: animations.
- `fig.py`, `gap.py`, `gaps.py`, `gapteorico.py`, `satelite.py`, `satelite2.py`, `satelitem.py`: figures and specialized analyses.

## Data and figures

- `.npy` files are simulation data/histories and may be large.
- `.png`, `.pdf`, and `.gif` files are visual results.
- `helix_spin_chain.html` and `html.html` are HTML resources.

## Suggested reading order

1. `Heff.py` and `E.py` for the model conventions.
2. `cadena0.py` to see how some initial states are constructed.
3. `chain.py` and the `rot*.py` routines for the dynamics.
4. `relax3D.py` / `relax.py` for diagnostics.
5. `fft.py` and `transmision.py` for observables.

This order is only a guide: some scripts are independent, experimental, or require local files that are not available in a clean checkout.
