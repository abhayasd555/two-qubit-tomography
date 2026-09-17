# Two-Qubit Polarization Tomography

**Live tool:** https://abhayasd555.github.io/two-qubit-tomography/

A self-contained, browser-based instrument for reconstructing the density matrix of a
two-photon polarization state from coincidence-count data — the standard measurement
produced by a spontaneous parametric down-conversion (SPDC) experiment.

Enter the 16 coincidence counts from a tomographically complete set of polarization
projections, and the tool reconstructs ρ two ways:

1. **Linear tomography** — a direct least-squares inversion. Exact for noise-free data,
   but for real (noisy) counts it routinely produces a matrix with negative eigenvalues,
   i.e. not a physically valid quantum state. The tool flags this explicitly.
2. **Maximum-likelihood estimation** — ρ is parametrized as ρ = T†T / Tr(T†T) with T a
   lower-triangular matrix, which is Hermitian, unit-trace, and positive-semidefinite by
   construction. The parameters are fit by minimizing a Gaussian log-likelihood via
   gradient-based optimization (Adam, numerical gradients, multiple random restarts).

## Features

- 16-row input table for the coincidence counts, pre-loaded with a representative example
- Linear and maximum-likelihood density matrices, both displayed as complex 4×4 grids
- Physicality check (Hermiticity / eigenvalues) with a pass/fail indicator
- Derived quantities: purity Tr(ρ²), von Neumann entropy, linear entropy, fidelity to the
  four Bell states
- Three-dimensional column-chart visualization of the real and imaginary parts of ρ, with
  editable axis labels and a settable z-axis scale
- Runs entirely client-side — no server, no build step, no dependencies. Works offline.

## Usage

Open the [live version](https://abhayasd555.github.io/two-qubit-tomography/), or download
`index.html` and open it directly in any modern browser.

1. Enter your 16 measured coincidence counts (or use the pre-loaded example).
2. Press **Compute**.
3. Read off the maximum-likelihood density matrix, its eigenvalues, and the derived
   quantities. Use the **Figure Axes** panel to relabel the basis states or fix the
   z-axis scale for comparing figures across runs.

### Measurement convention

The 16 rows correspond to coincidence measurements between mode 1 projected onto one of
{H, V, D, L, R} and mode 2 projected onto one of {H, V, D, L, R}, where

```
|D⟩ = (|H⟩ + |V⟩) / √2
|L⟩ = (|H⟩ + i|V⟩) / √2
|R⟩ = (|H⟩ − i|V⟩) / √2
```

Settings 1–4 (HH, HV, VV, VH) must stay in that order — their sum fixes the overall
pair-production rate `N` used to normalize the rest of the reconstruction.

## Running locally

No build step is required.

```bash
git clone https://github.com/abhayasd555/two-qubit-tomography.git
cd two-qubit-tomography
open index.html      # macOS
# or just double-click index.html in Finder/Explorer
```

## Development

Everything — math and UI — lives in the single `index.html` file:

- Complex linear algebra, the 16-state measurement basis, linear tomography via
  least-squares, a Jacobi eigenvalue solver, the T-matrix physical parametrization, and
  the Adam-based maximum-likelihood optimizer are all implemented in vanilla JavaScript
  with no external libraries.
- The UI is plain HTML/CSS with an intentionally retro, Windows-95-style GUI skin.

To make changes, edit `index.html` and push:

```bash
git add -A
git commit -m "describe your change"
git push
```

GitHub Pages redeploys automatically within about a minute of each push.

## Validation

The linear-tomography routine was checked against an independently worked reference
calculation and matches to 4 decimal places. The maximum-likelihood optimizer converges
to the same optimum from many independent starting points and under several different
optimization methods (Adam, BFGS, Powell), which is evidence it reaches the global
optimum rather than a local one.

## Known limitations

- No propagation of Poissonian shot-noise into error bars on ρ (would require a full
  derivative or Monte-Carlo error analysis).
- Assumes exactly the 16-setting measurement scheme described above; it does not
  generalize to other tomographic bases or to more than two qubits.

## License

No license has been specified yet — all rights reserved by default. Add a `LICENSE` file
if you want to permit reuse (e.g. MIT).
