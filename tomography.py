"""
Two-qubit polarization state tomography.

Reconstructs the density matrix rho of a two-photon polarization state from 16
coincidence counts measured over a tomographically complete set of projective
measurements, by two methods:

  1. linear_tomography    - direct least-squares inversion. Exact for noise-free
                             data, but for real (noisy) counts it routinely
                             produces a non-physical result (negative eigenvalues).

  2. maximum_likelihood_tomography - rho is parametrized as rho = T^dagger T / Tr(T^dagger T)
                             with T lower-triangular, which is Hermitian, unit-trace
                             and positive-semidefinite by construction. The 16 real
                             parameters of T are fit by minimizing a Gaussian
                             log-likelihood.

Requires: numpy, scipy.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Single-qubit polarization kets and the 16-state measurement basis
# ---------------------------------------------------------------------------

_s = 1 / np.sqrt(2)
H = np.array([1, 0], dtype=complex)
V = np.array([0, 1], dtype=complex)
D = np.array([_s, _s], dtype=complex)          # (|H> + |V>) / sqrt(2)
L = np.array([_s, 1j * _s], dtype=complex)     # (|H> + i|V>) / sqrt(2)
R = np.array([_s, -1j * _s], dtype=complex)    # (|H> - i|V>) / sqrt(2)

# The sixteen tomographically complete projective measurements: mode 1 (x) mode 2.
# Settings 1-4 (HH, HV, VV, VH) must stay first and in this order -- their sum
# fixes the overall pair-production rate N used to normalize the reconstruction.
MEASUREMENT_BASIS = [
    (H, H), (H, V), (V, V), (V, H),
    (R, H), (R, V), (D, V), (D, H),
    (D, R), (D, D), (R, D), (H, D),
    (V, D), (V, L), (H, L), (R, L),
]
PSIS = [np.kron(a, b) for a, b in MEASUREMENT_BASIS]  # sixteen length-4 kets

# Pauli basis used to span the space of 4x4 Hermitian matrices.
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS = [_I2, _X, _Y, _Z]
GAMMA = [[np.kron(_PAULIS[i], _PAULIS[j]) for j in range(4)] for i in range(4)]


# ---------------------------------------------------------------------------
# Linear tomography
# ---------------------------------------------------------------------------

def linear_tomography(counts: np.ndarray) -> tuple[np.ndarray, float]:
    """Reconstruct rho by direct least-squares inversion of n_n = N <psi_n|rho|psi_n>.

    Parameters
    ----------
    counts : array of 16 coincidence counts, in the order of MEASUREMENT_BASIS.

    Returns
    -------
    rho : the reconstructed 4x4 density matrix (Hermitian, unit-trace, but NOT
          guaranteed positive-semidefinite -- check its eigenvalues).
    N   : the inferred pair-production rate, N = n1 + n2 + n3 + n4.
    """
    counts = np.asarray(counts, dtype=float)
    N = counts[0] + counts[1] + counts[2] + counts[3]

    # c[n, i, j] = <psi_n| sigma_i (x) sigma_j |psi_n>, all real since the
    # operator is Hermitian and the expectation is taken in a definite state.
    c = np.zeros((16, 4, 4))
    for n, psi in enumerate(PSIS):
        for i in range(4):
            for j in range(4):
                c[n, i, j] = np.real(np.conj(psi) @ GAMMA[i][j] @ psi)

    # rho = (1/4) * sum_{i,j} r_ij * Gamma_ij, with r_00 = 1 fixing Tr(rho) = 1.
    # Solve for the other 15 real coefficients by least squares.
    idx_list = [(i, j) for i in range(4) for j in range(4) if (i, j) != (0, 0)]
    A = np.array([[(N / 4) * c[n, i, j] for i, j in idx_list] for n in range(16)])
    b = np.array([counts[n] - (N / 4) * c[n, 0, 0] for n in range(16)])
    x, *_ = np.linalg.lstsq(A, b, rcond=None)

    r = np.zeros((4, 4))
    r[0, 0] = 1.0
    for (i, j), xi in zip(idx_list, x):
        r[i, j] = xi

    rho = sum((r[i, j] / 4) * GAMMA[i][j] for i in range(4) for j in range(4))
    return rho, N


# ---------------------------------------------------------------------------
# Maximum-likelihood tomography
# ---------------------------------------------------------------------------

def _t_matrix(t: np.ndarray) -> np.ndarray:
    """Build the lower-triangular T(t) from 16 real parameters."""
    T = np.zeros((4, 4), dtype=complex)
    T[0, 0] = t[0]
    T[1, 0] = t[4] + 1j * t[5];  T[1, 1] = t[1]
    T[2, 0] = t[10] + 1j * t[11]; T[2, 1] = t[6] + 1j * t[7];  T[2, 2] = t[2]
    T[3, 0] = t[14] + 1j * t[15]; T[3, 1] = t[12] + 1j * t[13]; T[3, 2] = t[8] + 1j * t[9]; T[3, 3] = t[3]
    return T


def _rho_physical(t: np.ndarray) -> np.ndarray:
    """rho(t) = T^dagger T / Tr(T^dagger T) -- Hermitian, unit-trace, PSD by construction."""
    T = _t_matrix(t)
    M = T.conj().T @ T
    return M / np.real(np.trace(M))


def _initial_guess(rho_linear: np.ndarray) -> np.ndarray:
    """Seed T from a positivity-corrected version of the linear estimate.

    Clips negative eigenvalues of rho_linear to a small positive floor,
    renormalizes, then Cholesky-decomposes (via a row/column flip, since we
    need T lower-triangular with T^dagger T = rho, not T T^dagger = rho).
    """
    eigvals, eigvecs = np.linalg.eigh((rho_linear + rho_linear.conj().T) / 2)
    clipped = np.clip(eigvals, 1e-6, None)
    clipped /= clipped.sum()
    rho_pos = eigvecs @ np.diag(clipped) @ eigvecs.conj().T

    J = np.fliplr(np.eye(4))
    A = J @ rho_pos @ J
    A = (A + A.conj().T) / 2 + 1e-12 * np.eye(4)
    Lc = np.linalg.cholesky(A)
    T = J @ Lc.conj().T @ J

    t = np.zeros(16)
    t[0], t[1], t[2], t[3] = (np.real(T[0, 0]), np.real(T[1, 1]),
                               np.real(T[2, 2]), np.real(T[3, 3]))
    t[4], t[5] = np.real(T[1, 0]), np.imag(T[1, 0])
    t[10], t[11] = np.real(T[2, 0]), np.imag(T[2, 0])
    t[6], t[7] = np.real(T[2, 1]), np.imag(T[2, 1])
    t[14], t[15] = np.real(T[3, 0]), np.imag(T[3, 0])
    t[12], t[13] = np.real(T[3, 1]), np.imag(T[3, 1])
    t[8], t[9] = np.real(T[3, 2]), np.imag(T[3, 2])
    return t


def _neg_log_likelihood(t: np.ndarray, counts: np.ndarray, N: float) -> float:
    rho = _rho_physical(t)
    nbar = np.array([N * np.real(np.conj(psi) @ rho @ psi) for psi in PSIS])
    nbar = np.maximum(nbar, 1e-9 * N)
    return np.sum((nbar - counts) ** 2 / (2 * nbar))


def maximum_likelihood_tomography(
    counts: np.ndarray, N: float | None = None, rho_linear: np.ndarray | None = None,
    n_restarts: int = 8, seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Fit the physical (Hermitian, unit-trace, PSD) rho that best explains `counts`.

    Parameters
    ----------
    counts     : array of 16 coincidence counts, in the order of MEASUREMENT_BASIS.
    N          : pair-production rate; computed from `counts` if not given.
    rho_linear : linear-tomography estimate used to seed the optimizer; computed
                 from `counts` if not given.
    n_restarts : number of random restarts (in addition to the primary seeded start).
    seed       : RNG seed for the restarts, for reproducibility.

    Returns
    -------
    rho : the maximum-likelihood density matrix.
    L   : the negative log-likelihood at the optimum (lower is a better fit).
    """
    counts = np.asarray(counts, dtype=float)
    if N is None:
        N = counts[0] + counts[1] + counts[2] + counts[3]
    if rho_linear is None:
        rho_linear, _ = linear_tomography(counts)

    t0 = _initial_guess(rho_linear)
    rng = np.random.default_rng(seed)

    best_t, best_val = None, np.inf
    for trial in range(n_restarts + 1):
        x0 = t0 if trial == 0 else t0 + rng.normal(scale=0.15, size=16)
        res = minimize(_neg_log_likelihood, x0, args=(counts, N), method="Powell",
                        options={"maxiter": 20000, "maxfev": 80000, "xtol": 1e-12, "ftol": 1e-14})
        res = minimize(_neg_log_likelihood, res.x, args=(counts, N), method="BFGS",
                        options={"maxiter": 2000, "gtol": 1e-12})
        if res.fun < best_val:
            best_val, best_t = res.fun, res.x

    return _rho_physical(best_t), best_val


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def purity(rho: np.ndarray) -> float:
    """Tr(rho^2): 1 for a pure state, 1/4 for the maximally mixed two-qubit state."""
    return float(np.real(np.trace(rho @ rho)))


def von_neumann_entropy(rho: np.ndarray) -> float:
    """S = -sum(lambda_i log2 lambda_i), in bits."""
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]
    return float(-np.sum(eigvals * np.log2(eigvals)))


def linear_entropy(rho: np.ndarray) -> float:
    """S_L = (4/3) * (1 - Tr(rho^2)), the standard two-qubit linear entropy."""
    return (4 / 3) * (1 - purity(rho))


BELL_STATES = {
    "Phi+": np.array([_s, 0, 0, _s], dtype=complex),
    "Phi-": np.array([_s, 0, 0, -_s], dtype=complex),
    "Psi+": np.array([0, _s, _s, 0], dtype=complex),
    "Psi-": np.array([0, _s, -_s, 0], dtype=complex),
}


def bell_fidelities(rho: np.ndarray) -> dict[str, float]:
    """Fidelity <bell|rho|bell> to each of the four Bell states."""
    return {name: float(np.real(np.conj(ket) @ rho @ ket)) for name, ket in BELL_STATES.items()}


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # A representative set of 16 coincidence counts (same order as MEASUREMENT_BASIS).
    example_counts = [
        34749, 324, 35805, 444, 16324, 17521, 13441, 16901,
        17932, 32028, 15132, 17238, 13171, 17170, 16722, 33586,
    ]

    np.set_printoptions(precision=4, suppress=True)

    rho_lin, N = linear_tomography(example_counts)
    eigs_lin = np.sort(np.linalg.eigvalsh(rho_lin))[::-1]
    print(f"N = {N:.0f} pairs\n")
    print("Linear tomography:")
    print(rho_lin)
    print(f"eigenvalues: {eigs_lin}")
    print(f"Tr(rho^2)  : {purity(rho_lin):.4f}  "
          f"({'PHYSICAL' if eigs_lin.min() > -1e-6 else 'NOT PHYSICAL'})\n")

    rho_ml, L = maximum_likelihood_tomography(example_counts, N, rho_lin)
    eigs_ml = np.sort(np.linalg.eigvalsh(rho_ml))[::-1]
    print("Maximum-likelihood tomography:")
    print(rho_ml)
    print(f"eigenvalues: {eigs_ml}")
    print(f"Tr(rho^2)       : {purity(rho_ml):.4f}")
    print(f"von Neumann S   : {von_neumann_entropy(rho_ml):.4f} bit")
    print(f"linear entropy  : {linear_entropy(rho_ml):.4f}")
    print(f"likelihood at optimum: {L:.2f}")
    print(f"Bell-state fidelities: {bell_fidelities(rho_ml)}")
