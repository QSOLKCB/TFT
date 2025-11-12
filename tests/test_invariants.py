import numpy as np
from tft.resonance import spd_matrix, rotation_matrix, rotate_rank2, invariants, phi_lock_pair

def test_rotation_preserves_invariants():
    T = spd_matrix(dim=3, seed=123)
    R = rotation_matrix(dim=3, seed=456)
    T_rot = rotate_rank2(T, R)
    inv0 = invariants(T)
    inv1 = invariants(T_rot)
    assert np.isclose(inv0["fro"], inv1["fro"], rtol=1e-10, atol=1e-12)
    assert np.allclose(inv0["eigvals"], inv1["eigvals"], rtol=1e-10, atol=1e-12)

def test_phi_lock_quadrature():
    # A random signal; ensure quadrature parts are ~orthogonal with matched energy
    g = np.random.default_rng(42)
    x = g.normal(size=4096)
    a, b = phi_lock_pair(x)
    # Zero-mean before correlation
    a = a - a.mean()
    b = b - b.mean()
    corr = float(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    assert abs(corr) < 1e-2  # near-orthogonal
    # Energies similar
    ea = np.sqrt(np.mean(a**2))
    eb = np.sqrt(np.mean(b**2))
    assert np.isclose(ea, eb, rtol=1e-2, atol=1e-6)
