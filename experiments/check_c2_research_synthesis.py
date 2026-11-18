"""Deterministic CPU checks for the C2 and SCORE-DRO research comparison.

Run with `python experiments/check_c2_research_synthesis.py` (NumPy is required).
The confidence limits below concern iid identities, not a census of a fixed roster.
"""

from math import ceil, expm1, log, log1p, sqrt

import numpy as np


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def risk(kernel, probe, gallery):
    denominator = 1.0 - probe @ gallery
    assert denominator > 0
    return (probe @ kernel @ gallery) / denominator


def check_star():
    star = np.zeros((8, 8))
    star[0, 1:] = star[1:, 0] = 1.0
    t = np.array([0.0] + [1.0] * 7)
    uniform = np.full(8, 1 / 8)
    rho = 2.0
    lam = 0.8
    shell_denom = 1 - (1 + rho**2) / 8

    def objective(q):
        close(q.sum(), 1)
        assert 8 * np.sum((q - uniform) ** 2) <= rho**2 + 1e-12
        return lam * (q @ t) + (1 - lam) * risk(star, q, q)

    def equal_leaves(x):
        return np.array([x] + [(1 - x) / 7] * 7)

    # On the shell D is fixed. The exact objective is a concave quadratic in x=q[0].
    x_shell = (1 - lam * shell_denom / (2 * (1 - lam))) / 2
    close(x_shell, 1 / 8)
    shell_max = lam * (1 - x_shell) + (1 - lam) * 2 * x_shell * (1 - x_shell) / shell_denom
    # Eq. (12) optimizes the frozen-denominator *increment* over the whole ball.
    # At fixed x, the increment includes a positive multiple of ||q||^2 (r0>0),
    # so its maximum uses D_min; the physical witness below attains this bound.
    physical = uniform.copy()
    physical[1] += sqrt(3 / 7)
    physical[2:] -= sqrt(3 / 7) / 6
    assert np.all((physical >= 0) & (physical <= 10 / 8))
    close(1 - physical @ physical, shell_denom)
    close(objective(physical), shell_max)
    close(shell_max, 49 / 60)

    signed = equal_leaves((1 - sqrt(7)) / 8)
    close(8 * np.sum((signed - uniform) ** 2), 1)
    witness = objective(signed)
    close(witness, (14 + sqrt(7)) / 20)
    assert signed[0] < 0 and witness > shell_max

    # For x<0, numerator 2x(1-x)<0, hence equal leaves maximize D and risk.
    # Then F = .8(1-x) + .2*14x/(6+8x), strictly concave on the feasible x<0.
    # For x>=0, replacing D by D_min gives the upper bound shell_max.
    x_best = (sqrt(21) - 6) / 8
    close(-lam + (1 - lam) * 84 / (6 + 8 * x_best) ** 2, 0)
    signed_max = objective(equal_leaves(x_best))
    close(signed_max, 7 / 4 - sqrt(21) / 5)
    assert signed_max > witness
    print(f"star Eq12/physical={shell_max:.9f} signed_witness={witness:.9f} "
          f"signed_global={signed_max:.9f} x_global={x_best:.9f}")


def two_cliques(labels):
    labels = np.asarray(labels)
    return (labels[:, None] == labels[None, :]).astype(float) - np.eye(labels.size)


def check_graphs():
    old_labels = np.array([1, 1, 1, 1, -1, -1, -1, -1])
    new_labels = np.array([1, 1, -1, -1, 1, 1, -1, -1])
    old = two_cliques(old_labels)
    new = two_cliques(new_labels)
    block = np.ones((4, 4)) - np.eye(4)
    cube = np.block([[np.zeros((4, 4)), block], [block, np.zeros((4, 4))]])
    center = np.eye(8) - np.ones((8, 8)) / 8
    uniform = np.full(8, 1 / 8)
    r0 = risk(old, uniform, uniform)
    close(r0, 3 / 7)

    def leading_curvature(h):
        return np.linalg.eigvalsh((center @ h @ center + r0 * center) / 8)[-1]

    # The earlier graph pair matches rows, nominal risk and unsigned centered norm.
    close(old.sum(axis=1), cube.sum(axis=1))
    close(risk(cube, uniform, uniform), r0)
    close(np.linalg.norm(center @ old @ center, 2), np.linalg.norm(center @ cube @ center, 2))
    c_old, c_cube = leading_curvature(old), leading_curvature(cube)
    close(c_old, 3 / 7)
    close(c_cube, 5 / 28)
    print(f"old_graphs nominal={r0:.9f} curvature_2K4={c_old:.9f} "
          f"curvature_cube={c_cube:.9f} excess_ratio={c_old / c_cube:.9f}")

    # Conjugation by a permutation also preserves every individual robust maximum
    # for a permutation-invariant shift set, while a common shift sees regressions.
    permutation = [0, 1, 4, 5, 2, 3, 6, 7]
    close(old[np.ix_(permutation, permutation)], new)
    close(np.linalg.eigvalsh(old), np.linalg.eigvalsh(new))
    difference = new - old
    eigs = np.linalg.eigvalsh(center @ difference @ center)
    close(eigs[[0, -1]], [-4, 4])
    # Both individual graphs are realizable by hard-threshold cosine verification.
    for graph in [old, new, cube]:
        vectors = np.linalg.cholesky(np.eye(8) + graph / 4)
        gram = vectors @ vectors.T
        close(np.diag(gram), np.ones(8))
        close((gram > 1 / 8) & (~np.eye(8, dtype=bool)), graph)
    rho = 0.1
    outcomes = []
    for direction, sign in [(old_labels, -1), (new_labels, 1)]:
        q = (1 + rho * direction) / 8
        assert np.all(q > 0)
        close(8 * ((q - uniform) @ (q - uniform)), rho**2)
        measured = risk(new, q, q) - risk(old, q, q)
        close(measured, risk(difference, q, q))
        close(measured, sign * 4 * rho**2 / (7 - rho**2))
        outcomes.append(measured)
    print(f"permuted_2K4 rho={rho:.2f} signed_min={outcomes[0]:.9f} "
          f"signed_max={outcomes[1]:.9f} individual_maxima_difference=0")


def check_polarization():
    kernel = np.array([[0, .2, .1, .9], [.2, 0, .5, .3],
                       [.1, .5, 0, .4], [.9, .3, .4, 0]])
    p = np.array([.1, .2, .3, .4])
    v = np.array([.6, -.2, .1, -.5])
    close(v.sum(), 0)
    a = 0.06
    base = risk(kernel, p, p)
    contrast = 0.0
    for s in [1, -1]:
        for t in [1, -1]:
            probe, gallery = p + s * a * v, p + t * a * v
            assert np.all(probe > 0) and np.all(gallery > 0)
            denom = 1 - probe @ gallery
            contrast += s * t * denom * (risk(kernel, probe, gallery) - base)
    contrast /= 4 * a**2
    expected = v @ (kernel + base * np.eye(len(p))) @ v
    close(contrast, expected)
    print(f"four_role contrast={contrast:.9f} exact_curvature={expected:.9f}")


def zero_positive_bounds(n, alpha, rho, cap):
    # An even number of iid, atomless identity units supplies n/2 disjoint pairs.
    # The hidden-hub law realizes equality on observing zero positive paired flips.
    assert n >= 2 and n % 2 == 0 and 0 < alpha < 1 and rho >= 0 and cap >= 1
    p = -expm1(log(alpha) / n)
    nominal = -expm1(2 * log(alpha) / n)
    close(nominal, p * (2 - p))
    s = min(1, cap * p, p + rho * sqrt(p * (1 - p)))
    hub_lower = s * (2 - s)
    # chi2(Q x Q || P x P) <= (1 + rho^2)^2 - 1; cap gives w_i*w_j <= cap^2.
    factor = rho * sqrt(2 + rho**2)
    robust_upper = min(1, cap**2 * nominal, nominal + factor * sqrt(nominal * (1 - nominal)))
    assert nominal - 1e-14 <= hub_lower <= robust_upper + 1e-14
    return nominal, hub_lower, robust_upper


def first_even_n(predicate):
    high = 1
    while not predicate(2 * high):
        high *= 2
    low = 0
    while high - low > 1:
        mid = (low + high) // 2
        if predicate(2 * mid):
            high = mid
        else:
            low = mid
    return 2 * high


def check_confidence():
    alpha, rho, cap = .05, .1, 10
    nominal, lower, upper = zero_positive_bounds(2000, alpha, rho, cap)
    close((1 - nominal) ** 1000, alpha)
    print(f"iid_zero_positive n=2000 rho={rho:.2f} W={cap} nominal={nominal:.9f} "
          f"common_shift_lower={lower:.9f} common_shift_upper={upper:.9f}")
    for target, expected_n in [(1e-3, 5990), (1e-4, 59912)]:
        nominal_n = 2 * ceil(log(alpha) / log1p(-target))
        close(nominal_n, expected_n)
        def bounds(n):
            return zero_positive_bounds(n, alpha, rho, cap)

        counts = [first_even_n(lambda n: bounds(n)[i] <= target) for i in range(3)]
        assert counts[0] == nominal_n
        assert counts[0] <= counts[1] <= counts[2]
        print(f"iid_sample_size target={target:.4g} nominal_sharp={nominal_n} "
              f"common_shift_necessary={counts[1]} common_shift_sufficient={counts[2]}")


if __name__ == "__main__":
    check_star()
    check_graphs()
    check_polarization()
    check_confidence()
