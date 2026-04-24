import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


def is_grashof(a, b, c, d):
    """
    Checks if a 4-bar linkage satisfies Grashof's condition.
    Lengths: a (crank), b (coupler), c (rocker), d (ground)
    Grashof: S + L <= P + Q
    """
    lengths = sorted([a, b, c, d])
    S = lengths[0]
    L = lengths[3]
    P = lengths[1]
    Q = lengths[2]
    return (S + L) <= (P + Q)


def is_crank_rocker(a, b, c, d):
    """
    Returns True if the linkage is specifically a crank-rocker mechanism.

    Conditions (both must hold):
      1. Grashof condition: S + L <= P + Q
      2. The crank (a) is the shortest link, i.e. a <= b, a <= c, a <= d.
         Since a is adjacent to d (ground) in our setup, this guarantees
         that a rotates fully (crank) and c oscillates (rocker).

    This matches the reference paper (Roeder et al., 2025) which restricts
    the dataset to crank-rocker mechanisms only.
    """
    if not is_grashof(a, b, c, d):
        return False
    # a must be the shortest link
    return a <= b and a <= c and a <= d


def forward_kinematics_trajectory(a, b, c, d, p_dist, p_angle, num_points=100):
    """
    Compute the 2D coordinates of the coupler point over a full rotation of the crank.
    returns (x, y) arrays or None if non-computable.
    """
    # Define theta2 (crank angle) from 0 to 2*pi
    theta2 = np.linspace(0, 2*np.pi, num_points)

    # Pre-allocate arrays
    theta3 = np.zeros(num_points)
    theta4 = np.zeros(num_points)

    # Coupler coordinates
    Px = np.zeros(num_points)
    Py = np.zeros(num_points)

    for i, t2 in enumerate(theta2):
        # Freudenstein's equation approach to find theta4, then theta3
        K1 = d / a
        K2 = d / c
        K3 = (a**2 - b**2 + c**2 + d**2) / (2 * a * c)

        A = np.cos(t2) - K1 - K2 * np.cos(t2) + K3
        B = -2 * np.sin(t2)
        C = K1 - (K2 + 1) * np.cos(t2) + K3

        # We need b^2 - 4ac >= 0
        disc = B**2 - 4 * A * C
        if disc < 0:
            return None   # The linkage cannot reach this angle

        # Two modes (open/crossed), stick to one (negative mode)
        t4 = 2 * np.arctan((-B - np.sqrt(disc)) / (2 * A))

        # Calculate theta3
        s3 = (c * np.sin(t4) - a * np.sin(t2)) / b
        c3 = (d + c * np.cos(t4) - a * np.cos(t2)) / b
        t3 = np.arctan2(s3, c3)

        # Coupler point coordinates (crank pin A)
        Ax = a * np.cos(t2)
        Ay = a * np.sin(t2)

        # Coupler point P is offset by p_dist, p_angle from the crank pin A
        Px[i] = Ax + p_dist * np.cos(t3 + p_angle)
        Py[i] = Ay + p_dist * np.sin(t3 + p_angle)

    return Px, Py


def compute_fourier_descriptors(Px, Py, num_descriptors=15):
    """
    Computes scale-invariant Fourier Descriptors for a closed 2D trajectory.

    Normalisation applied:
      - DC component zeroed  (translation invariance)
      - All coefficients divided by |F_1|  (scale invariance)
      - Result is the same regardless of sampling scale (pixels vs physical units)

    Returns a float32 array of length num_descriptors * 2
    (real and imaginary parts interleaved).
    """
    Z = Px + 1j * Py
    fd = np.fft.fft(Z)

    # Translation invariance
    fd[0] = 0

    # Scale invariance
    f1_mag = np.abs(fd[1])
    if f1_mag > 1e-9:
        fd = fd / f1_mag

    fd_truncated = fd[1:num_descriptors + 1]

    features = np.zeros(num_descriptors * 2)
    features[0::2] = np.real(fd_truncated)
    features[1::2] = np.imag(fd_truncated)

    return features


def generate_dataset(num_samples=1000):
    dataset = []

    pbar = tqdm(total=num_samples)
    while len(dataset) < num_samples:
        a, b, c, d = np.random.uniform(0.5, 10.0, size=4)
        p_dist = np.random.uniform(0.5, 10.0)
        p_angle = np.random.uniform(-np.pi, np.pi)

        if not is_grashof(a, b, c, d):
            continue

        res = forward_kinematics_trajectory(a, b, c, d, p_dist, p_angle, num_points=128)
        if res is None:
            continue

        Px, Py = res
        if np.any(np.isnan(Px)) or np.any(np.isnan(Py)):
            continue

        fd = compute_fourier_descriptors(Px, Py, num_descriptors=15)

        max_len = max(a, b, c, d)
        norm_a = a / max_len
        norm_b = b / max_len
        norm_c = c / max_len
        norm_d = d / max_len
        norm_p = p_dist / max_len

        target = np.array([norm_a, norm_b, norm_c, norm_d, norm_p, p_angle])
        dataset.append({'fd': fd, 'params': target})
        pbar.update(1)

    pbar.close()
    return dataset


if __name__ == "__main__":
    print("Testing data generation methods...")
    print("Testing is_crank_rocker filter:")
    # Example: a=1, b=3, c=3, d=3 => crank-rocker (a shortest, Grashof holds)
    print(f"  a=1,b=3,c=3,d=3 -> crank_rocker={is_crank_rocker(1,3,3,3)}")
    # Example: all equal => all Grashof, but a not shortest if others are smaller
    print(f"  a=3,b=1,c=3,d=3 -> crank_rocker={is_crank_rocker(3,1,3,3)}  (b shorter, so not a crank-rocker)")
    ds = generate_dataset(5)
    print(f"\nGenerated {len(ds)} valid linkage geometries.")
    print("FD feature vector size:", ds[0]['fd'].shape)
    print("Target param ratios (a, b, c, d, p, angle):", ds[0]['params'])
