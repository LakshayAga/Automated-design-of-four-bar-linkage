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
            return None # The linkage cannot reach this angle
            
        # Two modes (open/crossed), let's stick to one (e.g., negative mode)
        # Using the standard half-angle substitution
        t4_1 = 2 * np.arctan((-B - np.sqrt(disc)) / (2 * A))
        t4_2 = 2 * np.arctan((-B + np.sqrt(disc)) / (2 * A))
        
        t4 = t4_1 # picking one valid assembly mode
        
        # Calculate theta3
        # a*cos(t2) + b*cos(t3) = d + c*cos(t4)
        # a*sin(t2) + b*sin(t3) = c*sin(t4)
        s3 = (c * np.sin(t4) - a * np.sin(t2)) / b
        c3 = (d + c * np.cos(t4) - a * np.cos(t2)) / b
        t3 = np.arctan2(s3, c3)
        
        # Coupler point coordinates
        # Position of crank pin A
        Ax = a * np.cos(t2)
        Ay = a * np.sin(t2)
        
        # Coupler point P is offset by p_dist, p_angle from the crank pin A
        # The angle of the coupler link is t3. 
        # So the absolute angle of AP is t3 + p_angle.
        Px[i] = Ax + p_dist * np.cos(t3 + p_angle)
        Py[i] = Ay + p_dist * np.sin(t3 + p_angle)
        
    return Px, Py

def compute_fourier_descriptors(Px, Py, num_descriptors=15):
    """
    Computes numerically condensed path functions (Fourier Descriptors) 
    for the given closed 2D trajectory.
    """
    # Combine to complex array
    Z = Px + 1j * Py
    
    # Compute FFT
    fd = np.fft.fft(Z)
    
    # We want a scale/translation invariant representation if needed,
    # but the model might benefit from raw normalized inputs.
    # At minimum, translation invariance: set fd[0] = 0
    fd[0] = 0
    
    # We take the first 'num_descriptors' low frequency components
    # The lowest frequencies hold the global shape
    fd_truncated = fd[1:num_descriptors+1]
    
    # We represent them as [real, imag] flattened
    features = np.zeros(num_descriptors * 2)
    features[0::2] = np.real(fd_truncated)
    features[1::2] = np.imag(fd_truncated)
    
    return features

def generate_dataset(num_samples=1000):
    dataset = []
    
    pbar = tqdm(total=num_samples)
    while len(dataset) < num_samples:
        # Generate random dimensions
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
        
        # normalize dimensions based on the proposal
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
    ds = generate_dataset(5)
    print(f"Generated {len(ds)} valid linkage geometries.")
    print("FD feature vector size:", ds[0]['fd'].shape)
    print("Target param ratios (a, b, c, d, p, angle):", ds[0]['params'])

