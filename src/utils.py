
import numpy as np
import logging

def generate_synthetic_data(size):
    # Generate synthetic data
    data = np.random.rand(size, 10)
    labels = np.random.randint(0, 2, size)
    data[:, -1] = labels
    return data

def calculate_drift(data, predictions):
    # Calculate drift
    drift = np.mean(np.abs(data[:, -1] - predictions))
    return drift
