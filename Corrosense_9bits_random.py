# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:49:33 2026

@author: Mira
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(100)  
F_MIN = 150000  # 150 kHz
F_MAX = 250000  # 250 kHz
BIN_SIZE = 1000  # 1 Hz
NUM_BINS = int((F_MAX - F_MIN) / BIN_SIZE)
FREQS = np.linspace(F_MIN, F_MAX, NUM_BINS)


def generate_spectrum(frequencies, num_peaks=3):
    """
    Generates synthetic frequency response data with random peaks.
    :param frequencies: Array of frequency bins to evaluate.
    :param num_peaks: Number of random peaks to generate.
    :return: Tuple of (Magnitude, Phase) arrays.
    """
    N = len(frequencies)
    magnitude = np.zeros(N)
    phase = np.zeros(N)

    # Randomly generate peak parameters
    for _ in range(num_peaks):
        f_c = np.random.uniform(F_MIN, F_MAX)      # random center frequency
        A = np.random.uniform(1.0, 3.0)           # random amplitude
        sigma = np.random.uniform(1000, 10000)    # random width
        phi_offset = np.random.uniform(-np.pi, np.pi)  # random phase offset

        # Magnitude
        magnitude += A * np.exp(-0.5 * ((frequencies - f_c)/sigma)**2)

        # Phase
        phase += phi_offset + np.arctan(5 * (frequencies - f_c)/ sigma)

    # Add baseline and noise
    magnitude += 5.0 + np.random.normal(0, 0.1, N)  # baseline + white noise
    phase = (phase + np.random.normal(0, 0.01, N)) % (2 * np.pi)  # wrapped phase

    return magnitude, phase

mag, phase = generate_spectrum(FREQS)

mag_9bit = ((mag - mag.min()) / (mag.max() - mag.min()) * 511).astype(int)

Trainingvectors = 5000

Training_matrix = np.zeros((NUM_BINS, Trainingvectors), dtype=int)

for i in range(Trainingvectors):
    mag, _ = generate_spectrum(FREQS)
    mag_9bit = ((mag - mag.min()) / (mag.max() - mag.min()) * 511).astype(int)
    Training_matrix[:, i] = mag_9bit

#Plot
mag, _ = generate_spectrum(FREQS)

plt.figure(figsize=(12, 5))
plt.plot(FREQS, mag, color='blue')
plt.title("Magnitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()
