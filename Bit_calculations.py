# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:52:34 2026

@author: Mira
"""
import numpy as np

def X_bits(X):
    bits=9*X.shape[1]*X.shape[0]
    return bits
   

def C_9bit(C):
    C_min = C.min()
    C_max = C.max()
    if C_max == C_min:
        C_quantized = np.zeros_like(C, dtype=int)
    else:
        C_quantized = ((C - C_min) / (C_max - C_min) * 511).astype(int)
    bits_C = C_quantized.size * 9
    return bits_C


def indencies_bits(D,K):
    Encoder_gamma = np.argmin(D, axis=0)
    bits_indices = Encoder_gamma.size * 9
    return bits_indices
    
