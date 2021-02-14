

import torch

import numpy as np
from pystoi.stoi import stoi

# Please see this https://github.com/schmiph2/pysepm
### PESQ : not able to install in windows
"""
from pesq import pesq
def compute_PESQ(clean_signal, noisy_signal, sr=16000):
    return pesq(sr, clean_signal, noisy_signal, "wb")
"""

def z_score(m):
    mean = np.mean(m)
    std_var = np.std(m)
    return (m - mean) / std_var, mean, std_var


def reverse_z_score(m, mean, std_var):
    return m * std_var + mean


def min_max(m):
    m_max = np.max(m)
    m_min = np.min(m)

    return (m - m_min) / (m_max - m_min), m_max, m_min


def reverse_min_max(m, m_max, m_min):
    return m * (m_max - m_min) + m_min

def compute_STOI(clean_signal, noisy_signal, sr=16000):
    return stoi(clean_signal, noisy_signal, sr, extended=False)