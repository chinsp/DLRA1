import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, qr

# First order Lie-Trotter splitting.
# We project the RHS of the problem onto a low rank manifold.
# We Have the following split: P(f) g = P_{\bar{V}} g - P_{\bar{V}} P_{\bar{X}} g + P_{\bar{X}} g

# time integration set-up
t0 = 0
t_end = 1
N_t = 100
time_steps = np.linspace(t0, t_end, N_t)

# Getting the initial conditons for the K-,S- and L- steps from the initial conditions of the PDE
X0, s0, v0 = svd(f_init_mat)
V0 = np.transpose(v0)
S0 = np.diag(s0)


def KSL_Lie(X0, S0, V0):
    # K-step
    K0 = X0.dot(S0)
    K1 = K_step(K0, V0)
    # QR decomposition
    X1, S_1 = qr(K1)

    # S-step
    S_2 = S_step(X1, S_1, V0)

    # L-step
    L0 = S_2.dot(V0.T)
    L1 = L_step(X1, L0)
    # QR decomposition
    V1, S1 = qr(L1.T)

    return X1, S1.T, V1



def LSK_Lie(X0, S0, V0):
    # L-step
    L0 = S0.dot(V0.T)
    L1 = L_step(X0, L0)
    # QR decomposition
    V1, S_1 = qr(L1.T)

    S_1 = S_1.T

    # S-step
    S_2 = S_step(X0, S_1, V1)

    # K-step
    K0 = X0.dot(S_2)
    K1 = K_step(K0, V1)
    # QR decomposition
    X1, S1 = qr(K1)

    return X1, S1, V1




