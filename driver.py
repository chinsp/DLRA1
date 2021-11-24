import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, qr
from Lubich_paper import K_step, S_step, L_step, A

# First order Lie-Trotter splitting.
# We project the RHS of the problem onto a low rank manifold.
# We Have the following split: P(f) g = P_{\bar{V}} g - P_{\bar{V}} P_{\bar{X}} g + P_{\bar{X}} g


def KSL_Lie(t, X0, S0, V0):
    #Computing time increment
    dA = A(t) - A(t-1)

    # K-step
    K0 = X0.dot(S0)
    K1 = K_step(K0, V0, dA)
    # QR decomposition
    X1, S_1 = qr(K1)

    # S-step
    S_2 = S_step(X1, S_1, V0, dA)

    # L-step
    L0 = S_2.dot(V0.T)
    L1 = L_step(L0, X1, dA)
    # QR decomposition
    V1, S1 = qr(L1)

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




