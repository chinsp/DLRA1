import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd, qr
from Lubich_paper import K_step, S_step, L_step, A


# We project the RHS of the problem onto a low rank manifold.
# We Have the following split: P(f) g = P_{\bar{V}} g - P_{\bar{V}} P_{\bar{X}} g + P_{\bar{X}} g

# First order Lie-Trotter splitting.


def KSL_Lie(t, dt, X0, S0, V0):
    # Computing time increment
    dA = A(t) - A(t - dt)

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


def LSK_Lie(t, dt, X0, S0, V0):
    # Computing time increment
    dA = A(t) - A(t - dt)

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


# Second order Strang splitting scheme

def KSL_Strang(t, dt, X0, S0, V0):
    # Defining increment of A for time t0 ,t_1/2 and t1
    A0, A_mid, A1 = A(t - dt), A(t - (dt/2)), A(t)
    dA_mid1, dA_mid2 = A_mid - A0, A1 - A_mid

    # half K-step t0 -> t_1/2
    K0 = X0.dot(S0)
    K_mid = K_step(K0, V0, dA_mid1)
    # QR decomposition
    X_mid1, S_mid1 = qr(K_mid)

    # half S-step t0 -> t_1/2
    S_mid2 = S_step(X_mid1, S_mid1, V0, dA_mid1)

    # full L-step t0 -> t1
    L0 = S_mid2.dot(V0.T)
    L1 = L_step(L0, X_mid1, dA_mid2 + dA_mid1)
    # QR decomposition
    V1, S_mid3 = qr(L1)
    S_mid3 = S_mid3.T

    # half S-step t_1/2 -> t1
    S_mid4 = S_step(X_mid1, S_mid3, V1, dA_mid2)

    # half K-step t_1/2 -> t1
    K0 = X_mid1.dot(S_mid4)
    K1 = K_step(K0, V1, dA_mid2)
    # QR decomposition
    X1, S1 = qr(K_mid)

    return X1, S1, V1
