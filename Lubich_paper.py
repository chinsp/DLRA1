import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg

# Creating the supplementary matrices required in the paper

n = 100  # size of matrices

n_skew = int(n * (n - 1) / 2)  # dimension of the skew symmwtric matrix Ti
np.random.seed(123)
T1_random = np.random.random((n_skew))
T2_random = np.random.random((n_skew))

T1 = np.zeros((n, n))
T2 = np.zeros((n, n))
k = 0
for i in range(0, n):
    for j in range(i + 1, n):
        T1[i, j] = T1_random[k]
        T1[j, i] = -T1_random[k]
        k = k + 1

k = 0
for i in range(0, n):
    for j in range(i + 1, n):
        T2[i, j] = T2_random[k]
        T2[j, i] = -T2_random[k]
        k = k + 1

# Creating matrices A1 and A2

A1_1 = np.random.uniform(0, 0.5, (10, 10))
A2_1 = np.random.uniform(0, 0.5, (10, 10))

A1_2 = np.zeros((n, n))
A2_2 = np.zeros((n, n))

A1_2[:10, :10] = A1_1
A2_2[:10, :10] = A2_1

epsilon = 0.01

E1 = np.random.uniform(0, epsilon, (n, n))
E2 = np.random.uniform(0, epsilon, (n, n))

A1 = A1_2 + E1
A2 = A2_2 + E2

Q0 = np.identity(n)


def Q1(t):
    return np.exp(t * T1).dot(Q0)


def Q2(t):
    return np.exp(t * T2).dot(Q0)


def A(t, Q10, Q20):
    A_mat = Q1(t).dot(A1 + np.exp(t) * A2).dot(Q2(t))
    return A_mat


def K_step(t, K0, V0, dA):
    K_new = K0 + dA(t).dot(V0)
    return K_new


def S_step(t, U0, S0, V0, dA):
    S_new = S0 - U0.T.dot(dA(t)).dot(V0)
    return S_new


def L_step(t, L0, U0, dA):
    L_new = L0.T + dA(t).T.dot(U0)
    return L_new
