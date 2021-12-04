import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as intg
import scipy.linalg as sclin

# Creating the supplementary matrices required in the paper

n = 100  # size of matrices

n_skew = int(n * (n - 1) / 2)  # dimension of the skew symmetric matrix Ti
np.random.seed(123)
T1_random = np.random.random(n_skew)
T2_random = np.random.random(n_skew)

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

epsilon = 10e-6

E1 = np.random.uniform(0, epsilon, (n, n))
E2 = np.random.uniform(0, epsilon, (n, n))

A1 = A1_2 + E1
A2 = A2_2 + E2

Q0 = np.identity(n)


def Q1(t):
    return sclin.expm(t * T1).dot(Q0)


def Q2(t):
    return sclin.expm(t * T2).dot(Q0)


def A(t):
    A_mat = Q1(t).dot(A1 + np.exp(t) * A2).dot(Q2(t))
    return A_mat


def K_step(K, V, dA):
    K_new = K + dA.dot(V)
    return K_new


def S_step(X, S, V, dA):
    S_new = S - X.T.dot(dA).dot(V)
    return S_new


def L_step(L, X, dA):
    L_new = L.T + dA.T.dot(X)
    return L_new
