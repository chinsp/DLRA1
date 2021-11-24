import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
import tests
import driver
import Lubich_paper

# setting a rank for the driver
rank_list = [10, 20]
r = rank_list[0]

# time integration set-up
t0 = 0
t_end = 1
N_t = 10
time_steps = np.linspace(t0, t_end, N_t)

# Getting the initial conditons for the K-,S- and L- steps from the initial conditions of the PDE
f_init_mat = Lubich_paper.A(t0)
x0, s0, v0 = svd(f_init_mat)
X0 = x0[:, :r]
V0 = v0[:, :r]
S0 = np.diag(s0)[:r, :r]


# Error analysis
E = np.zeros((len(time_steps) -1, len(rank_list)))
k,l = 0,0

for t in time_steps[1:]:
    X1, S1, V1 = driver.KSL_Lie(t, X0, S0, V0)
    # A1 = X1.dot(S1).dot(V1.T)
    A1_hat = X1.dot(S1).dot(V1.T)
    X0, S0, V0 = X1, S1, V1

    # Error analysis

    # Best approximation
    X1_t, S1_t, V1_t = np.linalg.svd(Lubich_paper.A(t))
    S1_t = np.diag(S1_t)
    X1_best, S1_best, V1_best = X1_t[:, :r], S1_t[:r, :r], V1_t[:, :r]
    A1_best = X1_best.dot(S1_best).dot(V1_best.T)
    # print(np.linalg.matrix_rank(A1_hat))

    # local error
    e = np.linalg.norm(A1_hat - A1_best)
    print(e)
    E[k, l] = e
    k = k + 1


plt.plot(time_steps[1:], E[:,0])
plt.show()