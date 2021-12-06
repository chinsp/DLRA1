# Defining the Vlasov-Poisson problem for flow of particles with their corresponding K-step, S-step and L-step
import numpy as np
import scipy.integrate as intg

# set velocity domain
va, vb = 0, 1
dv = 10e-2
v_dom = np.linspace(va, vb, int((vb - va) / dv))
diag_v = np.diag(v_dom)
nv = len(v_dom)

# set spatial domain
xa, xb = 0, 1
dx = 10e-2
x_dom = np.linspace(xa, xb, int((xb - xa) / dx))
nx = len(x_dom)

# setting matrices for approximating derivatives

# first derivatives
Dv = np.zeros((nv, nv))
Dv = Dv - np.diag(np.ones(nv - 1), -1) + np.diag(np.ones(nv - 1), 1)
Dv = Dv / 2 / dv
Dx = np.zeros((nx, nx))
Dx = Dx - np.diag(np.ones(nx - 1), -1) + np.diag(np.ones(nx - 1), 1)
Dx = Dx / 2 / dx

# second derivatives
Dv2 = np.zeros((nv, nv))
Dv2 = Dv2 + 2 * np.diag(np.ones(nv)) - np.diag(np.ones(nv - 1), -1) - np.diag(np.ones(nv - 1), 1)
Dv2 = Dv2 / dv ** 2
Dx2 = np.zeros((nx, nx))
Dx2 = Dx2 + 2 * np.diag(np.ones(nv)) - np.diag(np.ones(nv - 1), -1) - np.diag(np.ones(nv - 1), 1)
Dx2 = Dx2 / dx ** 2

# Defining initial conditions
k = 0.1
alpha = 0.01


def f_init(x, v):
    return (1 + alpha * np.cos(k * x)) * np.exp(-v ** 2 / 2) / (2 * np.pi) ** 0.5


def compute_Energy(K, V):
    rho_V = np.sum(V, axis=0)
    E = Dx.dot(np.linalg.inv(Dx2)).dot(np.identity(nx) - K.dot(rho_V))
    return E


def K_step(K, V, dt):
    E = compute_Energy(K, V)  # Encode the energy properly.
    C1 = dv * V.T.dot(diag_v).dot(V)
    C2 = V.T.dot(Dv).dot(V)
    # K_new = intg.RK45(lambda K_var: -Dx.dot(K_var).dot(C1.T) + E.dot(K_var).dot(C2.T), 0, K, 0.01)
    K_new = K + dt * (-Dx.dot(K).dot(C1.T) + E.dot(K).dot(C2.T))
    return K_new


def S_step(X, S, V):
    return 0.0


def L_step(X, L):
    return 0.0


F0 = np.empty((nx, nv))
for i in range(nx):
    for j in range(nv):
        F0[i, j] = f_init(x_dom[i], v_dom[j])

r = 10
x0, s0, v0 = np.linalg.svd(F0)
v0 = v0.T
X0 = x0[:, :r]
V0 = v0[:, :r]
S0 = np.diag(s0[:r])

K0 = X0.dot(S0)
K_new = K_step(K0, V0, 0.01)
print(K_new)
