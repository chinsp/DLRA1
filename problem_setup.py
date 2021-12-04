import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import svd
import driver
import Lubich_paper
import pandas as pd

# setting a rank for the driver
rank_list = [5, 10, 20, 30]


# Setting the initial conditions for the K-,S- and L- steps from the initial conditions of the PDE
# f_init_mat = Lubich_paper.A(t0)

def init_cond(f_init_mat, r):
    x0, s0, v0 = svd(f_init_mat)
    v0 = v0.T
    X0 = x0[:, :r]
    V0 = v0[:, :r]
    S0 = np.diag(s0[:r])
    return X0, S0, V0


def DLRA(problem, r, t0=0, t_end=1, dt=0.01, method='KSL', order=1, compute_error='False', error_plot='False'):

    # setting the time steps
    N_t = int((t_end - t0) / dt)
    time_steps = np.linspace(t0, t_end, N_t)

    # Setting the initial conditions
    X0, S0, V0 = init_cond(problem.A(t0), r)

    # set-up for error computation
    E = np.zeros(len(time_steps) - 1)

    if order == 1:
        if method == 'KSL':
            # Time integration
            for j in range(len(time_steps) - 1):
                t = time_steps[j + 1]
                X1, S1, V1 = driver.KSL_Lie(t, dt, X0, S0, V0)
                # A1 = X1.dot(S1).dot(V1.T)
                # A1_hat = X1.dot(S1).dot(V1.T)
                X0, S0, V0 = X1, S1, V1

                if compute_error == 'True' or error_plot == 'True':
                    E[j] = error(problem.A, t, X1.dot(S1).dot(V1.T))

        elif method == 'LSK':
            # Time integration
            for j in range(len(time_steps) - 1):
                t = time_steps[j + 1]
                X1, S1, V1 = driver.LSK_Lie(t, dt, X0, S0, V0)
                # A1 = X1.dot(S1).dot(V1.T)
                # A1_hat = X1.dot(S1).dot(V1.T)
                X0, S0, V0 = X1, S1, V1

                if compute_error == 'True' or error_plot == 'True':
                    E[j] = error(problem.A, t, X1.dot(S1).dot(V1.T))

        elif method == 'unconv':
            return print('Method not coded yet')
        else:
            return print('Desired method is not available yet')

    elif order == 2:
        if method == 'KSL':
            # Time integration
            for j in range(len(time_steps) - 1):
                t = time_steps[j + 1]
                X1, S1, V1 = driver.KSL_Strang(t, dt, X0, S0, V0)
                # A1 = X1.dot(S1).dot(V1.T)
                # A1_hat = X1.dot(S1).dot(V1.T)
                X0, S0, V0 = X1, S1, V1

                if compute_error == 'True' or error_plot == 'True':
                    E[j] = error(problem.A, t, X1.dot(S1).dot(V1.T))

        elif method == 'LSK':
            return print('Method not very effective, use another method for this order e.g. KSL')
        elif method == 'unconv':
            return print('Method not available yet')

    if error_plot == 'True':
        plt.plot(time_steps[1:], E)
        plt.ylabel('Approximation Error')
        plt.xlabel('time')
        plt.show()

    return X1, S1, V1, E


def error(A, t, A1_hat):
    # local error
    e = np.linalg.norm(A1_hat - A(t))
    return e
