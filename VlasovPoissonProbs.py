# Defining the Vlasov-Poisson problem for flow of particles with their corresponding K-step, S-step and L-step

import scipy.integrate as intg

def Energy(f):
    return f(t,x,v)

def K_step(K, V):
    C1 = dy * (V.T).dot(diag_v).dot(V)
    C2 = (V.T).dot(Dy).dot(V)
    intg.RK45(-Dx.dot(K).dot(C1.T) + E0.dot(K).dot(C2.T))
    return 0.0


def S_step(X, S, V):
    return 0.0


def L_step(X, L):
    return 0.0
