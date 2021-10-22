import numpy as np
import matplotlib.pyplot as plt

# Creating the supplementary matrices required in the paper

n = 100 # size of matrices

n_skew = int(n * (n - 1)/2) # dimension of the skew symmwtric matrix Ti
np.random.seed(123)
T1_random = np.random.random((n_skew))
T2_random = np.random.random((n_skew))

T1 = np.zeros((n,n))
T2 = np.zeros((n,n))
k = 0
for i in range(0,n):
    for j in range(i+1,n):
        T1[i,j] = T1_random[k]
        T1[j,i] = -T1_random[k]
        k = k + 1

k = 0
for i in range(0,n):
    for j in range(i+1,n):
        T2[i,j] = T2_random[k]
        T2[j,i] = -T2_random[k]
        k = k + 1

# Creating matrices A1 and A2

A1_1 = np.random.uniform(0,0.5,(10,10))
A2_1 = np.random.uniform(0,0.5,(10,10))

A1_2 = np.zeros((n,n))
A2_2 = np.zeros((n,n))

A1_2[:10,:10] = A1_1
A2_2[:10,:10] = A2_1

epsilon = 0.01

E1 = np.random.uniform(0,epsilon, (n,n))
E2 = np.random.uniform(0,epsilon, (n,n))

A1 = A1_2 + E1
A2 = A2_2 + E2

