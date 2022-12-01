"""
---
created: 2022-11-21-15-47-00
---

# Elastic potentials V1R0
tags: tags: #computational #constitutive #numeric

# Progress
- This is a collection of elastic potential functions to be 
    used with the Material Class of FEM1elem

# ToDo
- [ ] implementare temperature dependance

# log
- 21/11/22
    - creato
"""    

# from datetime import datetime
import numpy as np
import jax.numpy as jnp
from jax import grad
import matplotlib.pyplot as plt


def ET(T):
    if T <= 20:
        E = 10.5e9
    elif T>20 and T<=200:
        E = 10.5e9 - 1.4e6*T
    elif T>200 and T<=400:
        E = 9.58e9 - 3.41e6*T
    elif T>400 and T<=600:
        E = 24.4e9 - 33.7e6*T
    elif T>600 and T<=800:
        E = 5.25e9 - 1.76e6*T
    elif T>800 and T<=1000:
        E = 6.61e9 - 3.45e6*T
    elif T>1000 and T<=1200:
        E = 6.06e9 - 2.89e6*T
    elif T>1200 and T<=1400:
        E = 10.2e9 - 6.41e6*T
    else:
        E = 1.3e9    
    return E

def nuT(nu0, T):
    return nu0

def linear_elasticity(eps, T, params):
    E0, nu0 = params[0], params[1]
    
    E = ET(T)
    nu = nuT(nu0, T)
    
    # Lamé constants
    lam = E*nu/((1+nu)*(1-2*nu))
    # mu is also known as Shear Modulus G
    mu = E/(2*(1+nu))

    # second order identity tensor
    I2 = np.eye(3)

    # fourth order identity tensor aka symmetrizer
    def kron(i,j):
        if i==j:
            return 1
        else:
            return 0
    I4 = np.zeros((3,3,3,3))
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):
                    I4[i,j,k,l] = 0.5*(kron(i,k)*kron(j,l)+kron(i,l)*kron(j,k))

    # 4th order Elastic tensor
    C = lam*np.tensordot(I2, I2, axes = 0) + 2*mu*I4

    return 0.5*jnp.tensordot(eps, jnp.tensordot(C, eps, axes=2) , axes=2)


# # Testing the models
if __name__ == "__main__":
    W = linear_elasticity
    
    # parameters definition
    params = (210e9, 0.3)
    
    # test of the elastic potential function and plot
    eps_f = 0.004
    Nsteps = 10
    Deps = eps_f/Nsteps
    
    eps = np.zeros((Nsteps,3,3))
    sig = np.zeros((Nsteps,3,3))
    
    for n in range(0, Nsteps):
        eps[n,0,0] = eps[n-1,0,0] + Deps 
        sig[n,:,:] = grad(W,0)(eps[n,:,:], 20, params)
        
    S11 = sig[:,0,0]
    E11 = eps[:,0,0]
    plt.figure(1)
    plt.plot(E11*1e2,S11/1e6,'-bo')
    plt.xlabel("E11 [%]")
    plt.ylabel("S11 [MPa]")
    plt.grid()
    plt.title("S11-E11")
    plt.show()
    
    T0 = 21
    Tmax = 1200
    T = np.linspace(T0, Tmax, num=10)
    E = [ET(t)/1e9 for t in T]
    plt.figure(2)
    plt.plot(T,E,'-bo')
    plt.xlabel("T [°C]")
    plt.ylabel("E [GPa]")
    plt.grid()
    plt.show()

    