"""
---
created: 2022-11-29-15-20-00
---

# Yield functions V1R0
tags: tags: #computational #constitutive #numeric

# Progress
- This is a collection of yielding functions to be 
    used with the Material Class of FEM1elem
- Fixed some implemetation throuth the test in Voigt_stress_test_221125_2

# log
- 29/11/22
    - created as brach of V0R1
"""

import jax.numpy as jnp
import jax.lax as jlx
from jax import grad
import numpy as np
from matplotlib import pyplot as plt


def VonMises(sig, A, params):
    # Von Mises yield function
    I2 = jnp.eye(3)
    dev = sig - 1/3*jnp.trace(sig)*I2
    J2 = 0.5*jnp.tensordot(dev, dev, axes=2)
    F = (3*J2)**.5 - A
    return F


def modCamClay(sig, A, params):
    (M, pc, c) = A
    # HW stress space equivalent coordinates p and q
    invp = - 1/3 * jnp.trace(sig)
    I2 = jnp.eye(3)
    dev = sig + invp*I2
    J2 = 0.5*jnp.trace(dev**2)
    invq = (3*J2)**0.5
    F = invq**2 + M**2 * (invp + c) * (invp - pc)
    return F/pc


def BPsquared(sig, A, params):
    # unpack params tuple
    (M, pc, c, alpha, m, beta, gamma) = A
    # css = coordinate stress space
    invp = - 1/3 * jnp.trace(sig)
    I2 = jnp.eye(3)
    dev = sig + invp*I2
    J2 = 0.5*jnp.trace(dev**2) + 1
    J3 = 1/3*jnp.trace(dev**3)
    invq = (3*J2)**0.5
    
    # theta = 1/3 * jlx.acos(0.001*jlx.floor(3*3**0.5/2 * J3/J2**(3/2)*1000))     
    
    cos3theta = 3*3**0.5/2*J3*J2**(-3/2)
    
    Phi = (invp+c)/(pc+c)
    finvpsqr = M**2*pc**2*(abs(Phi)**(m-1)*Phi-Phi)*(2*(1-alpha)*Phi+alpha)
    gtheta = 1/jlx.cos(beta * 3.14/6 - 1/3*(jlx.acos(gamma*cos3theta)))
    F = finvpsqr + invq**2/gtheta**2
    F = F/pc**2
    
    return F    


# # simple tests on the yield functions
if __name__ == "__main__":
    # (M, pc, c, alpha, m, beta, gamma)
    A = (0.773, 200e6, 0.012*200e6, 0.264, 2.125, 0.75, 0.7)
    
    sig = np.zeros((3,3))
    sig[0,0] = 1e6
    
    # sig = np.array([[0, 250e6, 280e6],
    #                 [250e6, 0, 0],
    #                 [280e6, 0, 0]])
    
    # sig = np.zeros((3,3))
    # hydro = -200e6
    # sig[0,0] = hydro
    # sig[1,1] = hydro
    # sig[2,2] = hydro
    
    F = BPsquared(sig, A, 0)
    
    print("BPsquared: %.2e" %F)
    
    dFdsig = grad(BPsquared,0)(sig, A, 0)
    print(dFdsig)
    
    (M, pc, c, alpha, m, beta, gamma) = A
    delta = pc/100
    
    xrange = np.arange(-1.5*c, 1.5*pc, delta)
    yrange = np.arange(-1.5*M*pc/2, 1.5*M*pc/2, delta)
    invp, invq = np.meshgrid(xrange,yrange)
    
    # calculation of BPsquared
    cos3theta = 1
    Phi = (invp+c)/(pc+c)
    finvpsqr = M**2*pc**2*(abs(Phi)**(m-1)*Phi-Phi)*(2*(1-alpha)*Phi+alpha)
    gtheta = 1/jlx.cos(beta * 3.14/6 - 1/3*(jlx.acos(gamma*cos3theta)))
    F = finvpsqr + invq**2/gtheta**2
    F = F/pc**2
    
    CS = plt.contour(invp, invq, F , levels=50)
    plt.xlabel("p [Pa]")
    plt.ylabel("q [Pa]")
    plt.clabel(CS, inline=1, fontsize=10)
    plt.grid()
    plt.title("BPsquared")
    plt.show()

    

































