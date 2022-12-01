"""
---
created: 2022-11-29-15-20-00
---

# Hardening laws V0R0
tags: #computational #constitutive #numeric

# Progress
- This is a collection of hardening laws to be 
    used with the Material Class of FEM1elem


# log
- 29/11/22
    - created 
"""

import math
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt


def Bezier_pc_T_hardening(T ,T_params):        
    (pc0, T0, angle_at_T0, pcTmax, Tmax, angle_at_Tmax, pc_intersection_at_Tc, Tc, angle_at_Tc) = T_params
      
    scale_factor = 1e6
    pc0 = pc0/scale_factor
    pcTmax = pcTmax/scale_factor
    pc_intersection_at_Tc = pc_intersection_at_Tc/scale_factor    
    
    # convert angles into angular coefficient
    slope_at_T0 = math.tan(angle_at_T0)
    slope_at_Tc = math.tan(angle_at_Tc)
    slope_at_Tmax = math.tan(angle_at_Tmax)
    
    P0 = np.array([T0, pc0])
       
    P1x = (pc0 - pc_intersection_at_Tc + slope_at_Tc*Tc - slope_at_T0*T0)/(slope_at_Tc - slope_at_T0)
    P1y = pc0 + slope_at_T0*(P1x - T0)
    
    P1 = np.array([ P1x, P1y])
    
    P2 = np.array([Tc, pc_intersection_at_Tc])
    
    P3x = (pc_intersection_at_Tc - pcTmax + slope_at_Tmax*Tmax - slope_at_Tc*Tc)/(slope_at_Tmax - slope_at_Tc)
    P3y = pc_intersection_at_Tc + slope_at_Tc*(P3x - Tc)
    P3 = np.array([ P3x, P3y])
    
    P4 = np.array([Tmax, pcTmax])
    
    # define first bezier curve
    B1 = lambda t: (1 - t)**2 * P0 + 2 * t * (1 - t) * P1 + t**2 * P2
    
    # repeat operations on the second bezier curve
    B2 = lambda t: (1 - t)**2 * P2 + 2 * t * (1 - t) * P3 + t**2 * P4
    
    # Find t corresponding to given x
    if T == P2[0]:
        pc_T = P2[1]
    else:
        if T < P2[0] :
            P0x = P0[0]
            P1x = P1[0]
            P2x = P2[0]
        elif T > P2[0] and T < P4[0] :
            P0x = P2[0]
            P1x = P3[0]
            P2x = P4[0]
            
        a = P0x - 2 * P1x + P2x
        b = - 2 * P0x + 2 * P1x
        c = P0x - T
        
        if a != 0:
            t_solution = (- b + (b**2-4*a*c)**.5)/(2*a)
            
            if t_solution > 0 and t_solution < 1:
                t_of_T = t_solution
            else:
                t_of_T = (- b -(b**2-4*a*c)**.5)/(2*a)    
        else:
            t_of_T = (T - P0x)/(- 2 * P0x + 2 * P1x)
        
        
        if T < P2[0] :
            pc_T = B1(t_of_T)[1]
        elif T > P2[0] and T < P4[0] :
            pc_T = B2(t_of_T)[1]
    
    return pc_T*scale_factor



def modCamClay_hardening(alp, T, params):
    # unpack params tuple
    T_params = params[3:]
    pcT = Bezier_pc_T_hardening(T, T_params)
    (h, omega, M) = params[:3]
       
    pc = pcT + h*alp
    c = omega*pc
    A = jnp.array([M, pc, c])
    return A


def BPsquared_hardening(alp, T, params):
    T_params = params[7:]
    pcT = Bezier_pc_T_hardening(T, T_params)
    
    (h, omega, M, alpha, m, beta, gamma) = params[:7]
    pc = pcT + h*alp
    c = omega*pc
    A = jnp.array([M, pc, c, alpha, m, beta, gamma])
    return A

def BPsquared_hardening_Penasa_2017(alp, T, params):
    (omega, M, alpha, m, beta, gamma, Apc, Bpc, Cpc, T0pc, Ak1, Bk1, Ck1, T0k1, delta0, T0delta, TF) = params
    
    pcT = Apc - Bpc*jnp.tanh((T - T0pc)/Cpc)
    
    k1T = Ak1 - Bk1*jnp.tanh((T - T0k1)/Ck1)
    
    deltaT = delta0 * jnp.exp(- (T-T0delta)/(T-TF) )
    
    pc = pcT  + k1T*alp/(1+deltaT*alp)
    
    c = omega*pc
    A = jnp.array([M, pc, c, alpha, m, beta, gamma])
    return A


# # simple tests on the hardening laws
if __name__ == "__main__":
    # (M, pc, c, alpha, m, beta, gamma)
    A = (1, 250e6, .2*250e6)
    
    M, pc, c = A

    Tparams = (80e6, 20, 0, 60e6, 1200, 0, 70e6, 600, 0.98*3.14)

    N_steps = 20
    T_step = np.linspace(21,1199,num=N_steps)
    
    pcT = np.zeros(N_steps)
    for j in range(0,N_steps):
        val = modCamClay_hardening(0, T_step[j],Tparams)
        pcT[j] =  val
        print("T:", T_step[j], "\t", "pc(T):", val)
    
    plt.plot(T_step, pcT,'-bo')
    plt.grid()
    plt.show()

























