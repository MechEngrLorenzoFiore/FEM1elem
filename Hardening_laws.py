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
import matplotlib

# ----------------------------------------------------------------
# Evolution laws of parameters with temperature
# ----------------------------------------------------------------

def Bezier_pc_T_hardening(T ,T_params):        
    # WARNING! This function suffers of a bug in the definition, if you ack for a 
    # temperature on the definition extreme it will fail!!!
    
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


def Penasa_2017_pc_T_hardening(T, T_params):
    
    (Apc, Bpc, Cpc, T0pc) = T_params
    
    pcT = Apc - Bpc*jnp.tanh((T - T0pc)/Cpc)
    
    return pcT

# ----------------------------------------------------------------
# Evolution laws of parameters with cumulated plastic strain
# ----------------------------------------------------------------

def modCamClay_hardening(alp, T, params):
    # unpack params tuple
    T_params = params[3:]
    pcT = Bezier_pc_T_hardening(T, T_params)
    (h, omega, M) = params[:3]
       
    pc = pcT + h*alp
    c = omega*pc
    A = jnp.array([M, pc, c])
    return A

def BPsquared_hardening_Bezier_T_linear_alp(alp, T, params):
    T_params = params[7:]
    pcT = Bezier_pc_T_hardening(T, T_params)
    
    (h, omega, M, alpha, m, beta, gamma) = params[:7]
    pc = pcT + h*alp
    c = omega*pc
    A = jnp.array([M, pc, c, alpha, m, beta, gamma])
    return A

def BPsquared_hardening_Bezier_T_Penasa_2017_alp(alp, T, params):
    (omega, M, alpha, m, beta, gamma,
     Ak1, Bk1, Ck1, T0k1, delta0, T0delta, TF,
     pc0, T0, angle_at_T0, 
     pcTmax, Tmax, angle_at_Tmax, 
     pc_intersection_at_Tc, Tc, angle_at_Tc) = params
    
    T_params = (pc0, T0, angle_at_T0, pcTmax, Tmax, angle_at_Tmax, pc_intersection_at_Tc, Tc, angle_at_Tc)

    pcT = Bezier_pc_T_hardening(T, T_params)
    
    k1T = Ak1 - Bk1*jnp.tanh((T - T0k1)/Ck1)
    
    deltaT = delta0 * jnp.exp(- (T-T0delta)/(T-TF) )
    
    pc = pcT  + k1T*alp/(1+deltaT*alp)
    
    c = omega*pc
    A = jnp.array([M, pc, c, alpha, m, beta, gamma])
    return A

def BPsquared_hardening_Penasa_2017(alp, T, params):
    (omega, M, alpha, m, beta, gamma, 
     Apc, Bpc, Cpc, T0pc, 
     Ak1, Bk1, Ck1, T0k1, delta0, T0delta, TF) = params
    
    T_params = (Apc, Bpc, Cpc, T0pc)
    
    pcT = Penasa_2017_pc_T_hardening(T, T_params)
    # pcT = Apc - Bpc*jnp.tanh((T - T0pc)/Cpc)
    
    k1T = Ak1 - Bk1*jnp.tanh((T - T0k1)/Ck1)
    
    deltaT = delta0 * jnp.exp(- (T-T0delta)/(T-TF) )
    
    pc = pcT  + k1T*alp/(1+deltaT*alp)
    
    c = omega*pc
    A = jnp.array([M, pc, c, alpha, m, beta, gamma])
    return A


# # simple tests on the hardening laws
if __name__ == "__main__":
    
    test = "pcT"
    # test = "Aalp"
    
    if test == "pcT":
                               # (Apc, Bpc, Cpc, T0pc)
        T_params_Penasa_2017 = (158e6, 43.5e6, 200, 651)
    
                         # (pc0, T0, angle_at_T0, 
                         #  pcTmax, Tmax, angle_at_Tmax, 
                         #  pc_intersection_at_Tc, Tc, angle_at_Tc)
        T_params_Bezier = (201340528, 20, 0,
                           114861232, 1200, 0,
                           160173184, 641,  0.89*3.14)
    
        N_steps = 20
        T_step = np.linspace(21,1199,num=N_steps)
        
        pcT = np.zeros((N_steps, 2))
        
        print(" \t\t T \t\t pc(T) Penasa \t\t pc(T) Bezier")
        print(20*3*"-")
        for i in range(0,N_steps):
            pcT[i,0] = Penasa_2017_pc_T_hardening(T_step[i], T_params_Penasa_2017)
            pcT[i,1] = Bezier_pc_T_hardening(T_step[i], T_params_Bezier)
            print("%10d \t\t %10.2f \t\t %10.2f" %(T_step[i], pcT[i,0], pcT[i,1]) )
        
        
        # plot settings
        figdpi = 600
        axlebelfont = 8 #[pt]
        axtickfont = axlebelfont - 1 #[pt]
        width = 12 #[cm]
        heigth = 9 #[cm]
        
        fig = plt.figure(figsize=(width/2.54, heigth/2.54), dpi=figdpi)
        
        matplotlib.rc('xtick', labelsize=axtickfont) 
        matplotlib.rc('ytick', labelsize=axtickfont) 

        plt.plot(T_step, pcT[:,0]/1e6,'-bo', label="Penasa_2017_pc_T_hardening")
        plt.plot(T_step, pcT[:,1]/1e6,'-ro', label="Bezier_pc_T_hardening")
        
        plt.xlabel("T [°C]")
        plt.ylabel("pc [MPa]")
        plt.legend( 
                   # title=model.replace(" ","\n"),
                   title_fontsize=axtickfont,
                   fontsize=axtickfont,
                   # markerscale=.8,
                   frameon=False,
                                # (x, y, width, height)
                   # bbox_to_anchor=(1, .5, 0.2, 0.2),
                   loc='best') 
        plt.grid()

        if False:        
            from datetime import datetime
            now = datetime.now()
            dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
            plt.savefig("pcT_confronto" + dt_string + ".png", dpi=figdpi, bbox_inches='tight')
        else:
            plt.show()

    elif test == "Aalp":

                            # (omega, M, alpha, m, beta, gamma, 
                            # Apc, Bpc, Cpc, T0pc,
                            # Ak1, Bk1, Ck1, T0k1, delta0, T0delta, TF) 
        params_Penasa_2017 = (0.012, 0.773, 0.264, 2.125, 0.75, 0.7,
                              158e6, 43.5e6, 200, 651,
                              119e9, 0, 1, 0, 922, 0, 0) 

                        # (omega, M, alpha, m, beta, gamma,
                        #  Ak1, Bk1, Ck1, T0k1, delta0, T0delta, TF,
                        #  pc0, T0, angle_at_T0, 
                        # pcTmax, Tmax, angle_at_Tmax, 
                        # pc_intersection_at_Tc, Tc, angle_at_Tc)
        params_Bezier = (0.012, 0.773, 0.264, 2.125, 0.75, 0.7,
                         119e9, 0, 1, 0, 922, 0, 0,
                         201340528, 20, 0,
                         114861232, 1200, 0,
                         160173184, 641,  0.89*3.14)
                         
        N_steps = 20
        alp_step = np.linspace(0,0.001,num=N_steps)
        T = 21
        
        for i in range(0,N_steps):
            A_Penasa = BPsquared_hardening_Penasa_2017(alp_step[i], T, params_Penasa_2017)
            A_Bezier = BPsquared_hardening_Bezier_T_Penasa_2017_alp(alp_step[i], T, params_Bezier)
            print("T = ", T, 80*"-")
            [print("%10.2g" %a, end="\t") for a in A_Penasa]
            print("")
            [print("%10.2g" %a, end="\t") for a in A_Bezier]
            print("")
                         
        
        




















