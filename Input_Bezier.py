"""

"""

# ----------------------------------------------------------------
# Import the necessary modules and track them
# ----------------------------------------------------------------
import os
import numpy as np
import jax.numpy as jnp
# Track the imported modules
modules1 = dir()

# import the user created modules from the respective files
from FEM1elem_engine import Material, Finite_Element, PDE_problem
from Elastic_Potentials import linear_elasticity
from Yield_functions import BPsquared
from Hardening_laws import BPsquared_hardening_Bezier_T_Penasa_2017_alp

# Track the imported modules
modules2 = dir()
# remove from imported modules the non relevant information 
rawimportedmod = [string for string in modules2 if string not in modules1]
importedmod = [string for string in rawimportedmod if string not in {"modules1", "Material", "Finite_Element", "PDE_problem"}]
# track the test script name
testscript = os.path.basename(__file__)


# ----------------------------------------------------------------
# Define the material through its parameters
# ----------------------------------------------------------------

# Material definition
Elas_params = (1e9, 0.07)
            # (omega, M, alpha, m, beta, gamma,
            #  Ak1, Bk1, Ck1, T0k1, delta0, T0delta, TF,
            #  pc0, T0, angle_at_T0, 
            # pcTmax, Tmax, angle_at_Tmax, 
            # pc_intersection_at_Tc, Tc, angle_at_Tc)
Hard_params = (0.012, 0.773, 0.264, 2.125, 0.75, 0.7,
                 119e9, 0, 1, 0, 922, 0, 0,
                 201340528, 20, 0,
                 114861232, 1200, 0,
                 160173184, 641,  0.89*3.14)
Yield_params = ()
params = (Elas_params, Hard_params, Yield_params)

# Material definition
Ceramic_1 = Material(params, linear_elasticity, BPsquared_hardening_Bezier_T_Penasa_2017_alp, BPsquared)

# ----------------------------------------------------------------
# Set the material points internal variables ICs
# ----------------------------------------------------------------

# Initial conditions definition
sig = jnp.zeros((3,3))
internal_vars_size = 3
A = jnp.zeros((1,internal_vars_size))[0]
eps = jnp.zeros((3,3))
eps_p = jnp.zeros((3,3))
alp = 0
gam = 0

# ----------------------------------------------------------------
# Define the analisys parameters
# ----------------------------------------------------------------

# prescribed displacement at selected dofs
displ_value = -0.012

# Step definition
N_steps = 30

# Definition of temperature for each step
T0 = 600
Tmax = 601
T_step = np.linspace(T0, Tmax, num=N_steps)

# ----------------------------------------------------------------
# Set the kind of test
# ----------------------------------------------------------------

# testID = "MPI"
testID = "1elem"

# ----------------------------------------------------------------
# Perform the selected test
# ----------------------------------------------------------------

if testID == "MPI":
    # Material Point Integration simulation
    # activate Voigt_stress DEBUG features
    Ceramic_1.DEBUG = True
    compsig = [np.zeros(N_steps)]
    STATEV_GP = (sig, A, eps, eps_p, alp, gam)
    E11 = np.zeros(N_steps+1)
    for n in range(0,N_steps):
        print("----- Step %d -----" %(n+1) )
        # eps is related to displ by a factor 1
        E11[n+1] = E11[n] + displ_value/N_steps
        eps_Voigt = jnp.array([E11[n+1], 0, 0, 0, 0, 0])
        val, STATEV_GP = Ceramic_1.Voigt_stress(eps_Voigt, STATEV_GP, T_step[n], 1)
        compsig.append( val )

    S11 = [float(compsig[n][0]) for n in range(0, N_steps+1)]

elif testID == "1elem":
    # one finite element simulation
    
    # ----------------------------------------------------------------
    # Define the finite element, the mesh, BCs and ICs
    # ----------------------------------------------------------------
    
    # Finite element definition
    HEX8 = Finite_Element("3D", 1, 8)
    
    # Problem definition
    # mesh
    elem_1    =     np.array([1,1,1, 0,1,1, 0,0,1, 1,0,1, 1,1,0, 0,1,0, 0,0,0, 1,0,0])  
    
    # Boundary conditions
    Dirichlet_BCs = np.array([0,0,0, 1,0,0, 1,1,0, 0,1,0, 0,0,1, 1,0,1, 1,1,1, 0,1,1])
    
    # displacement initial conditions
    u_ICs = np.zeros((1,24))[0]
    
    # prescribed displacement at selected dofs
    prescr_displ =  displ_value*np.array([0,0,1, 0,0,1, 0,0,1, 0,0,1, 0,0,0, 0,0,0, 0,0,0, 0,0,0])
    

    
    STATEV_ICs_GP = (sig, A, eps, eps_p, alp, gam)
    STATEV_ICs = [STATEV_ICs_GP]
    for i in range(0,8):
        STATEV_ICs.append(STATEV_ICs_GP)
    
    # ----------------------------------------------------------------
    # Set the simulation and launch it
    # ----------------------------------------------------------------
    
    # PDE problem definition
    one_element_compression = PDE_problem(Ceramic_1, HEX8, elem_1, Dirichlet_BCs, u_ICs, STATEV_ICs, prescr_displ, N_steps, T_step)
    
    # Eventual override of constants in the classes for convergence
    # one_element_compression.Material.MaxNewton = 200
    # one_element_compression.Material.TOL_r = 100
    
    u = one_element_compression.Solutor_NL_statics_displ_control()
    STATEV_History = one_element_compression.STATEV_History

    # post process print of S11 & E11 
    E11 = np.zeros(N_steps+1)
    S11 = np.zeros(N_steps+1)
    for i in range(0, N_steps+1):
        E11[i] = STATEV_History[i][0][2][2,2]
        S11[i] = STATEV_History[i][0][0][2,2]


# ----------------------------------------------------------------
# Print to file
# ----------------------------------------------------------------

# unique identifier for output 
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")

# write the output file
with open("test_" + dt_string + ".dat", "w") as f:
    print("# " + testscript, file=f)

    print("# " + "Material_model:", end=" ", file=f)
    print(*importedmod, sep=", ", file=f)

    print("# " + "Elas_params:", end=" ", file=f)
    print(*Elas_params, sep=", " ,file=f)
    print("# " + "Hard_params:", end=" ", file=f)
    print(*Hard_params, sep=", " ,file=f)
    print("# " + "Yield_params:", end=" ", file=f)
    print(*Yield_params, sep=", " ,file=f)

    print("# " + "testID: " + testID, file=f)
    
    print("# " + "Total displ value: %.5f" %displ_value, file=f)
    
    print("# " + "Number of steps: %d" %N_steps, file=f)
    
    print("# " + "T range: %4.0f : %4.0f" %(T0, Tmax), file=f)
    print("# " + "E11 \t\t S11", file=f)
    [print("%f \t, \t %f" %(E11[i], S11[i]), file=f) for i in range(0, N_steps+1)]

print("\n", "Simulation ID: ", "test_" + dt_string) 

