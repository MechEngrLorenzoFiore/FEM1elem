"""
---
created: 2022-11-30-16-04-00
---

# Fem1elem V3R0
tags: tags: #computational #constitutive #numeric

# Progress
- aggiunta la dipendenza dalla temperatura nella V2
- sistemato il problema con il calcolo Voigt stress nel test Voigt_stress_test_221125_2

# ToDo
- [ ] creare print di debug di Voigt_stress per MPI

# log
- 30/11/22
    - creato come branch da Fem1elem_V3R0 per aggiungere feature di stampa 
        dato il lavoro di postprocess
"""

import numpy as np
import jax.numpy as jnp
from jax import grad
from jax import jacfwd
from scipy.optimize import fsolve

class Material:
    def __init__(self, params, ElasPotential, HardeningLaw, YieldFunction):
        self.Elas_params = params[0]
        self.Hard_params = params[1]
        self.Yield_params = params[2]
        self.ElasPotential = ElasPotential
        self.HardeningLaw = HardeningLaw
        self.YieldFunction = YieldFunction
        self.TOL_r = 1e-8
        self.TOL_f = 1e-5
        self.MaxNewton = 50
        self.DEBUG = False

    def Voigt_stress(self, eps_Voigt, GP_STATEV, T, STATEV_lck):        
        
        def Voigt_transform(ten, direction):
          if direction == "Voigt":
              return jnp.array([ten[0,0], ten[1,1], ten[2,2], ten[1,2], ten[0,2], ten[0,1]])
          elif direction == "tensor":
              return jnp.array([[ten[0], ten[5], ten[4]], [ten[5], ten[1], ten[3]], [ten[4], ten[3], ten[2]]])

        def Newton_solve_plasticity(eps_np1, eps_p_n, D_gam, alp_n):
            def plastic_residual(u, b):
                # un-known quantities to be computed
                eps_p_np1 = u[0:9].reshape((3,3))
                D_gam = u[9]
                alp_np1 = u[10]

                # known quantities needed for the computation
                eps_np1 = b[0:9].reshape((3,3))
                eps_p_n = b[9:18].reshape((3,3))
                alp_n = b[18]

                sig_np1 =  grad(self.ElasPotential, argnums= 0)(eps_np1 - eps_p_np1, T, self.Elas_params)
                A_np1 = self.HardeningLaw(alp_np1, T, self.Hard_params)
                
                dFdsig = grad(self.YieldFunction, argnums=0)(sig_np1,A_np1, self.Yield_params)
                
                # non-linear sistem statement
                r1 = - eps_p_np1 + eps_p_n + D_gam * dFdsig
                r2 = -alp_np1 + alp_n + D_gam * ( jnp.tensordot(dFdsig, dFdsig, axes=2) )**0.5
                r3 = self.YieldFunction(sig_np1, A_np1, self.Yield_params)

                return jnp.concatenate((jnp.concatenate(r1), jnp.array([r2]), jnp.array([r3])))
            
            # vector of un-dependant variables to be computed
            u = jnp.concatenate((jnp.concatenate(eps_p_n), jnp.array([D_gam]), jnp.array([alp_n])))
            # known quantities needed for the computation
            b = jnp.concatenate((jnp.concatenate(eps_np1), jnp.concatenate(eps_p_n), jnp.array([alp_n])))

            if self.DEBUG == True:
                r = plastic_residual(u,b)
                normr = jnp.linalg.norm(r, ord=2)
                print(f"Plastic residual norm before scipy {normr:20.2E} ")
                
                # scipy fsolve
                u, infodict, ier, mesg = fsolve(plastic_residual, u, args=(b), full_output=1, xtol=self.TOL_r, maxfev=self.MaxNewton) 
                # u, infodict, ier, mesg = fsolve(plastic_residual, u, args=(b), full_output=1 ) 
                print(mesg)
                
                r = plastic_residual(u,b)
                normr = jnp.linalg.norm(r, ord=2)
                print(f"Plastic residual norm after scipy {normr:20.2E} ")
            
            else:
                # Newton-Raphson solution of the non-linear system
                # fsolve can cause information leakege
                r = plastic_residual(u,b)
                normr = jnp.linalg.norm(r, ord=2)
                k = 0
                while normr > self.TOL_r:
                    if k > self.MaxNewton:
                        break
                    dr_du = jacfwd(plastic_residual, argnums=0)(u, b)
                    Du = jnp.linalg.solve(dr_du, -r)
                    u = u + Du
                    r = plastic_residual(u,b)
                    normr = jnp.linalg.norm(r, ord=2)
                    k += 1
            
            eps_p_np1 = u[0:9].reshape((3,3)) 
            D_gam = u[9]
            alp_np1 = u[10]

            return (eps_p_np1, D_gam, alp_np1)

        eps_p, alp, gam = GP_STATEV[3], GP_STATEV[4], GP_STATEV[5]  
                
        eps = Voigt_transform(eps_Voigt, "tensor")
        sig = grad(self.ElasPotential, argnums= 0)(eps - eps_p, T, self.Elas_params)
        A = self.HardeningLaw(alp, T, self.Hard_params)
        
        F =  self.YieldFunction(sig, A, self.Yield_params)
        if self.DEBUG == True:
            print("Yield function value: %.2e" %F)
        
        if F < self.TOL_f:
            if self.DEBUG == True:
                print("Step is ELASTIC!")
            pass
        else:
            if self.DEBUG == True:
                print("Step is PLASTIC!")
            # ------------------------ return mapping algo ------------------------
            (eps_p, D_gam, alp) = Newton_solve_plasticity(eps, eps_p, 0, alp)
            sig = grad(self.ElasPotential, argnums= 0)(eps - eps_p, T, self.Elas_params)
            A = self.HardeningLaw(alp, T, self.Hard_params)
            gam = gam + D_gam
                
        # DEBUG prints
        if self.DEBUG == True:
            print("Plastic deformation: \n", eps_p)
            print("alp: ", alp)
        
        sigma_Voigt = Voigt_transform(sig, "Voigt")
        updated_GP_STATEV = (sig, A, eps, eps_p, alp, gam)
        if STATEV_lck == 0:
            return sigma_Voigt
        elif STATEV_lck == 1:
            return sigma_Voigt, updated_GP_STATEV


class Finite_Element:
    def __init__(self, formulation, h_order, GPn):
        self.formulation = formulation
        self.h_order = h_order
        self.GPn = GPn
        if self.formulation == "3D":
            # Gauss point coordinate parameter
            self.GP_coord = 0.577
            # Gauss point natural coordinates
            self.GP = np.array([[self.GP_coord, self.GP_coord, self.GP_coord],
                          [-self.GP_coord, self.GP_coord, self.GP_coord],
                          [-self.GP_coord, -self.GP_coord, self.GP_coord],
                          [self.GP_coord, -self.GP_coord, self.GP_coord],
                          [self.GP_coord, self.GP_coord, -self.GP_coord],
                          [-self.GP_coord, self.GP_coord, -self.GP_coord],
                          [-self.GP_coord, -self.GP_coord, -self.GP_coord],
                          [self.GP_coord, -self.GP_coord, -self.GP_coord]])
            # Gauss weights
            self.G_weight = np.array([1,1,1,1, 1,1,1,1])
        self.elem_STATEV = []

    def shape_function(self, nat_coord ,ID):
        # linear element 8 node of the rectangular serendipity family
        # pag 121 Zienkiewicz, The finite element method: its basis and fundamentals
        if self.formulation == "3D":
            if ID == 1:
                xi_nod, eta_nod, zita_nod = 1, 1, 1
            elif ID == 2:
                xi_nod, eta_nod, zita_nod = -1, 1, 1
            elif ID == 3:
                xi_nod, eta_nod, zita_nod = -1, -1, 1
            elif ID == 4:
                xi_nod, eta_nod, zita_nod = 1, -1, 1
            elif ID == 5:
                xi_nod, eta_nod, zita_nod = 1, 1, -1
            elif ID == 6:
                xi_nod, eta_nod, zita_nod = -1, 1, -1
            elif ID == 7:
                xi_nod, eta_nod, zita_nod = -1, -1, -1
            elif ID == 8:
                xi_nod, eta_nod, zita_nod = 1, -1, -1
            else:
                print("Please specify correct node ID from 1 to 8")
            xi, eta, zita = nat_coord[0], nat_coord[1], nat_coord[2]
            return 1/8 * (1 + xi_nod * xi) * (1 + eta_nod * eta) * (1 + zita_nod * zita)
    
    def coord_trans(self, nat_coord, coord_nod_elem):
        # isoparametric element, same shape functions for field description 
        # and coordinate transformation
        if self.formulation == "3D":
            x, y, z = 0, 0, 0    
            # iterate through the nodes
            for i in range(1,9):
                x = self.shape_function(nat_coord, i) * coord_nod_elem[0+3*(i-1)] + x
                y = self.shape_function(nat_coord, i) * coord_nod_elem[1+3*(i-1)] + y
                z = self.shape_function(nat_coord, i) * coord_nod_elem[2+3*(i-1)] + z        
            return jnp.array([x, y, z])
    
    def Jacobian_matr(self, nat_coord, coord_nod_elem):
        return jacfwd(self.coord_trans, argnums=0)(nat_coord, coord_nod_elem)
    
    def B_strain_disp_matr(self, nat_coord, coord_nod_elem):
        # pag 204 di Zienkiewicz
        Jacobian_matr_inv = jnp.linalg.inv(self.Jacobian_matr(nat_coord, coord_nod_elem))
        # [dNa_dx, dNa_dy, dNa_dz]
        # iterate through the nodes
        if self.formulation == "3D":
            for i in range(1,9):
                shape_function_der = jnp.dot(Jacobian_matr_inv, grad(self.shape_function, argnums=0)(nat_coord ,i))
                # B is to apply to a vector of displacement of a certain node
                # to obtain vector form of strain (Voigt notation) 
                B_ID = jnp.zeros((6,3))
                B_ID = B_ID.at[0,0].set(shape_function_der[0])
                B_ID = B_ID.at[1,1].set(shape_function_der[1])
                B_ID = B_ID.at[2,2].set(shape_function_der[2])
                B_ID = B_ID.at[3,0].set(shape_function_der[1])
                B_ID = B_ID.at[3,1].set(shape_function_der[0])
                B_ID = B_ID.at[4,1].set(shape_function_der[2])
                B_ID = B_ID.at[4,2].set(shape_function_der[1])
                B_ID = B_ID.at[5,0].set(shape_function_der[2])
                B_ID = B_ID.at[5,2].set(shape_function_der[0])
                if i == 1:
                    B = B_ID
                else:
                    B = jnp.concatenate((B,B_ID), axis=1)
            return B
    
    def Gauss_integration_int(self, current_nod_coords, u, T, Voigt_stress, elem_STATEV, STATEV_lck):
        # Gauss integration of specific element for internal force vector
        # for Gauss Points and Weights pag 162 of Zienkiewicz
        def integrand(i):
            nat_coord = self.GP[i]
            j = jnp.linalg.det(self.Jacobian_matr(nat_coord, current_nod_coords))
            eps_Voigt = jnp.einsum('ij,j->i', self.B_strain_disp_matr(self.GP[i], current_nod_coords), u)
            GP_STATEV = elem_STATEV[i]
            if STATEV_lck == 0:
                sigma_Voigt = Voigt_stress(eps_Voigt, GP_STATEV, T, STATEV_lck)
                return j*jnp.einsum('ij,j->i',jnp.transpose(self.B_strain_disp_matr(nat_coord, current_nod_coords)), sigma_Voigt)
            elif STATEV_lck == 1:
                sigma_Voigt, updated_GP_STATEV = Voigt_stress(eps_Voigt, GP_STATEV, T, STATEV_lck)
                return j*jnp.einsum('ij,j->i',jnp.transpose(self.B_strain_disp_matr(nat_coord, current_nod_coords)), sigma_Voigt), updated_GP_STATEV
        
        if STATEV_lck == 0:
            for i in range(0, self.GPn):
                # print("Gauss point %d under integration" %(i+1))
                if i == 0:
                    integral = integrand(i)*self.G_weight[i]
                else:
                    integral = integrand(i)*self.G_weight[i] + integral
            return integral
        elif STATEV_lck == 1:
            updated_elem_STATEV = []
            for i in range(0, self.GPn):
                # print("Gauss point %d under integration" %(i+1))
                if i == 0:
                    integrand_i, updated_GP_STATEV = integrand(i)
                    integral = integrand_i*self.G_weight[i]
                    updated_elem_STATEV.append(updated_GP_STATEV)
                else:
                    integrand_i, updated_GP_STATEV = integrand(i)
                    integral = integrand_i*self.G_weight[i] + integral
                    updated_elem_STATEV.append(updated_GP_STATEV)
            return integral, updated_elem_STATEV
    
class PDE_problem:
    def __init__(self, Material, Finite_Element, mesh, BCs, ICs, STATEV_ICs, prescr_displ, N_steps, T_step):
        # given class attributes
        self.Material = Material
        self.Finite_Element = Finite_Element
        self.mesh = mesh
        self.BCs = BCs
        self.ICs = ICs
        self.STATEV_ICs = STATEV_ICs
        self.N_steps = N_steps
        # Specified temperature for every step, the same at all the integration points
        self.T_step = T_step
        # built class attributes
        self.Dup = prescr_displ/N_steps
        self.up_logic = BCs + prescr_displ != 0
        self.uf_logic = self.up_logic == 0
        # Simulation history data structure
        self.STATEV_History = []
        # Class constants
        self.TOL_r = 1e-1
        self.MaxNewton = 10
        
    def Solutor_NL_statics_displ_control(self):
        def Newton_solve_NL_statics_displ_control(u, T, n):
            def global_residual(uf, up, STATEV_lck):
                u = jnp.zeros((1,24))[0]
                u = u.at[jnp.where(self.uf_logic)].set(uf)
                u = u.at[jnp.where(self.up_logic)].set(up)
                current_nod_coords = self.mesh + u
                # the next section should be iterated on the available elements
                elem_STATEV = self.STATEV_History[-1]
                if STATEV_lck == 0:
                    f_int = self.Finite_Element.Gauss_integration_int(current_nod_coords, u, T, self.Material.Voigt_stress, elem_STATEV, STATEV_lck)
                elif STATEV_lck == 1:
                    f_int, updated_elem_STATEV = self.Finite_Element.Gauss_integration_int(current_nod_coords, u, T, self.Material.Voigt_stress, elem_STATEV, STATEV_lck)
                # once calculated on the elements, the residual should be assembled
                # the updated_elem_STATEV should be assembled in a updated_elems_STATEV list
                # and the only free DOF positions should be used
                r = f_int[jnp.where(self.uf_logic)]
                if STATEV_lck == 0:
                    return r
                elif STATEV_lck == 1:
                    return r, updated_elem_STATEV
            
            # split free and prescribed displacements
            uf = u[jnp.where(self.uf_logic)]
            up = u[jnp.where(self.up_logic)]
            
            r = global_residual(uf, up, 0)
            rff = jacfwd(global_residual, argnums=0)(uf, up, 0)
            rfp = jacfwd(global_residual, argnums=1)(uf, up, 0)
            Duf = - jnp.linalg.solve(rff, jnp.einsum('ij,j->i', rfp, self.Dup[jnp.where(self.up_logic)] ) + r )
            uf = uf + Duf

            normr = jnp.linalg.norm(r, ord=2)
            print(f"Global residual norm {normr:20.2E} ")
            k = 0
            while normr > self.TOL_r:
                if k > self.MaxNewton:
                    break
                r, updated_elems_STATEV = global_residual(uf, up, 1)
                rff = jacfwd(global_residual, argnums=0)(uf, up, 0)
                # correction of displacement control Newton cycle
                Duf = - jnp.linalg.solve(rff, r)
                uf = uf + Duf
                normr = jnp.linalg.norm(r, ord=2)
                print(f"Global residual norm {normr:20.2E} ")
                k += 1

            self.STATEV_History.append(updated_elems_STATEV)
            u[np.where(self.uf_logic)] += Duf
            return u
        
        u = self.ICs
        self.STATEV_History.append(self.STATEV_ICs)
        for n in range(0, self.N_steps):
            print("----- Step %d -----" %(n+1) )
            u += self.Dup
            print("Displacement: %f" %u[2])
            T = self.T_step[n]
            print("Temperature at all GP: %f" %T )
            u = Newton_solve_NL_statics_displ_control(u, T, n) 
        return u


