import numpy as np
import math

class POA_Enhanced:
    def __init__(self, obj_func, lb, ub, dim, pop_size=50, max_iter=500):
        """
        Enhanced Pufferfish Optimization Algorithm 
        Includes: OBL Initialization + Levy Flight Exploration
        """
        self.obj_func = obj_func
        self.lb = np.array(lb) if isinstance(lb, (list, np.ndarray)) else np.ones(dim) * lb
        self.ub = np.array(ub) if isinstance(ub, (list, np.ndarray)) else np.ones(dim) * ub
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter

    def levy_flight(self):
        """Mantegna's algorithm for Levy flight step generation"""
        beta = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
                 (math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / abs(v)**(1 / beta)
        return step

    def optimize(self):
        # ---------------------------------------------------------
        # ENHANCEMENT 1: Opposition-Based Learning (OBL) Initialization
        # ---------------------------------------------------------
        # Generate random population
        X_rand = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        # Generate opposite population
        X_obl = self.lb + self.ub - X_rand
        
        # Combine and evaluate both
        X_combined = np.vstack((X_rand, X_obl))
        fitness_combined = np.apply_along_axis(self.obj_func, 1, X_combined)
        
        # Sort and keep the best 'pop_size' individuals
        sorted_indices = np.argsort(fitness_combined)
        X = X_combined[sorted_indices[:self.pop_size]]
        fitness = fitness_combined[sorted_indices[:self.pop_size]]
        
        best_idx = 0 # Since it's sorted, index 0 is the best
        best_X = X[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        convergence_curve = np.zeros(self.max_iter)
        
        # Main Algorithm Loop
        for t in range(1, self.max_iter + 1):
            for i in range(self.pop_size):
                
                # Determine Candidate Pufferfish (CP)
                CP_indices = np.where(fitness < fitness[i])[0]
                CP_indices = CP_indices[CP_indices != i]
                
                if len(CP_indices) > 0:
                    sp_idx = np.random.choice(CP_indices)
                    SP = X[sp_idx]
                else:
                    SP = best_X
                    
                # ---------------------------------------------------------
                # ENHANCEMENT 2: Levy Flight in Exploration Phase
                # ---------------------------------------------------------
                levy_step = self.levy_flight()
                I = np.random.choice([1, 2], size=self.dim)
                
                # Replace r1 with the Levy step
                X_P1 = X[i] + levy_step * (SP - I * X[i])
                X_P1 = np.clip(X_P1, self.lb, self.ub)
                
                fit_P1 = self.obj_func(X_P1)
                if fit_P1 < fitness[i]:
                    X[i] = X_P1
                    fitness[i] = fit_P1
                    
                # PHASE 2: Exploitation (Remains standard for now)
                r2 = np.random.rand(self.dim)
                X_P2 = X[i] + (1 - 2 * r2) * ((self.ub - self.lb) / t)
                X_P2 = np.clip(X_P2, self.lb, self.ub)
                
                fit_P2 = self.obj_func(X_P2)
                if fit_P2 < fitness[i]:
                    X[i] = X_P2
                    fitness[i] = fit_P2
            
            # Update global best
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_X = X[current_best_idx].copy()
                
            convergence_curve[t-1] = best_fitness
            
        return best_fitness, best_X, convergence_curve