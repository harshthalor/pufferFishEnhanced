import numpy as np

class POA:
    def __init__(self, obj_func, lb, ub, dim, pop_size=50, max_iter=500):
        """
        Standard Pufferfish Optimization Algorithm (POA)
        """
        self.obj_func = obj_func
        # Support scalar or array boundaries
        self.lb = np.array(lb) if isinstance(lb, (list, np.ndarray)) else np.ones(dim) * lb
        self.ub = np.array(ub) if isinstance(ub, (list, np.ndarray)) else np.ones(dim) * ub
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        
    def optimize(self):
        # Step 1: Initialize population randomly (Equation 2)
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        
        # Evaluate initial fitness
        fitness = np.apply_along_axis(self.obj_func, 1, X)
        
        # Track the best solution found so far
        best_idx = np.argmin(fitness)
        best_X = X[best_idx].copy()
        best_fitness = fitness[best_idx]
        
        convergence_curve = np.zeros(self.max_iter)
        
        # Main Algorithm Loop
        for t in range(1, self.max_iter + 1):
            for i in range(self.pop_size):
                
                # ---------------------------------------------------------
                # PHASE 1: Exploration - Predator Attack (Equations 4, 5, 6)
                # ---------------------------------------------------------
                # Determine Candidate Pufferfish (CP) with better fitness
                CP_indices = np.where(fitness < fitness[i])[0]
                CP_indices = CP_indices[CP_indices != i] # Exclude self
                
                # Randomly select a target pufferfish from the candidates
                if len(CP_indices) > 0:
                    sp_idx = np.random.choice(CP_indices)
                    SP = X[sp_idx]
                else:
                    SP = best_X # If this is the best fish, it explores around itself
                    
                r1 = np.random.rand(self.dim)
                I = np.random.choice([1, 2], size=self.dim)
                
                # Calculate new position for Phase 1
                X_P1 = X[i] + r1 * (SP - I * X[i])
                
                # Return to boundaries if leaped out of bounds
                X_P1 = np.clip(X_P1, self.lb, self.ub)
                
                # Evaluate and conditionally update
                fit_P1 = self.obj_func(X_P1)
                if fit_P1 < fitness[i]:
                    X[i] = X_P1
                    fitness[i] = fit_P1
                    
                # ---------------------------------------------------------
                # PHASE 2: Exploitation - Defense Mechanism (Equations 7, 8)
                # ---------------------------------------------------------
                r2 = np.random.rand(self.dim)
                
                # Calculate new position for Phase 2
                X_P2 = X[i] + (1 - 2 * r2) * ((self.ub - self.lb) / t)
                
                # Return to boundaries if stepped out of bounds
                X_P2 = np.clip(X_P2, self.lb, self.ub)
                
                # Evaluate and conditionally update
                fit_P2 = self.obj_func(X_P2)
                if fit_P2 < fitness[i]:
                    X[i] = X_P2
                    fitness[i] = fit_P2
            
            # Update global best after both phases are complete
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_X = X[current_best_idx].copy()
                
            convergence_curve[t-1] = best_fitness
            
        return best_fitness, best_X, convergence_curve