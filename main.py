import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from opfunu.cec_based.cec2017 import F12017, F32017, F42017
from POA_Standard import POA
from POA_Enhanced import POA_Enhanced
from POA_Enhanced_v2 import POA_Enhanced_v2
from POA_Enhanced_v3 import POA_Enhanced_v3

def run_comparison():
    dimensions = 10
    pop_size = 50
    max_iter = 500
    
    funcs = [
        ("C17-F1 (Target: 100)", F12017(ndim=dimensions)),
        ("C17-F3 (Target: 300)", F32017(ndim=dimensions)),
        ("C17-F4 (Target: 400)", F42017(ndim=dimensions))
    ]
    
    print(f"--- 4-Way Comparison: The Evolution of Enhanced POA ---")
    print(f"Dim: {dimensions} | Pop: {pop_size} | Iter: {max_iter}\n")
    
    for name, func_obj in funcs:
        lb = func_obj.lb
        ub = func_obj.ub
        
        # 1. Run Standard POA
        std_optimizer = POA(obj_func=func_obj.evaluate, lb=lb, ub=ub, dim=dimensions, pop_size=pop_size, max_iter=max_iter)
        std_best, _, _ = std_optimizer.optimize()
        
        # 2. Run Enhanced v1 (OBL + Levy)
        enh_v1_optimizer = POA_Enhanced(obj_func=func_obj.evaluate, lb=lb, ub=ub, dim=dimensions, pop_size=pop_size, max_iter=max_iter)
        enh_v1_best, _, _ = enh_v1_optimizer.optimize()

        # 3. Run Enhanced v2 (The Explosive Exponential Decay)
        enh_v2_optimizer = POA_Enhanced_v2(obj_func=func_obj.evaluate, lb=lb, ub=ub, dim=dimensions, pop_size=pop_size, max_iter=max_iter)
        enh_v2_best, _, _ = enh_v2_optimizer.optimize()
        
        # 4. Run Enhanced v3 (The Adaptive Guided Fix)
        enh_v3_optimizer = POA_Enhanced_v3(obj_func=func_obj.evaluate, lb=lb, ub=ub, dim=dimensions, pop_size=pop_size, max_iter=max_iter)
        enh_v3_best, _, _ = enh_v3_optimizer.optimize()
        
        # Print Results
        print(f"Function: {name}")
        print(f"  Standard POA : {std_best:,.4f}")
        print(f"  Enhanced v1  : {enh_v1_best:,.4f}")
        print(f"  Enhanced v2  : {enh_v2_best:,.4f}")
        print(f"  Enhanced v3  : {enh_v3_best:,.4f}")
        print("-" * 45)

if __name__ == "__main__":
    run_comparison()