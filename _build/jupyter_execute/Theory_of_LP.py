#!/usr/bin/env python
# coding: utf-8

# # Theory of LP and the Simplex method
# 

# To be done later on
# 
# To place here a python code that follows the Simplex algorithm

# In[1]:


import numpy as np
get_ipython().system('pip install gurobipy')

# Import gurobi library
from gurobipy import * # This command imports the Gurobi functions and classes.


c = [40,65,70,30]    
A = [[1,1,0,0 ],
     [0,0,1,1 ],
     [1,0,1,0 ],
     [0,1,0,1 ]]
b =  [250, 400, 300, 350]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("C4Q33")

x = []
for i in decision_variables:
    x.append(m.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = 'x' + str(i)))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MINIMIZE) 
m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           == b[j] for j in constraints), "constraints")
m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, (var.obj,var.SAObjLow, var.SAObjUp, var.RC))

for con in m.getConstrs(): # constraints
    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal)

