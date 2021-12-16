#!/usr/bin/env python
# coding: utf-8

# # Chapter 5: Transportation, Transshipment, and Assignment Problems
# 

# Characteristics of the Transportation Model: 
# * A product is transported from several sources to a number of destinations at the **minimum** possible cost.
# * Each source can **supply** a fixed number of units of the product, and each destination has a fixed **demand** for the product.
# * The linear programming model has constraints for supply at each source and demand at each destination.
# * All constraints are equalities in a balanced transportation model where **supply equals demand**.
# * Constraints contain inequalities in **unbalanced** models where **supply does not equal demand**.
# 

# ## Transportation Model Example
# How many tons of wheat to transport from each factory to each city on a monthly basis in order to **minimize** the total cost of transportation?
# 
# Factory | Supply | City  | Demand
# -|-|-|-
# 1. Kansas City | 150 | A. Chicago | 220
# 2. Omaha | 175 | B. St. Louis | 100
# 3. Des Moines | 275 | C. Cincinnati | 300
#  Total | 600 | Total |600
# 
# Transport Cost from factory to city ($/ton):
# 
# Factory | A. Chicago | B. St. Louis  | C. Cincinnati
# -|-|-|-
# 1. Kansas City | 6 | 8 | 10
# 2. Omaha | 7 | 11 | 11
# 3. Des Moines | 4 | 5 | 12
# 
# The LP transportation model is the following:
# \begin{align}
# &\text{min}\\
# &\qquad z=6x_{1A}+8x_{1B}+10x_{1C}
# +7x_{2A}+11x_{2B}+11x_{2C}
# +4x_{3A}+5x_{3B}+12x_{3C}\\
# &\text{s.t.}\\
# &\qquad x_{1A}+x_{1B}+x_{1C} = 150\\
# &\qquad x_{2A}+x_{2B}+x_{2C} = 175\\
# &\qquad x_{3A}+x_{3B}+x_{3C} = 275\\
# &\qquad x_{1A}+x_{2A}+x_{3A} = 200\\
# &\qquad x_{1B}+x_{2B}+x_{3B} = 100\\
# &\qquad x_{1C}+x_{2C}+x_{3C} = 300\\
# &\qquad x_{ij} \ge 0~ \text{and integer}\\
# \end{align}  
# 
# 

# In[1]:


get_ipython().system('pip install gurobipy')

# Import gurobi library
from gurobipy import * # This command imports the Gurobi functions and classes.

import numpy as np

# Transportation Model Example
c = [6,8,10,7,11,11,4,5,12]    
A = [[1,1,1,0,0,0,0,0,0],
     [0,0,0,1,1,1,0,0,0],
     [0,0,0,0,0,0,1,1,1],
     [1,0,0,1,0,0,1,0,0],
     [0,1,0,0,1,0,0,1,0],
     [0,0,1,0,0,1,0,0,1]]
b =  [150,175,275,200,100,300]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("example")

x = []
for i in decision_variables:
    x.append(m.addVar(lb = 0, vtype = GRB.INTEGER, name = 'x' + str(i)))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MINIMIZE) 
m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           == b[j] for j in constraints), "constraints")
m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, var.obj)

#for con in m.getConstrs(): # constraints
#    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
#          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal)


# ## Problem Example: C6Q4
# Solve the following LP problem:
# 
# \begin{align}
# &\text{min}\\
# &\qquad z=14x_{A1} + 9x_{A2} + 16x_{A3} + 18x_{A4}\\
# &\qquad ~~+ 11x_{B1} + 8x_{B2} + 100x_{B3} + 16x_{B4}\\
# &\qquad ~~+ 16x_{C1} + 12x_{C2} + 10x_{C3} + 22x_{C4}\\
# &\text{subject to}\\
# &\qquad x_{A1} +x_{A2} +x_{A3} +x_{A4} \le 150 \\
# &\qquad x_{B1} +x_{B2} +x_{B3} +x_{B4} \le 210\\ 
# &\qquad x_{C1} +x_{C2} +x_{C3} +x_{C4} \le 320\\
# &\qquad x_{A1} +x_{B1} +x_{C1} =130 \\
# &\qquad x_{A2} +x_{B2} +x_{C2} =70 \\
# &\qquad x_{A3} +x_{B3} +x_{C3} =180\\
# &\qquad x_{A4} +x_{B4} +x_{C4} =240\\
# &\qquad x_{ij} \ge 0~ \text{and integer}\\
# \end{align}  
# 

# In[2]:


c = [14,9,16,18,11,8,1000,16,16,12,10,22]    
A = [[1,1,1,1,0,0,0,0,0,0,0,0],
     [0,0,0,0,1,1,1,1,0,0,0,0],
     [0,0,0,0,0,0,0,0,1,1,1,1],
     [1,0,0,0,1,0,0,0,1,0,0,0],
     [0,1,0,0,0,1,0,0,0,1,0,0],
     [0,0,1,0,0,0,1,0,0,0,1,0],
     [0,0,0,1,0,0,0,1,0,0,0,1]]
b =  [150,210,320,130,70,180,240]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("example")

x = []
for i in decision_variables:
    x.append(m.addVar(lb = 0, vtype = GRB.INTEGER, name = 'x' + str(i)))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MINIMIZE) 

m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           <= b[j] for j in constraints[:3]), "constraints")

m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           == b[j] for j in constraints[3:]), "constraints")
              
m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, var.obj)

#for con in m.getConstrs(): # constraints
#    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
#          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal)


# In[3]:


c = [14,9,16,18,11,8,1000,16,16,12,10,22]    
A = [[1,1,1,1,0,0,0,0,0,0,0,0],
     [0,0,0,0,1,1,1,1,0,0,0,0],
     [0,0,0,0,0,0,0,0,1,1,1,1],
     [1,0,0,0,1,0,0,0,1,0,0,0],
     [0,1,0,0,0,1,0,0,0,1,0,0],
     [0,0,1,0,0,0,1,0,0,0,1,0],
     [0,0,0,1,0,0,0,1,0,0,0,1]]
b =  [150,210,290,130,70,180,240]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("example")

x = []
for i in decision_variables:
    x.append(m.addVar(lb = 0, vtype = GRB.INTEGER, name = 'x' + str(i)))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MINIMIZE) 

m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           <= b[j] for j in constraints[:3]), "constraints")

m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           == b[j] for j in constraints[3:]), "constraints")
              
m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, var.obj)

#for con in m.getConstrs(): # constraints
#    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
#          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal)


# ## Problem Example: C6Q30
# Transshipment probelm is an extension of the transportation model in which intermediate trans-shipment points are added between sources and destinations.

# In[4]:


c = [420,390,610,510,590,470,450,360,380,75,63,81,125,110,95,68,82,95]    
A = [[1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0],
     [0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0],
     [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1],
     [1,0,0,1,0,0,1,0,0,-1,-1,-1,0,0,0,0,0,0],
     [0,1,0,0,1,0,0,1,0,0,0,0,-1,-1,-1,0,0,0],
     [0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,-1,-1,-1]]
b =  [55,78,37,60,45,50,0,0,0]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("example")

x = []
for i in decision_variables:
    x.append(m.addVar(lb = 0, vtype = GRB.INTEGER, name = 'x' + str(i)))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MINIMIZE) 

m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           <= b[j] for j in constraints[:3]), "constraints")

m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           == b[j] for j in constraints[3:]), "constraints")
              
m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, var.obj)

#for con in m.getConstrs(): # constraints
#    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
#          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal)


# ## Problem Example: C6Q43
# Solve the following LP problem:
# 
# \begin{align}
# &\text{min}\\
# &\qquad z=12x_{1A} + 11x_{1B} + 8x_{1C} + 14x_{1D}\\
# &\qquad~~+ 10x_{2A} + 9x_{2B} + 10x_{2C} + 8x_{2D}\\
# &\qquad~~+ 14x_{3A} + 100x_{3B} + 7x_{3C} + 11x_{3D}\\ 
# &\qquad~~+ 6x_{4A} + 8x_{4B} + 10x_{4C} + 9x_{4D}\\
# &\text{subject to}\\
# &\qquad x_{1A} +x_{1B} +x_{1C} +x_{1D} =1 \\
# &\qquad x_{2A} +x_{2B} +x_{2C} +x_{2D} =1 \\
# &\qquad x_{3A} +x_{3B} +x_{3C} +x_{3D} =1 \\
# &\qquad x_{4A} +x_{4B} +x_{4C} +x_{4D} =1 \\
# &\qquad x_{1A} +x_{2A} +x_{3A} +x_{4A} =1 \\
# &\qquad x_{1B} +x_{2B} +x_{3B} +x_{4B} =1 \\
# &\qquad x_{1C} +x_{2C} +x_{3C} +x_{4C} =1 \\
# &\qquad x_{1D} +x_{2D} +x_{3D} +x_{4D} =1 \\
# &\qquad x_{ij} \ge 0~ \text{and integer}\\
# \end{align} 

# In[5]:


c = [12,11,8,14,10,9,10,8,14,1000,7,11,6,8,10,9]    
A = [[1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
     [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
     [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],
     [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
     [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]]
b =  [1,1,1,1,1,1,1,1]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("example")

x = []
for i in decision_variables:
    x.append(m.addVar(lb = 0, vtype = GRB.INTEGER, name = 'x' + str(i)))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MINIMIZE) 

m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           == b[j] for j in constraints), "constraints")
           
m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, var.obj)

#for con in m.getConstrs(): # constraints
#    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
#          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal)

