#!/usr/bin/env python
# coding: utf-8

# # Integer Programming
# 

# There are three types of model problems:
# * Total integer model: where all the decision variables required to have integer solution values.
# * Binary integer (0-1) model: where all decision variables required to have integer values of zero or one.
# 
# * Mixed integer (MI) model: where some of the decision variables (but not all) required to have integer values.

# ## A 0 - 1 Integer Model:
# Solve the following LP problem:
# 
# \begin{align}
# &\text{max}\\
# &\qquad z=300x_1+90x_2+400x_3+150x_4\\
# &\text{s.t.}\\
# &\qquad 35000x_1+10000x_2+25000x_3+90000x_4\le120000\\
# &\qquad 4x_1+2x_2+7x_3+3x_4\le 12\\
# &\qquad x_1+x_2 \le 1\\
# &\qquad x_1,x_2,x_3,x_4 = 0~\text{or}~1\\
# \end{align} 

# In[1]:


get_ipython().system('pip install gurobipy')

# Import gurobi library
from gurobipy import * # This command imports the Gurobi functions and classes.

import numpy as np

c = [300,90,400,150]    
A = [[35000,10000,25000,90000 ],
     [4,2,7,3 ],
     [1,1,0,0 ]]
b =  [120000,12,1]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("Example")

x = []
for i in decision_variables:
    x.append(m.addVar(lb = 0, vtype = GRB.BINARY, name = 'x' + str(i)))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MAXIMIZE) 
m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           <= b[j] for j in constraints), "constraints")
m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, var.obj)
print("objective value =", m.objVal)


# ## A Total Integer Model:
# Solve the following LP problem:
# 
# \begin{align}
# &\text{max}\\
# &\qquad z=100x_1+150x_2\\
# &\text{s.t.}\\
# &\qquad 8000x_1+4000x_2\le 40000\\
# &\qquad 15x_1+30x_2\le200\\
# &\qquad x_1,x_2\ge \text{and integer}\\
# \end{align} 

# In[3]:


c = [100,150]    
A = [[8000,4000],
     [15,30 ]]
b =  [40000,200]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("Example")

x = []
for i in decision_variables:
    x.append(m.addVar(lb = 0, vtype = GRB.INTEGER, name = 'x' + str(i)))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MAXIMIZE) 
m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           <= b[j] for j in constraints), "constraints")
m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, var.obj)
print("objective value =", m.objVal)


# ## Mixed Integer Model 
# Solve the following LP problem:
# 
# \begin{align}
# &\text{max}\\
# &\qquad z=9000x_1+1500x_2+1000x_3\\
# &\text{s.t.}\\
# &\qquad 50000x_1+12000x_2+8000x_3\le25000\\
# &\qquad x_1\le 4\\
# &\qquad x_2\le 15\\
# &\qquad x_3\le 20\\
# &\qquad x_2\ge 0\\
# &\qquad x_1,x_3\ge \text{and integer}\\
# \end{align} 

# In[4]:


c = [9000,1500,1000]    
A = [[50000,12000,8000],
     [1,0,0,0 ],
     [0,1,0,0 ],
     [0,0,1,0 ],
     [0,-1,0,0 ]]
b =  [250000,4,15,20,0]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("Example")

x = []
x.append(m.addVar(lb = 0, vtype = GRB.INTEGER, name = 'x' + str(i)))
x.append(m.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = 'x' + str(i)))
x.append(m.addVar(lb = 0, vtype = GRB.INTEGER, name = 'x' + str(i)))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MAXIMIZE) 
m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           <= b[j] for j in constraints), "constraints")
m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, var.obj)
print("objective value =", m.objVal)


# ## Problem Example: C5Q5
# Solve the following LP problem:
# 
# \begin{align}
# &\text{max}\\
# &\qquad z=50x_1+40x_2\\
# &\text{s.t.}\\
# &\qquad 3x_1+5x_2\le 150\\
# &\qquad 10x_1+4x_2\le 200\\
# &\qquad x_1,x_2\ge \text{and integer}\\
# \end{align} 

# In[5]:


c = [50,40]    
A = [[3,5 ],
     [10,4 ]]
b =  [150,200]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("C5Q5")

x = []
for i in decision_variables:
    x.append(m.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = 'x' + str(i)))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MAXIMIZE) 
m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           <= b[j] for j in constraints), "constraints")
m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, (var.obj,var.SAObjLow, var.SAObjUp, var.RC))

for con in m.getConstrs(): # constraints
    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal)


# In[6]:


y=[]
for var in m.getVars():
    y.append(np.floor(var.x))

np.array(y).dot(np.array((50,40)))


# In[7]:


c = [50,40]    
A = [[3,5 ],
     [10,4 ]]
b =  [150,200]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("C5Q5")

x = []
for i in decision_variables:
    x.append(m.addVar(lb = 0, vtype = GRB.INTEGER, name = 'x' + str(i)))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MAXIMIZE) 
m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           <= b[j] for j in constraints), "constraints")
m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, var.obj)

#for con in m.getConstrs(): # constraints
#    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
#          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal)


# ## Problem Example: C5Q13
# Solve the following LP problem:
# \begin{align}
# &\text{max}\\
# &\qquad z=85000 x_1 + 60000 x_2 – 18000 y_1\\
# &\text{s.t.}\\
# &\qquad x_1+x_2\le 10\\
# &\qquad 10000x_1+7000x_2\le 72000\\
# &\qquad x_1-10y_1\le 0\\
# &\qquad x_1,x_2\ge \text{and integer}\\
# &\qquad y_1= 0~\text{or}~1\\
# \end{align} 

# In[8]:


c = [85000,60000,-18000]    
A = [[1,1,0 ],
     [10000,7000,0 ],
     [1,0,-10],
     [-1,0,10]]
b =  [10,72000,0,9]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("C5Q13")

x = []
x.append(m.addVar(lb = 0, vtype = GRB.INTEGER, name = 'x1'))
x.append(m.addVar(lb = 0, vtype = GRB.INTEGER, name = 'x2'))
x.append(m.addVar(lb = 0, vtype = GRB.BINARY, name = 'x3'))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MAXIMIZE) 
m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           <= b[j] for j in constraints), "constraints")
m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, var.obj)

#for con in m.getConstrs(): # constraints
#    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
#          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal)


# ## Problem Example: C5Q14
# Solve the following LP problem:
# \begin{align}
# &\text{max}\\
# &\qquad z=.36x_1 + .82x_2 + .29x_3 + .16x_4
# + .56x_5 + .61x_6 + .48x_7 + .41x_8\\
# &\text{s.t.}\\
# &\qquad 60x_1 + 110x_2 + 53x_3 + 47x_4 +
# 92x_5 +85x_6 +73x_7 +65x_8\le 300\\
# &\qquad 7x_1 + 9x_2 + 8x_3 + 4x_4 + 7x_5 +
# 6x_6 +8x_7 +5x_8\le 40\\
# &\qquad x_2-x_5\le 0\\
# &\qquad x_i= 0~\text{or}~1\\
# \end{align} 

# In[9]:


c = [.36,.82,.29,.16,.56,.61,.48,.41]    
A = [[60,110,53,47,92,85,73,65 ],
     [7,9,8,4,7,6,8,5 ],
     [0,1,0,0,-1,0,0,0]]
b =  [300,40,0]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("C5Q14")

x = []
for i in decision_variables:
    x.append(m.addVar(lb = 0, vtype = GRB.BINARY, name = 'x' + str(i)))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MAXIMIZE) 
m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           <= b[j] for j in constraints), "constraints")
m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, var.obj)

#for con in m.getConstrs(): # constraints
#    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
#          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal*1e6)


# ## Problem Example: C5Q15
# Solve the following LP problem:
# \begin{align}
# &\text{max}\\
# &\qquad z=x_1 +x_2 +x_3 +x_4 +x_5 +x_6\\
# &\text{s.t.}\\
# &\qquad x_6+x_1 \ge 90\\
# &\qquad x_1+x_2 \ge 215\\
# &\qquad x_2+x_3 \ge 250\\
# &\qquad x_3+x_4 \ge 65\\
# &\qquad x_4+x_5 \ge 300\\
# &\qquad x_5+x_6 \ge 125\\
# &\qquad x_i \ge 0~ \text{and integer}\\
# \end{align} 

# In[10]:


c = [1,1,1,1,1,1]    
A = [[1,0,0,0,0,1],
     [1,1,0,0,0,0],
     [0,1,1,0,0,0],
     [0,0,1,1,0,0],
     [0,0,0,1,1,0],
     [0,0,0,0,1,1]]
b =  [90,215,250,65,300,125]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("C5Q15")

x = []
for i in decision_variables:
    x.append(m.addVar(lb = 0, vtype = GRB.INTEGER, name = 'x' + str(i)))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MINIMIZE) 
m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           >= b[j] for j in constraints), "constraints")
m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, var.obj)

#for con in m.getConstrs(): # constraints
#    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
#          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal)


# ## Problem Example: C5Q17
# Solve the following LP problem:
# \begin{align}
# &\text{min}\\
# &\qquad z=25000x_1 + 7000x_2 + 9000x_3\\
# &\text{s.t.}\\
# &\qquad 53000x_1+30000x_2+41000x_3 \ge 200,000\\
# &\qquad (32000x_1 + 20000x_2 +18000x_3)/(21000x_1 +10000x_2 +23000x_3) \ge 1.5\\
# &\qquad (34000x_1 +12000x_2 + 24000x_3)/(53000x_1 +30000x_2 +41000x_3) \ge .60\\
# &\qquad x_i \ge 0~ \text{and integer}\\
# \end{align} 
# 

# In[11]:


c = [25000,7000,9000]    
A = [[53000,30000,41000],
     [32000-1.5*21000,20000-1.5*10000,18000-1.5*23000 ],
     [34000-.6*53000,12000-.6*30000,24000-.6*41000]]
b =  [200000,0,0]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("C5Q17")

x = []
for i in decision_variables:
    x.append(m.addVar(lb = 0, vtype = GRB.INTEGER, name = 'x' + str(i)))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MINIMIZE) 
m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           >= b[j] for j in constraints), "constraints")
m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, var.obj)

#for con in m.getConstrs(): # constraints
#    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
#          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal*1e6)


# In[12]:


c = [25000,7000,9000]    
A = [[53000,30000,41000],
     [32000-1.5*21000,20000-1.5*10000,18000-1.5*23000 ],
     [34000-.6*53000,12000-.6*30000,24000-.6*41000]]
b =  [200000,0,0]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("C5Q17")

x = []
for i in decision_variables:
    x.append(m.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = 'x' + str(i)))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MINIMIZE) 
m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           >= b[j] for j in constraints), "constraints")
m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, var.obj)

#for con in m.getConstrs(): # constraints
#    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
#          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal*1e6)

