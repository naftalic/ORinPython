#!/usr/bin/env python
# coding: utf-8

# # Multicriteria Decision Making

# Goal programming is a variation of linear programming considering more than one objective (goals) in the objective function. Goal programming solutions do not always achieve all goals and they are not optimal; they achieve the best or most satisfactory solution possible.
# 
# Goal Constraint Requirements:
# * All goal constraints are equalities that include deviational variables $d^-$ and $d^+$
# * A negative deviational variable, $d^-$, is the amount by which a goal level is **under-achieved**
# * A positive deviational variable, $d^+$, is the amount by which a goal level is **exceeded**
# * **At least one** or both deviational variables in a goal constraint must equal zero
# * The objective function seeks to **minimize** the deviation from the respective goals in the order of the goal priorities.
# 
# 
# 
#  
# 

# ##Goal programming example:
# Solve the following problem:
# 
# \begin{align}
# &\text{min}\\
# &\qquad z=P_1d_1^-,P_2d_2^-,P_3d_3^+,P_4d_1^+\\
# &\text{s.t.}\\
# &\qquad x_1+2x_2+d_1^--d_1^+=40\\
# &\qquad 40x_1+50x_2+d_2^--d_2^+=1600\\
# &\qquad 4x_1+3x_2+d_3^--d_3^+=120\\
# &\qquad x_i,d_j^-,d_j^+ \ge 0~ \text{and integer}\\
# \end{align}  
# 

# In[1]:


get_ipython().system('pip install gurobipy')

# Import gurobi library
from gurobipy import * # This command imports the Gurobi functions and classes.

import numpy as np

m = Model('multiobj')

# Add Variables
x = m.addVars(range(2), name='x', lb=0, vtype = GRB.INTEGER)
dm = m.addVars(range(3), name='dm', lb=0, vtype = GRB.INTEGER)
dp = m.addVars(range(3), name='dp', lb=0, vtype = GRB.INTEGER)
#totSlack = m.addVar(name='totSlack')
m.update()

# Add Constraints
c1 = m.addConstr(1 *x[0]  + 2*x[1] + 1*dm[0] - 1*dp[0] == 40)
c2 = m.addConstr(40*x[0] + 50*x[1] + 1*dm[1] - 1*dp[1] == 1600)
c3 = m.addConstr(4 *x[0]  + 3*x[1] + 1*dm[2] - 1*dp[2] == 120)

# Add Objective Function
m.ModelSense = GRB.MINIMIZE
m.setObjectiveN(dm[0], index=0, priority=3)
m.setObjectiveN(dm[1], index=1, priority=2)
m.setObjectiveN(dp[2], index=2, priority=1)
m.setObjectiveN(dp[0], index=3, priority=0)
# Optimize Model
m.optimize()

# Output formatted solution
for v in m.getVars():
    print(v.varName, v.x)
#print('Obj :', m.objVal)


# ##Altered goal programming example:
# Solve the following problem:
# \begin{align}
# &\text{min}\\
# &\qquad z=P_1d_1^-,P_2d_2^-,P_3d_3^+,P_4d_1^+,4P_5d_5^-+5P_5d_6^-\\
# &\text{s.t.}\\
# &\qquad x_1+2x_2+d_1^--d_1^+=40\\
# &\qquad 40x_1+50x_2+d_2^--d_2^+=1600\\
# &\qquad 4x_1+3x_2+d_3^--d_3^+=120\\
# &\qquad  d_1^++d_4^--d_4^+=10 \\
# &\qquad x_1+d_5^-=30\\
# &\qquad x_2+d_6^-=20\\
# &\qquad x_i,d_j^-,d_j^+ \ge 0~ \text{and integer}\\
# \end{align}  

# In[3]:


from gurobipy import *
m = Model('multiobj')

# Add Variables
x = m.addVars(range(2), name='x', lb=0, vtype = GRB.INTEGER)
dm = m.addVars(range(6), name='dm', lb=0, vtype = GRB.INTEGER)
dp = m.addVars(range(4), name='dp', lb=0, vtype = GRB.INTEGER)
#totSlack = m.addVar(name='totSlack')
m.update()

# Add Constraints
c1 = m.addConstr(1 *x[0]  + 2 *x[1]  + 1*dm[0] - 1*dp[0] == 40)
c2 = m.addConstr(40*x[0]  + 50*x[1]  + 1*dm[1] - 1*dp[1] == 1600)
c3 = m.addConstr(4 *x[0]  + 3 *x[1]  + 1*dm[2] - 1*dp[2] == 120)
c4 = m.addConstr(                      1*dp[0] + 1*dm[3] - 1*dp[3] == 10)
c5 = m.addConstr(1 *x[0]             + 1*dm[4]           == 30)
c6 = m.addConstr(           1 *x[1]  + 1*dm[5]           == 20)

# Add Objective Function
m.ModelSense = GRB.MINIMIZE
m.setObjectiveN(dm[0], index=0, priority=4)
m.setObjectiveN(dm[1], index=1, priority=3)
m.setObjectiveN(dp[2], index=2, priority=2)
m.setObjectiveN(dp[3], index=3, priority=1)
m.setObjectiveN(4*dm[4]+5*dm[5], index=4, priority=0)

# Optimize Model
m.optimize()

# Output formatted solution
for v in m.getVars():
    print(v.varName, v.x)
#print('Obj :', m.objVal)


# ## Problem Example: C9Q8
# Solve the following goal programming model:
# \begin{align}
# &\text{min}\\
# &\qquad z=P_1d_1^-+P_1d_1^+,P_2d_2^-,P_3d_3^-,3P_4d_2^++5P_4d_3^+\\
# &\text{s.t.}\\
# &\qquad x_1+x_2+d_1^--d_1^+=800\\
# &\qquad 5x_1+d_2^--d_2^+=2500\\
# &\qquad 3x_2+d_3^--d_3^+=1400\\
# &\qquad x_i,d_j^-,d_j^+ \ge 0~ \text{and integer}\\
# \end{align}  

# In[4]:


from gurobipy import *
m = Model('C9Q8')

# Add Variables
x  = m.addVars(range(2), name='x',  lb=0, vtype = GRB.INTEGER)
dm = m.addVars(range(3), name='dm', lb=0, vtype = GRB.INTEGER)
dp = m.addVars(range(3), name='dp', lb=0, vtype = GRB.INTEGER)
#totSlack = m.addVar(name='totSlack')
m.update()

# Add Constraints
c1 = m.addConstr(1*x[0]  + 1*x[1]  + 1*dm[0] - 1*dp[0] == 800)
c2 = m.addConstr(5*x[0]            + 1*dm[1] - 1*dp[1] == 2500)
c3 = m.addConstr(          3*x[1]  + 1*dm[2] - 1*dp[2] == 1400)

# Add Objective Function
m.ModelSense = GRB.MINIMIZE
m.setObjectiveN(dm[0]+dp[0], index=0, priority=3)
m.setObjectiveN(dm[1], index=1, priority=2)
m.setObjectiveN(dm[2], index=2, priority=1)
m.setObjectiveN(3*dp[1]+5*dp[2], index=3, priority=0)

# Optimize Model
m.optimize()

# Output formatted solution
for v in m.getVars():
    print(v.varName, v.x)
#print('Obj :', m.objVal)


# ## Problem Example: C9Q10
# Solve the following goal programming model:
# \begin{align}
# &\text{min}\\
# &\qquad z=P_1d_1^-,5P_2d_2^-+2P_2d_3^-,P_3d_4^+\\
# &\text{s.t.}\\
# &\qquad 8x_1+6x_2+d_1^--d_1^+=480\\
# &\qquad x_1+d_2^--d_2^+=40\\
# &\qquad x_2+d_3^--d_3^+=50\\
# &\qquad d_1^++d_4^--d_4^+=20\\
# &\qquad x_i,d_j^-,d_j^+ \ge 0~ \text{and integer}\\
# \end{align}  

# In[5]:


from gurobipy import *
m = Model('C9Q10')

# Add Variables
x  = m.addVars(range(2), name='x',  lb=0, vtype = GRB.INTEGER)
dm = m.addVars(range(4), name='dm', lb=0, vtype = GRB.INTEGER)
dp = m.addVars(range(4), name='dp', lb=0, vtype = GRB.INTEGER)
#totSlack = m.addVar(name='totSlack')
m.update()

# Add Constraints
c1 = m.addConstr(8*x[0]  + 6*x[1]  + 1*dm[0] - 1*dp[0] == 480)
c2 = m.addConstr(1*x[0]            + 1*dm[1] - 1*dp[1] == 40)
c3 = m.addConstr(          1*x[1]  + 1*dm[2] - 1*dp[2] == 50)
c4 = m.addConstr(          1*dp[0] + 1*dm[3] - 1*dp[3] == 20)

# Add Objective Function
m.ModelSense = GRB.MINIMIZE
m.setObjectiveN(dm[0], index=0, priority=2)
m.setObjectiveN(5*dm[1]+2*dm[2], index=1, priority=1)
m.setObjectiveN(dp[3], index=2, priority=0)

# Optimize Model
m.optimize()

# Output formatted solution
for v in m.getVars():
    print(v.varName, v.x)
#print('Obj :', m.objVal)

