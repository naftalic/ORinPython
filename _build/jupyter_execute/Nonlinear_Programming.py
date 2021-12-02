#!/usr/bin/env python
# coding: utf-8

# # Nonlinear Programming

# Problems that fit the general linear programming format but contain nonlinear functions are termed nonlinear programming problems.
# * Solution methods are more complex than linear programming methods.
# * Determining an optimal solution is often difficult, if not impossible.
# * Solution techniques generally involve searching a solution surface for high or low points requiring the use of advanced mathematics.
# 
# * A nonlinear problem containing one or more constraints becomes a constrained optimization model or a nonlinear programming model.
# * A nonlinear programming model has the same general form as the linear programming model except that the objective function and/or the constraint(s) are nonlinear.
# * Solution procedures are much more complex and no guaranteed procedure exists for all nonlinear problem models.
# * Unlike linear programming, the solution is often not on the boundary of the feasible solution space.
# * Cannot simply look at points on the solution space boundary but must consider other points on the surface of the objective function.
# * This greatly complicates solution approaches.
# * Solution techniques can be very complex.
# 
# 
# 

# ## Nonlinear problem example: 
# Solve the following nonlinear programming model:
# \begin{align}
# &\text{max}\\
# &\qquad z=(4-0.1x_1)x_1+(5-0.2x_2)x_2\\
# &\text{s.t.}\\
# &\qquad x_1+2x_2=40\\
# &\qquad x_i \ge 0\\
# \end{align}  

# In[1]:


get_ipython().system('pip install gurobipy')

# Import gurobi library
from gurobipy import * # This command imports the Gurobi functions and classes.

import numpy as np

# create new model
m = Model('Beaver Creek Pottery Company ')
x1 = m.addVar(lb = 0, vtype = GRB.CONTINUOUS, name='x1') 
x2 = m.addVar(lb = 0, vtype = GRB.CONTINUOUS, name='x2')

m.setObjective((4-0.1*x1)*x1+(5-0.2*x2)*x2, GRB.MAXIMIZE)
m.addConstr(1*x1+2*x2==40, 'const1')

m.optimize()

#display optimal production plan
for v in m.getVars():
  print(v.varName, v.x)
print('optimal total revenue:', m.objVal)


# ## A Nonlinear Programming Model with Multiple Constraints:
#  
# Solve the following nonlinear programming model:
# \begin{align}
# &\text{max}\\
# &\qquad z=\left(\frac{1500-x_1}{24.6}-12\right)x_1
# +\left(\frac{2700-x_2}{63.8}-9\right)x_2\\
# &\text{s.t.}\\
# &\qquad 2x_1+2.7x_2\le 6000\\
# &\qquad 3.6x_1+2.9x_2\le 8500\\
# &\qquad 7.2x_1+8.5x_2\le 15000\\
# &\qquad x_i \ge 0\\
# \end{align} 

# In[3]:


get_ipython().system('pip install gurobipy')
from gurobipy import * # This command imports the Gurobi functions and classes.

# create new model
m = Model('Western Clothing Company')
x1 = m.addVar(lb = 0, vtype = GRB.CONTINUOUS, name='x1') 
x2 = m.addVar(lb = 0, vtype = GRB.CONTINUOUS, name='x2')


m.setObjective(((1500-x1)/24.6-12)*x1+((2700-x2)/63.8-9)*x2, GRB.MAXIMIZE)
m.addConstr(2*x1+2.7*x2<=6000, 'const1')
m.addConstr(3.6*x1+2.9*x2<=8500, 'const2')
m.addConstr(7.2*x1+8.5*x2<=15000, 'const3')

m.optimize()

#display optimal production plan
for v in m.getVars():
  print(v.varName, v.x)
print('optimal total revenue:', m.objVal)


# ## Problem Example: C10Q14
# Solve the following nonlinear programming model:
# \begin{align}
# &\text{max}\\
# &\qquad z=\left(15000-\frac{9000}{x_1} \right)\\
# &\qquad ~~+\left(24000-\frac{15000}{x_2} \right)\\
# &\qquad ~~+\left(8100-\frac{5300}{x_3} \right)\\
# &\qquad ~~+\left(12000-\frac{7600}{x_4} \right)\\
# &\qquad ~~+\left(21000-\frac{12500}{x_4} \right)\\
# &\text{s.t.}\\
# &\qquad x_1+x_2+x_3+x_4+x_5 \le 15\\
# &\qquad 355x_1+540x_2+290x_3+275x_4+490x_5 \le 6500\\
# &\qquad x_i \ge 0,~\text{and integer}\\
# \end{align} 

# In[4]:


from gurobipy import * 
import numpy as np

m = Model('C10Q14')
m.params.NonConvex = 2

x  = m.addVars(range(5), name='x',  lb=1, vtype = GRB.INTEGER)
u  = m.addVars(range(5), name='u',  lb=0, vtype = GRB.CONTINUOUS)

m.setObjective((15000 -9000*u[0])+(24000-15000*u[1])+
               (8100  -5300*u[2])+(12000 -7600*u[3])+
               (21000-12500*u[4]), GRB.MAXIMIZE)
m.addConstr(x[0]+x[1]+x[2]+x[3]+x[4]<=15, 'const1')
m.addConstr(355*x[0]+540*x[1]+290*x[2]+275*x[3]+490*x[4]<=6500, 'const2')

m.addConstrs((x[i]*u[i] == 1 for i in range(5)), name='bilinear')

m.optimize()

for v in m.getVars():
  print(v.varName, v.x)
print('optimal total revenue:', m.objVal)


# ## Facility Location Problem Example: C10Q19
# Solve the following nonlinear programming model:
# \begin{align}
# &\text{min}\\
# &\qquad z=7000\sqrt{(1000-x)^2+(1250-y)^2}\\
# &\qquad ~~+9000\sqrt{(1500-x)^2+(2700-y)^2}\\
# &\qquad ~+11500\sqrt{(2000-x)^2+(700-y)^2}\\
# &\qquad ~~+4300\sqrt{(2200-x)^2+(2000-y)^2}\\
# \\
# &\text{s.t.}\\
# &\qquad x, y \ge 0\\
# \end{align} 

# In[5]:


import gurobipy as gp
from gurobipy import GRB

# create new model
m = gp.Model('C10Q19')
m.params.NonConvex = 2

x  = m.addVars(range(2), name='x',  lb=0, vtype = GRB.CONTINUOUS)
u  = m.addVars(range(4), name='u',   vtype = GRB.CONTINUOUS)
v  = m.addVars(range(4), name='v',   vtype = GRB.CONTINUOUS)

m.setObjective(7000*v[0]+9000*v[1]+11500*v[2]+4300*v[3], GRB.MINIMIZE)
m.addConstr( (x[0]-1000)*(x[0]-1000)+(x[1]-1250)*(x[1]-1250) == u[0])
m.addConstr( (x[0]-1500)*(x[0]-1500)+(x[1]-2700)*(x[1]-2700) == u[1])
m.addConstr( (x[0]-2000)*(x[0]-2000)+(x[1]-700) *(x[1]-700)  == u[2])
m.addConstr( (x[0]-2200)*(x[0]-2200)+(x[1]-2000)*(x[1]-2000) == u[3])

m.addConstrs((u[i] == v[i]*v[i] for i in range(4)), name='v_squared')

m.optimize()

#display optimal production plan
for w in m.getVars():
  print(w.varName, w.x)
print('optimal total revenue:', m.objVal)


# In[6]:


import numpy as np
x=1658.8;y=1416.7;
d=7000*np.sqrt((x-1000)**2+(y-1250)**2)+9000*np.sqrt((x-1500)**2+(y-2700)**2)+11500*np.sqrt((x-2000)**2+(y-700)**2)+4300*np.sqrt((x-2200)**2+(y-2000)**2)
print(x,y,d)

x=1665.4;y=1562.9;
d=7000*np.sqrt((x-1000)**2+(y-1250)**2)+9000*np.sqrt((x-1500)**2+(y-2700)**2)+11500*np.sqrt((x-2000)**2+(y-700)**2)+4300*np.sqrt((x-2200)**2+(y-2000)**2)
print(x,y,d)

