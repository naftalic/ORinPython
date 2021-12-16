#!/usr/bin/env python
# coding: utf-8

# # Linear Algebra Review
# 
# [Colab notebook](https://colab.research.google.com/drive/1T5Tg-xY9y3cVS6HuFw_D8814H4oYrP-8?usp=sharing)
# 

# ## Scalars, Vectors, Matrices
# 
# Values consisting of one numerical quantity are called **scalars**. 
# For example, the average annual temperature in New York City is $53^\circ$F. 
# In the following, we adopt the mathematical notation where **scalar** variables are denoted by ordinary lower-cased letters (e.g., $x, y,$ and $z$). 
# 
# We denote the space of real-valued scalars by $\mathbb{R}$. Formally, $x \in \mathbb{R}$ means that $x$ is a real-valued scalar, where the symbol $\in$ is pronounced "in," and simply denotes membership in a set. 
# 
# 
# 

# In[1]:


a = 1 #scalar
b = 2.5 #scalar
a, b


# **Vector** is a list (a.k.a., array) of scalar values.
# We denote vectors in bold, lower-case letters (e.g., $\mathbf{x}$). 
# Vectors can be presented either as a **column** or as a **row**.
# In example,
# 
# $$
# \mathbf{x} =
# \begin{bmatrix}
# x_{1}\\ x_{2} \\ \vdots \\x_{n}
# \end{bmatrix}
# \quad\text{or}\quad
# \mathbf{x}^\top = \begin{bmatrix}x_1 & x_2 & ... & x_n\end{bmatrix}.
# $$
# 
# We refer to each element in a vector using a subscript.
# For example, the $i^\mathrm{th}$ element of $\mathbf{x}$ is denoted as $x_i$ and is a scalar.
# Formally, we say that a vector $\mathbf{x}$
# consists of $n$ real-valued scalars, $\mathbf{x} \in \mathbb{R}^n$.
# The length of a vector is commonly called the **dimension** of the vector.
# 
# Vectors can be interpreted in multilpe ways. One interpretation is that vectors symbolize the location of points in space relative to some reference location which is called the **origin**. This view is useful when separating points to distinct clusters of points in space.
# Corresponding interpretation is as a **directions** in space from some reference location. This view allows us to operate on vectors: adding, subtructing, multiplying, etc.
# 

# In[2]:


import numpy as np
x = np.array([1, 2, 3, 4]) #vector, 1-d array
x


# In[3]:


len(x), x.shape


# **Matrices** concatenate vectors similar to the way that vectors concatenate scalars.
# Here, matrices are denoted with bold, capital letters (e.g., $\mathbf{A}$).
# Formally, $\mathbf{A} \in \mathbb{R}^{m \times n}$ means that the matrix $\mathbf{A}$ consists of $m$ rows and $n$ columns of real-valued scalars.
# Visually, we can illustrate the matrix $\mathbf{A}$ as a table,
# where each element $a_{ij}$ belongs to the $i^{\mathrm{th}}$ row and $j^{\mathrm{th}}$ column. That is,
# 
# $$\mathbf{A}=
# \begin{bmatrix} 
# a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ 
# \end{bmatrix}.$$
# 
# The shape of $\mathbf{A}$ is $m \times n$.
# When a matrix has the same number of rows and columns it is called a **square** matrix.

# In[4]:


A = np.array([[1, 2], [3, 4], [5, 6]]) #3x2 matrix 
A


# In[5]:


A.shape


# In[6]:


A[2,1]


# We can exchange a matrix's rows and columns and the result is called the **transpose** of the matrix.
# Formally, we denote the transpose of $\mathbf{A}$ by $\mathbf{A}^\top$.
# Thus, the transpose of $\mathbf{A}$ is
# a $n \times m$ matrix.
# 

# In[7]:


A_T = A.T # Vector transposition
A_T


# In[8]:


A_T.shape


# As a special type of a square matrix is the **symmetric** matrix $\mathbf{A}$ in which its transpose is equal to itself. That is, $\mathbf{A}=\mathbf{A}^\top $.  

# In[9]:


A=np.array([[1,5,0],[5,2,0],[0,0,3]])
(A == A.T).all()


# Matrices are useful data structures, and they allow us to organize data efficiently. As an example, it is common to organize data in a way that the **rows** of a matrix correspond to different measurments, while the **columns** correspond to the various features or attributes. 

# ## Arithmetic Operations
# 
# Adding/subtracting scalars, vectors, or matrices of the same shape is performed **elementwise**. It is also possible to multiply/divide scalars, vectors, or matrices elementwise.

# In[10]:


A = np.array([[1, 2], [3, 4], [5, 6]])
A


# In[11]:


B = np.array([[2, 5], [7, 4], [4, 3]])
B


# In[12]:


A+B # elementwise addition


# In[13]:


A-B # elementwise subtraction


# In[14]:


A*B # elementwise multiplication, aka Hadamard product


# In[15]:


A/B # elementwise division


# ## Reduction
# A common operation that can be performed on arbitrary matrices
# is to **sum** elements.
# Formally, we express sums using the $\sum$ symbol.
# To express the sum of the elements in a vector $\mathbf{x}$ of length $n$,
# we write $\sum_{i=1}^n x_i$.
# 

# In[16]:


x = np.arange(10)
x, x.sum()


# In[17]:


A = np.array([[1, 2], [3, 4], [5, 6]])


# The sum of **all** the elements in $\mathbf{A}$ is denoted $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.

# In[18]:


A.sum()


# Summing up elements along **rows** reduces the row dimension (axis 0) $\sum_{i=1}^{m} a_{ij}$.

# In[19]:


A.sum(axis=0)


# Summing up elements along **columns** reduces the row dimension (axis 1) $\sum_{j=1}^{n} a_{ij}$.

# In[20]:


A.sum(axis=1)


# ## Broadcasting

# Broadcasting describes how numpy handles arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is "broadcast" across the larger array so that they have compatible shapes. 
# NumPy operations are done on **pairs** of arrays on an element-by-element basis. 
# 
# The simplest broadcasting example occurs when an array and a scalar value are combined in an operation

# In[21]:


A = np.array([1, 2,3])
A+4 # broadcasting


# This result is equivalent to 

# In[22]:


A + np.array([4, 4, 4])


# When operating on two arrays, NumPy compares their shapes element-wise. It starts with the **rightmost** dimensions and works its way left. Two dimensions are compatible when
# 
# * they are equal, or
# * one of them is 1.
# 
# If these conditions are not met, a ValueError is thrown. Here are a few more examples:
# 
# 
# Ex 1.  | Shape  | Ex 2.  | Shape      | Ex 3.  | Shape         |
# --     | --     | --     | --         |  --    | --            | 
# A      | 3 x 2  | A      | 10 x 3 x 3 | A      | 6 x 1 x 4 x 1 | 
# B      | 1 x 2  | B      | 10 x 1 x 3 | B      | 1 x 5 x 1 x 3 | 
# A+B    | 3 x 2  | A+B    | 10 x 3 x 3 | A+B    | 6 x 5 x 4 x 3 | 
# 
# 
# 
# 
# 

# ## Multiplying Vectors and Matrices
# 
# One of the fundamental operations in Linear Algebra is the **dot product**.
# Given two vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$, their dot product $\mathbf{x} \cdot \mathbf{y}=\mathbf{x}^\top \mathbf{y}$ is a sum over the elements at the same positions. In example, $\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{n} x_i y_i$ resulting in a scalar.

# In[23]:


x = np.ones(4); y = 2*np.ones(4)
x, y, np.dot(x, y)


# The dot product has a geometric interpretation related to the **angle** between the two vectors
# 
# $$
# \mathbf{x}\cdot\mathbf{y} = \mathbf{x}^\top\mathbf{y} = \|\mathbf{x}\|\|\mathbf{y}\|\cos\theta,
# $$
# 
# where $\| \cdot \|$ is the **norm** operator applied to tha vector. Norms will be discussed later on. 
# 
# In the Machine Learning contexts, the angle $\theta$ is often used to measure the closeness, or similarity, of two vectors. This can be computed as
# 
# $$
# \cos\theta = \frac{\mathbf{x}^\top\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}.
# $$
# 
# This (cosine) similarity takes values between -1 and 1. A maximum value of $1$
# when the two vectors point in the same direction, a minimum value of $-1$ when pointing in opposite directions, and a value of $0$ when orthogonal.

# In[24]:


np.arccos(x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y)))


# Consider now the dot product of a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ and a vector $\mathbf{x} \in \mathbb{R}^n$.
# To follow with the above dot-product definition we write the matrix $\mathbf{A}$ in terms of its **row** vectors
# 
# $$\mathbf{A}=
# \begin{bmatrix}
# \mathbf{a}^\top_{1} \\
# \mathbf{a}^\top_{2} \\
# \vdots \\
# \mathbf{a}^\top_m \\
# \end{bmatrix},$$
# 
# where each $\mathbf{a}^\top_{i} \in \mathbb{R}^n$
# is a **row** vector of length $n$, representing the $i^\mathrm{th}$ row of the matrix $\mathbf{A}$.
# Now, the matrix-vector dot product $\mathbf{A}\mathbf{x}$
# is simply a column vector of length $m$,
# whose $i^\mathrm{th}$ element is the dot product $\mathbf{a}^\top_i \cdot \mathbf{x}=\mathbf{a}_i \mathbf{x}$
# 
# $$
# \mathbf{A}\cdot\mathbf{x}
# = \begin{bmatrix}
# \mathbf{a}^\top_{1} \\
# \mathbf{a}^\top_{2} \\
# \vdots \\
# \mathbf{a}^\top_m \\
# \end{bmatrix}\cdot\mathbf{x}
# = \begin{bmatrix}
#  \mathbf{a}^\top_{1}\cdot \mathbf{x}  \\
#  \mathbf{a}^\top_{2}\cdot \mathbf{x} \\
# \vdots\\
#  \mathbf{a}^\top_{m}\cdot \mathbf{x}\\
# \end{bmatrix}
# = \begin{bmatrix}
#  \mathbf{a}_{1} \mathbf{x}  \\
#  \mathbf{a}_{2} \mathbf{x} \\
# \vdots\\
#  \mathbf{a}_{m} \mathbf{x}\\
# \end{bmatrix}
# = \begin{bmatrix}
#  b_{1} \\
#  b_{2} \\
# \vdots\\
#  b_{m}\\
# \end{bmatrix}.
# $$
# 
# To make it clear, in the above example there are $m$ dot products, each $\mathbf{a}^\top_i \mathbf{x}$ which is a multiplcation of an $n$-sized $\mathbf{a}^\top_i$ by an $n$-sized $\mathbf{x}$. The result of each dot product is a scalar.
# 
# We can think of the multiplication of $\mathbf{A}\in \mathbb{R}^{m \times n}$ by $\mathbf{x}\in \mathbb{R}^n$ as a transformation that projects vectors, $\mathbf{x}\in \mathbb{R}^n$,
# from $\mathbb{R}^{n}$ to $\mathbb{R}^{m}$.
# Note that the column dimension of $\mathbf{A}$ must be the same as the dimension of $\mathbf{x}$.
# 

# In[25]:


A = np.array([[1, 2], [3, 4], [5, 6]])
x = np.array([1,1])
np.dot(A, x)


# Whereas this is computation results in broadcasting 

# In[26]:


A*x


# Next we consider matrix-matrix multiplication.
# Given two matrices $\mathbf{A} \in \mathbb{R}^{m \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times n}$:
# 
# $$\mathbf{A}=\begin{bmatrix}
#  a_{11} & a_{12} & \cdots & a_{1k} \\
#  a_{21} & a_{22} & \cdots & a_{2k} \\
# \vdots & \vdots & \ddots & \vdots \\
#  a_{m1} & a_{m2} & \cdots & a_{mk} \\
# \end{bmatrix},\quad
# \mathbf{B}=\begin{bmatrix}
#  b_{11} & b_{12} & \cdots & b_{1n} \\
#  b_{21} & b_{22} & \cdots & b_{2n} \\
# \vdots & \vdots & \ddots & \vdots \\
#  b_{k1} & b_{k2} & \cdots & b_{kn} \\
# \end{bmatrix}.$$
# 
# 
# Denote by $\mathbf{a}^\top_{i} \in \mathbb{R}^k$
# the row vector representing the $i^\mathrm{th}$ row of the matrix $\mathbf{A}$,
# and let $\mathbf{b}_{j} \in \mathbb{R}^k$
# be the column vector of the $j^\mathrm{th}$ column of the matrix $\mathbf{B}$.
# The dot matrix-matrix product $\mathbf{C} = \mathbf{A}\cdot\mathbf{B}$, is easiest to understand with $\mathbf{A}$ specified by its row vectors and $\mathbf{B}$ specified by its column vectors,
# 
# $$
# \mathbf{A}=
# \begin{bmatrix}
# \mathbf{a}^\top_{1} \\
# \mathbf{a}^\top_{2} \\
# \vdots \\
# \mathbf{a}^\top_m \\
# \end{bmatrix},
# \quad \mathbf{B}=\begin{bmatrix}
#  \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{n} \\
# \end{bmatrix}.
# $$
# 
# 
# The matrix product $\mathbf{C}\in \mathbb{R}^{m \times n}$ is produced by computing each element $c_{ij}$ as the dot product $\mathbf{a}^\top_i \mathbf{b}_j$:
# 
# $$
# \begin{align}
# \mathbf{C} &= \mathbf{A\cdot B} \\
# &= \begin{bmatrix}
# \mathbf{a}^\top_{1} \\
# \mathbf{a}^\top_{2} \\
# \vdots \\
# \mathbf{a}^\top_m \\
# \end{bmatrix}
# \cdot
# \begin{bmatrix}
#  \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{n} \\
# \end{bmatrix}\\
# &= \begin{bmatrix}
# \mathbf{a}^\top_1\cdot \mathbf{b}_1 & \mathbf{a}^\top_1\cdot\mathbf{b}_2& \cdots & \mathbf{a}^\top_1\cdot \mathbf{b}_n \\
#  \mathbf{a}^\top_2\cdot\mathbf{b}_1 & \mathbf{a}^\top_2\cdot \mathbf{b}_2 & \cdots & \mathbf{a}^\top_2\cdot \mathbf{b}_n\\
#  \vdots & \vdots & \ddots &\vdots\\
# \mathbf{a}^\top_m\cdot \mathbf{b}_1 & \mathbf{a}^\top_m\cdot\mathbf{b}_2& \cdots& \mathbf{a}^\top_m\cdot \mathbf{b}_n
# \end{bmatrix}\\
# &= \begin{bmatrix}
# \mathbf{a}_1\mathbf{b}_1 & \mathbf{a}_1\mathbf{b}_2& \cdots & \mathbf{a}_1 \mathbf{b}_n \\
#  \mathbf{a}_2\mathbf{b}_1 & \mathbf{a}_2 \mathbf{b}_2 & \cdots & \mathbf{a}_2 \mathbf{b}_n\\
#  \vdots & \vdots & \ddots &\vdots\\
# \mathbf{a}_m \mathbf{b}_1 & \mathbf{a}_m\mathbf{b}_2& \cdots& \mathbf{a}_m \mathbf{b}_n
# \end{bmatrix}.
# \end{align}
# $$
# 
# Again, $\mathbf{a}_1\mathbf{b}_1$ is the dot product of the 1st row in $\mathbf A$ by the 1st column in $\mathbf B$, etc.
# 

# It is common notation practice to *omit* the dot while writing $\mathbf A \cdot \mathbf x$ or $\mathbf A \cdot \mathbf B$.

# In[27]:


A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[5, 3], [2, 2]])
A.dot(B) 


# Matrix mutliplication is **distributive**, i.e. $\mathbf{A}(\mathbf{B}+\mathbf{C})=\mathbf{A}\mathbf{B}+\mathbf{A}\mathbf{C}$:

# In[28]:


A = np.array([[2, 3], [1, 4], [7, 6]])
B = np.array([[5], [2]])
C = np.array([[4], [3]])
(A.dot(B+C) == A.dot(B)+A.dot(C)).all()


# Matrix mutliplication is **associative**, i.e. $\mathbf{A}(\mathbf{B}\mathbf{C})=(\mathbf{A}\mathbf{B})\mathbf{C}$:

# In[29]:


A = np.array([[2, 3], [1, 4], [7, 6]])
B = np.array([[5, 3], [2, 2]])
C = np.array([[4], [3]])
(A.dot(B.dot(C)) == (A.dot(B)).dot(C)).all()


# Vector multiplication is commutative, i.e. $\mathbf{x}^\top\mathbf{y}=\mathbf{y}^\top\mathbf{x}$

# In[30]:


x = np.array([[2], [6]])
y = np.array([[5], [2]])
(x.T.dot(y) == y.T.dot(x)).all()


# However, matrix multiplication is **not commutative**, i.e. $\mathbf{A}\mathbf{B}\ne\mathbf{B}\mathbf{A}$:

# In[31]:


A = np.array([[2, 3], [6, 5]])
B = np.array([[5, 3], [2, 2]])
(np.dot(A, B) == np.dot(B, A)).all()


# ## Norms
# 
# One of the important operators in linear algebra are the **norms**.
# The norm of a vector tells you how large a vector is by maping a vector
# to a scalar. This scalar measures, not the dimensions, but the magnitude of the components.
# Formally, given any vector $\mathbf{x}$, the vector norm is a function $f$ that has multiple propeerties: 
# * The first is a scaling property that says
# that if we scale all the elements of a vector
# by a constant factor $\alpha$,
# its norm also scales by the **absolute value**
# of the same constant factor. In example,
# 
# $$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$
# 
# * The second property is the triangle inequality:
# 
# $$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$
# 
# * The third property simply says that the norm must be non-negative:
# 
# $$f(\mathbf{x}) \geq 0.$$
# 
# The final property requires that the smallest norm is achieved if and only if 
# by a vector consisting of all zeros.
# 
# $$\forall i, \mathbf{x}_i = 0 \Leftrightarrow f(\mathbf{x})=0.$$
# 
# The familiar **Euclidean distance** is a norm, typically named 
# the $L_2$ norm, and is defined as square root of the sum of vector elements' squares:
# 
# $$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$
# 

# In[32]:


import numpy as np
u = np.array([1,2,3]) # L2 norm
np.linalg.norm(u, ord=2)


# The $L_1$ norm is defined as the sum of the absolute values of the vector elements:
# $$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$

# In[33]:


u = np.array([1,2,3])
np.linalg.norm(u, ord=1) # L1 norm


# In more generality, the $L_p$ norm is defined as 
# $$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

# In[34]:


u = np.array([1,2,3])
np.linalg.norm(u, ord=np.inf) # L-inf norm, basically choosing the maximum value within the array


# ## Identity matrices
# 
# The identity matrix of size n is the $n \times n$ square matrix with ones on the main diagonal and zeros elsewhere. It is denoted by $\mathbf I_n$.

# In[35]:


A = np.eye(3)
A


# In[36]:


x = np.array([[2], [6], [3]])
x


# In[37]:


b = A.dot(x)
b


# ## Inverse matrices
# 
# An $n\times n$ square matrix $\mathbf A$ is called invertible (also nonsingular or nondegenerate), if there exists an $n\times n$ square matrix $\mathbf B$ such that
# $$
# \mathbf {AB} =\mathbf {BA} =\mathbf {I} _{n}.
# $$
# 
# In this case, the matrix $\mathbf B$ is uniquely determined by $\mathbf A$, and is called the inverse of $\mathbf A$, denoted by $\mathbf A^{-1}$.
# A square matrix that is not invertible is called singular or degenerate. 
# This topic will be discussed further in later sections.

# In[38]:


A = np.array([[3, 0, 2], [2, 0, -2], [0, 1, 1]])
A


# In[39]:


A_inv = np.linalg.inv(A)
A_inv


# In[40]:


A_inv.dot(A)


# ## Hyperplanes

# 
# 
# A key object in linear algebra is the **hyperplane**.
# In two dimensions the hyperplane is a **line**, and in three dimensions it is a **plane**. In more generality, in an $d$-dimensional vector space, a hyperplane has $d-1$ dimensions
# and **divides the space into two half-spaces**. 
# In the Machine Learning language in the contex of classification hyperplanes that separate the space are named **decision planes**. 
# 
# In example, 
# * if $\mathbf{a}=(1,2)$ the equation $\mathbf{a} \cdot \mathbf{b}=1$ defines the **line** $2y = 1 - x$ in the **two dimensional** space, while
# * if $\mathbf{a}=(1,2,3)$ and $\mathbf{a} \cdot \mathbf{b}=1$ defines the **plane** $ z= 1 - x-2y$ in the **three dimensional** space.
# 
# 
# 

# ## Solving a system of linear equations
# 
# The set of equations
# 
# $$
# \mathbf A \mathbf x = \mathbf b
# $$
# 
# where $\mathbf{A}\in \mathbb{R}^{m \times n}$, $\mathbf{x}\in \mathbb{R}^n$, and $\mathbf{b}\in \mathbb{R}^m$, has the solution
# 
# $$
# \mathbf A^{-1} \mathbf A \mathbf x =\mathbf I_n\mathbf x =\mathbf x=\mathbf A^{-1}  \mathbf b
# $$
# 
# given that $\mathbf A^{-1}$ exists.
# 

# with $m=n=2$:

# In[41]:


A = np.array([[2, -1], [1, 1]])
b = np.array([0, 3])
x = np.linalg.inv(A).dot(b)
x


# with $m=3$, and $n=2$ we use a **pseudoinverse**:

# In[42]:


A = np.array([[2, -1], [1, 1], [1, 1]])
b = np.array([0,3,0])
x = np.linalg.pinv(A).dot(b)
x


# ## Linear equations

# The set of equations
# 
# $$\mathbf{A}\mathbf{x}=\mathbf{b}$$
# 
# correspond to an $m$ equations with $n$ unknowns
# 
# $$
# \begin{align}
# a_{1,1}x_1 + a_{1,2}x_2 + \cdots + a_{1,n}x_n = b_1 \\\\
# a_{2,1}x_1 + a_{2,2}x_2 + \cdots + a_{2,n}x_n = b_2 \\\\
# \cdots \\\\
# a_{m,1}x_1 + a_{m,2}x_2 + \cdots + a_{m,n}x_n = b_m
# \end{align}
# $$
# 
# or
# 
# $$
# \begin{align}
# \begin{bmatrix}
#     a_{1,1} & a_{1,2} & \cdots & a_{1,n} \\\\
#     a_{2,1} & a_{2,2} & \cdots & a_{2,n} \\\\
#     \cdots & \cdots & \cdots & \cdots \\\\
#     a_{m,1} & a_{m,2} & \cdots & a_{m,n}
# \end{bmatrix}
# \begin{bmatrix}
#     x_1 \\\\
#     x_2 \\\\
#     \cdots \\\\
#     x_n
# \end{bmatrix}
# =
# \begin{bmatrix}
#     b_1 \\\\
#     b_2 \\\\
#     \cdots \\\\
#     b_m
# \end{bmatrix}.
# \end{align}
# $$
# 
# Commonly, we are given $\mathbf{A}$ and $\mathbf{b}$ as inputs, and we solve for $\mathbf{x}$ that satisfy the above equations.
# 
# How many solutions does $\mathbf{A}\mathbf{x}=\mathbf{b}$ have? There are the three possibilites:
# 1. No solution
# 2. One solution
# 3. Infinite number of solutions
#  
# Why can't there be more than one solution and less than an infinite number of solutions?
# 
# On the intuition level, it is because we are working with linear equations.
# It is clear if we consider the simple scenario of two equations and two unknowns: the solution of this system corresponds to the intersection of lines. 
# 1. One possibility is that the two lines never cross (parallel). 
# 2. Another possibility is that they cross exactly once. 
# 3. And the last possibility is that they cross everywhere (superimposed).

# There are two ways to describe a system of linear equations: the **row view** and the **column view**.

# ## Row view
# 
# The row view is probably more intuitive than the column view.
# We mentioned that the solution of a linear system of equations are the sets of values of $\mathbf x = (x_1, x_2, ... x_n)$ that satisfy all equations. For instance, in the case of $\mathbf{A}\in \mathbb{R}^{m \times n}$ with $m=n=2$ the equations correspond to 2 lines in the 2-dimensional space and the solution of the system is the intersection of these lines.
# 
# As mentioned before, a linear system can be viewed as a set of $(n-1)$-dimensional hyperplanes in a $n$-dimensional space. The system can also be characterized by its number of equations $m$, and the number of unknown variables $n$.
# 
# * If $m \gt n$, there are more equations than unknowns, the system is called **overdetermined**. For example, consider a system of 3 equations (represented by 3 lines) and 2 unknowns (corresponding to 2 dimensions). If the equations (or lines) are independent in this system, there is no solution since there is no point that belongs to the three lines.
# * If $m \lt n$, there are more unknowns than equations, and the system is termed **underdetermined**. For example, consider a system of 1 equation (1 line) and 2 dimensions. In this case, each point along the line is a solution to the system, resulting in an infinite number of solutions.
# 
# Let's look at these examples in more depth:
# 

# ### Example 1: 1 equation and 2 variables
# 
# Let us start with an **underdetermined** example with  $m=1$ and  $n=2$:
# 
# $$
# a_{1,1}x_1 + a_{1,2}x_2 = b_1
# $$
# 
# The graphical interpretation of $n=2$ is that we have a 2-dimensional space represented visually by 2 axes. Since the hyperplane is of $n-1$-dimensional, we have a 1-dimensional hyperplane which is simply a line. As $m=1$, we have only one equation. This means that we have only one line characterizing the linear system.
# 
# Note that the last equation can also be written in a way that may be more common:
# 
# $$
# x_2 = \frac{a_{1,1}}{a_{1,2}}x_1 +\frac{b_1}{a_{1,2}}b_1,\quad\text{or}\quad
# y = \frac{a_{1,1}}{a_{1,2}}x +\frac{b_1}{a_{1,2}}b_1,
# $$
# 
# where $y=x_2$, and $x=x_1$.
# 
# The solutions of this linear system correspond to ALL the values of $x$ and $y$ such that $y = \frac{a_{1,1}}{a_{1,2}}x +\frac{b_1}{a_{1,2}}b_1$ holds. Practicly, it corresponds to each and every point along the line so there are  infinite number of solutions.
# 

# ### Example 2: 2 equation and 2 variables
# 
# Let us explore an example with  $m=2$ and  $n=2$:
# 
# $$
# a_{1,1}x_1 + a_{1,2}x_2 = b_1\\
# a_{2,1}x_1 + a_{2,2}x_2 = b_2.
# $$
# 
# The graphical interpretation of $n=2$ is as before. That is, we have a system in the 2-dimensional space, represented visually with 2 axes, and each equation represents a $1$-dimensional hyperplane which is simply a line. 
# 
# But, here $m=2$, which means that we have two equations or 2 lines in the 2-dimensional space.
# Now, there are multiple scenarios:
# 
# * The two lines run in parallel, and there is no solution,
# * The two lines intersect, and there is a unique solution, or
# * The two lines superimpose representing the same equation (or line). In this case, there is an infinite number of solutions since each point along the lines correspond to a solution.
# 
# <!-- Same conclusion can be drawn with higher values of $m$ (number of equations) and $n$ (number of dimensions). For example, two 2-dimensional planes in a 3-dimensional space can be 
# * parallel (no solution),
# * cross (infinitely many solutions since their crossing is a line), or
# * superimposed (infinitely many solutions).  -->
# 
# 

# ### Example 3: 3 equation and 2 variables
# 
# Let us explore an **overdetermined** example with  $m=3$ and  $n=2$:
# 
# $$
# a_{1,1}x_1 + a_{1,2}x_2 = b_1\\
# a_{2,1}x_1 + a_{2,2}x_2 = b_2\\
# a_{3,1}x_1 + a_{3,2}x_2 = b_3
# $$
# 
# The graphical interpretation of $n=2$ is as before. That is, we have a system in the 2-dimensional space, represented by 2 axes, and each equation represents an $1$-dimensional hyperplane which, as before, is simply a line. 
# 
# Here $m=3$, which means that we have three equations, or 3 lines in a 2-dimensional space.
# 
# There are multiple scenarios:
# * The three lines run in parallel, and there is no solution,
# * The three lines intersect. If the three lines are independent, there are multiple intersections with no unique solution,
# * Two lines are dependent, and the third is independent: we have a unique solution at the intersection of the dependent's two with the third line, or
# * The three lines superimpose: there is an infinite number of solutions.
# 
# These examples can be generalized to higher spaces.

# ## Linear combination
# 
# The linear combination of 2 vectors corresponds to their weighted sum.
# In example:
# 
# $$
# \mathbf{x}^\top=(1,2,3),~\mathbf{y}^\top=(4,5,6),
# \quad \text{and}
# \quad \mathbf{z}^\top=a\mathbf{x}^\top+b\mathbf{y}^\top,\text{with}~ a=1,~b=2.
# $$
# 

# In[43]:


a=1.;b=2.;
a*np.array([1,2,3])+b*np.array([4,5,6])


# ## Span

# 
# 
# Consider the above $\mathbf{x}$ and $\mathbf{y}$: ALL the points that can be reached by their combination via some $a$ and $b$ are called the set of points is the **span** of the vectors $\{\mathbf{x}$, $\mathbf{y}\}$.
# Vector **space** means all the points that can be reached by this vector. In example, 
# * the space of all real numbers is $\mathbb{R}$, and
# * for $n$-dimensions the space is $\mathbb{R}^n$. 
# 
# Vector **subspace** means that the space occupies only part of the space. In example, 
# * a 2-dimensional plane is a subspace in $\mathbb{R}^3$, and 
# * a line is a subspace in $\mathbb{R}^2$.
# 
# In general, we will say that a collection of vectors
# $\mathbf{x}_1,\mathbf{x}_2, \ldots, \mathbf{x}_n$ are linearly **dependent**
# if there exist coefficients $a_1,a_2 \ldots, a_n$, not all zeros, such that
# 
# $$
# \sum_{i=1}^n a_i\mathbf{x_i} = 0.
# $$
# 
# In this case, we can solve for one of the vectors in terms of a combination of the others, and thus this vector is **redundant**.
# Similarly, a linear dependence in the columns of a matrix means that this matrix can be **compressed** down to a lower dimension. 
# 
# If there is no linear dependence, we say the vectors are **linearly independent**. In a matrix, if the columns are linearly independent, the matrix cannot be compressed to a lowed dimension.
# 
# A set of vectors that is linearly **independent** and **spans** some vector space forms a **basis** for that vector space. 
# The **standard basis** (also called natural basis or canonical basis) in $\mathbb{R}^n$ is the set of vectors whose components are all zero, except one that equals 1. For example, the standard basis in $\mathbb{R}^2$ is formed by the vectors $\mathbf{x}^\top=(1,0)^\top$ and $\mathbf{y}^\top=(0,1)^\top$.
# 
# The linear combination of vectors in a space stays in the same space. For instance, any linear combination of two lines in a $\mathbb{R}^2$ results in another vector in $\mathbb{R}^2$.

# ## Column view
# 
# It is also possible to represent the solution $\mathbf{b}$ to the equations $\mathbf{A}\mathbf{x}=\mathbf{b}$ as a linear combination of the columns in $\mathbf{A}$.
# 
# To see that, the set of equations
# 
# $$
# a_{1,1}x_1 + a_{1,2}x_2 + a_{1,n}x_n = b_1 \\\\
# a_{2,1}x_1 + a_{2,2}x_2 + a_{2,n}x_n = b_2 \\\\
# \cdots \\\\
# a_{m,1}x_1 + a_{m,2}x_2 + a_{m,n}x_n = b_m
# $$
# 
# can be written by grouping the columns in $\mathbf{A}$
# 
# $$
# \begin{align}
# x_1
# \begin{bmatrix}
#     a_{1,1}\\\\
#     a_{2,1}\\\\
#     a_{m,1}
# \end{bmatrix}
# +
# x_2
# \begin{bmatrix}
#     a_{1,2}\\\\
#     a_{2,2}\\\\
#     a_{m,2}
# \end{bmatrix}
# +\cdots+
# x_n
# \begin{bmatrix}
#     a_{1,n}\\\\
#     a_{2,n}\\\\
#     a_{m,n}
# \end{bmatrix}
# =
# \begin{bmatrix}
#     b_1\\\\
#     b_2\\\\
#     b_m
# \end{bmatrix}.
# \end{align}
# $$
# 
# In this view, the solution $\mathbf{b}$ is a linear combination of the columns of $\mathbf{A}$, weighted by the components of $\mathbf{x}$.

# 
# * For the **overdetermined** system $m\gt n$ and there is **no solution**. For example, in the column view,
# 
# $$
# \begin{align}
# a_{1,1}x_1+a_{1,2}x_2=b_1\\
# a_{2,1}x_1+a_{2,2}x_2=b_2\\
# a_{3,1}x_1+a_{3,2}x_2=b_3\\
# \end{align}
# \quad\text{corresponds to}\quad\
# x_1
# \begin{bmatrix}
#     a_{1,1}\\
#     a_{2,1}\\
#     a_{3,1}\\
# \end{bmatrix}
# +x_2
# \begin{bmatrix}
#     a_{1,2}\\
#     a_{2,2}\\
#     a_{3,2}
# \end{bmatrix}
# =
# \begin{bmatrix}
#     b_{1}\\
#     b_{2}\\
#     b_{3}
# \end{bmatrix}.
# $$
# 
# In the row view, we have 3 lines, and we are looking for a unique intersection in the 2-dimensional plane. The three lines (if independent) will intersect in multiple points resulting in no solution. In the column view, because $n=2$ the linear combination of two 3-dimensional vectors is not enough to span the 3-dimensional space, unless the vector $\mathbf{b}$ lies, for some reason, in the subspace formed by these two vectors.
# 
# * For the **underdetermined** system, $m \lt n$ and the system has **infinite number of solutions**. As an example in the column view,
# 
# $$
# \begin{align}
# a_{1,1}x_1+a_{1,2}x_2+a_{1,3}x_3=b_1\\
# a_{2,1}x_1+a_{2,2}x_2+a_{2,3}x_3=b_2
# \end{align}
# \quad\text{corresponds to}\quad\
# x_1
# \begin{bmatrix}
#     a_{1,1}\\
#     a_{2,1}
# \end{bmatrix}
# +x_2
# \begin{bmatrix}
#     a_{1,2}\\
#     a_{2,2}
# \end{bmatrix}
# +x_3
# \begin{bmatrix}
#     a_{1,3}\\
#     a_{2,3}
# \end{bmatrix}
# =
# \begin{bmatrix}
#     b_{1}\\
#     b_{2}
# \end{bmatrix}.
# $$
# 
# But, if the columns of $\mathbf{A}$ are independent, two of them are enough to reach any point in $\mathbb{R}^2$. So, two components out of the three in $\mathbf{x}$ are used to determine the solution while the third is free, meaning that there is an infinite number of solutions.
# 

# ## Linear dependency
# 
# The number of **columns** can thus provide information on the number of solutions. But the number that we have to take into account is the number of linearly **independent** columns. Columns are linearly **dependent** if one of them is a linear combination of the others. In the column view, the direction of two linearly dependent vectors is the same, and this doesn't add value in spanning the space.
# 
# As an example,
# 
# $$
# \begin{align}
# x_1+2x_2=b_1\\
# 2x_1+4x_2=b_2
# \end{align}
# $$
# 
# which in the column view is
# 
# $$
# \begin{align}
# x_1
# \begin{bmatrix}
#     1 \\
#     2
# \end{bmatrix}
# +
# x_2
# \begin{bmatrix}
#     2 \\
#     4
# \end{bmatrix}
# =
# \begin{bmatrix}
#     b_1 \\
#     b_2
# \end{bmatrix}.
# \end{align}
# $$
# 
# The columns $(1,2)^\top$ and $(2,4)^\top$ are **dependent** and hence their linear combination is not enough to span the full $\mathbb{R}^2$ and reach all points in this space. If $\mathbf{b}=(3,6)^\top$ there is a solution $\mathbf{x}^\top=(1,1)^\top$ because the vector $(1,2)^\top$ spans a subspace of $\mathbb{R}^2$ that contains the vector $\mathbf{b}=(3,6)^\top$.
# But for a more general solution, as say $\mathbf{b}=(3,7)$, there is no solution as linear combinations of $(1,2)^\top$ are not enough to reach all points in $\mathbb{R}^2$. This is an example of an **overdetermined** system with $m=2\gt n=1$ because of the linear dependency between the columns.
# 

# ## Square matrix
# 
# When $\mathbf{A}\in\mathbb{R}^{m\times n}$ and $m=n$, the matrix $\mathbf{A}$ is called square matrix and if the columns are linearly **independant** there is a unique solution to $\mathbf{A}\mathbf{x}=\mathbf{b}$.
# The solution is simply $\mathbf{x}=\mathbf{A}^{-1}\mathbf{b}$, where $\mathbf{A}^{-1}$ is the inverse of $\mathbf{A}$.
# 
# When $\mathbf{A}^{-1}$ exist, we say that $\mathbf{A}$ is invertible (a.k.a., nonsingular, or nondegenerate). 
# In this case, the columns and rows of $\mathbf{A}$ are linearly independent.
# 
# 
# 
# <!-- A is invertible, that is, A has an inverse, is nonsingular, or is nondegenerate.
# A is row-equivalent to the n-by-n identity matrix In.
# A is column-equivalent to the n-by-n identity matrix In.
# A has n pivot positions.
# det A ≠ 0. In general, a square matrix over a commutative ring is invertible if and only if its determinant is a unit in that ring.
# A has full rank; that is, rank A = n.
# Based on the rank A=n, the equation Ax = 0 has only the trivial solution x = 0. and the equation Ax = b has exactly one solution for each b in Kn.
# The kernel of A is trivial, that is, it contains only the null vector as an element, ker(A) = {0}.
# The columns of A are linearly independent.
# The columns of A span Kn.
# Col A = Kn.
# The columns of A form a basis of Kn.
# The linear transformation mapping x to Ax is a bijection from Kn to Kn.
# There is an n-by-n matrix B such that AB = In = BA.
# The transpose AT is an invertible matrix (hence rows of A are linearly independent, span Kn, and form a basis of Kn).
# The number 0 is not an eigenvalue of A.
# The matrix A can be expressed as a finite product of elementary matrices.
# The matrix A has a left inverse (that is, there exists a B such that BA = I) or a right inverse (that is, there exists a C such that AC = I), in which case both left and right inverses exist and B = C = A−1. -->

# In[44]:


A = np.array([[3, 0, 2], [2, 0, -2], [0, 1, 1]])
np.linalg.inv(A)


# ## Rank
# 
# The **column rank** of $\mathbf{A}$ is the dimension of the column space of $\mathbf{A}$ (i.e., the number of linearly independent columns), while the **row rank** of $\mathbf{A}$ is the dimension of the row space of $\mathbf{A}$ (i.e., the number of linearly independent rows).
# 
# A fundamental result in linear algebra is that the column rank and the row rank are **always equal**, and this number is called the rank of $\mathbf{A}$.
# A matrix $\mathbf{A}\in\mathbb{R}^{m\times n}$ is said to have **full rank** if it's rank equals to $\min(m, n$); otherwise, the matrix is **rank deficient**.

# In[45]:


A = np.array([[1,2],[3,4]]); 
print(A)
B = np.array([[1,2],[2,4]]); 
print(B)
C = np.array([[1,0,0],[0,1,0],[0,0,1]])
print(C)

np.linalg.matrix_rank(A), np.linalg.matrix_rank(B), np.linalg.matrix_rank(C)


# ## Invertibility
# 
# An $n\times n$ square matrix $\mathbf{A}$ is called **invertible** (also **nonsingular** or **nondegenerate**), if there exists an $n\times n$ square matrix $\mathbf{B}$ such that
# $$
# \mathbf{A}\mathbf{B}=\mathbf{B}\mathbf{A}=\mathbf{I}_n
# $$
# 
# where $\mathbf{I}_n$ denotes the $n\times n$ identity matrix. In this case, then the matrix $\mathbf{B}$ is uniquely determined by $\mathbf{A}$, and is called the inverse of $\mathbf{A}$, denoted by $\mathbf{A}^{-1}$.
# 
# A square matrix that is not invertible is called **singular** or **degenerate**. Non-square matrices ($m\times n$ matrices with $n\ne n$) do not have an inverse (but they may have a left inverse or right inverse). 
# A square matrix is singular if and only if its determinant is zero. In practice, singular square matrices are pretty rare. 
# 
# 
# 
# $$
# \mathbf{A} = \begin{bmatrix}
# a & b \\
# c & d
# \end{bmatrix},
# $$
# 
# then we can see that the inverse is
# 
# $$
#  \frac{1}{ad-bc}  \begin{bmatrix}
# d & -b \\
# -c & a
# \end{bmatrix}.
# $$
# 
# 

# In[46]:


A = np.array([[1,2],[2,4]]); 
np.linalg.det(A)


# In[46]:




