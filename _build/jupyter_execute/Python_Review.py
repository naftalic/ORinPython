#!/usr/bin/env python
# coding: utf-8

# # Python Review 
# 
# [Colab notebook](https://colab.research.google.com/drive/16hzyq99m4rVF8qRCXDhMAMX8NdW9y-J7?usp=sharing)
# 
# In the following sections, we will repeatedly use Python scripts. If you are **less** familiar with Python, here is a short tutorial on what you need to know.
# Also, please take a look here: [Google's Python Class](https://developers.google.com/edu/python)

# ## Introduction

# Python is a general-purpose programming language that becomes a robust environment for scientific computing, combined with a few popular libraries (numpy, scipy, matplotlib).

# In[1]:


get_ipython().system('python --version')


# ## Basics of Python

# Python is a high-level, dynamically typed multiparadigm programming language. Python code is often almost like pseudocode since it allows you to express compelling ideas in a few lines of code while being very readable.

# ### Basic data types

# #### Numbers

# Integers and floats work as you would expect from other languages:

# In[2]:


x = 3
print(x, type(x))


# In[3]:


print(x + 1)  #addition
print(x - 1)  #subtraction
print(x * 2)  #multiplication
print(x ** 2) #exponentiation


# In[4]:


x = 10; x += 1
print(x)

x = 10; x *= 2
print(x)


# In[5]:


y = 2.5
print(y, y+1, y*2, y *2, type(y))


# #### Booleans

# Python implements all of the usual operators for Boolean logic, but uses English words rather than symbols:

# In[6]:


t, f = True, False; print(type(t))


# Now we let's look at some of the operations:

# In[7]:


print(t and f) # Logical AND;
print(t or f)  # Logical OR;
print(not t)   # Logical NOT;
print(t != f)  # Logical XOR;


# #### Strings

# In[8]:


hello = 'hello'   # single quotes or double quotes; it does not matter
world = "world"   
print(hello, len(hello))


# In[9]:


hw = hello + '-' + world+'!'  # String concatenation
print(hw)


# In[10]:


hw12 = '{} {} {}'.format(hello, world, 12)  # string formatting
print(hw12)


# String objects have a bunch of useful methods; for example:

# In[11]:


s = "hello"
print(s.capitalize())  # Capitalize a string
print(s.upper())       # Convert a string to uppercase; prints "HELLO"
print(s.rjust(7))      # Right-justify a string, padding with spaces
print(s.center(7))     # Center a string, padding with spaces
print(s.replace('l', '(ell)'))  # Replace all instances of one substring with another
print('  world '.strip())  # Strip leading and trailing whitespace


# #### Lists

# A list is the Python equivalent of an array, but is resizeable and can contain elements of different types:

# In[12]:


xs = [3, 1, 2]   # Create a list
print(xs, xs[2])
print(xs[-1])     # Negative indices count from the end of the list


# In[13]:


xs[2] = 'foo77'    # Lists can contain elements of different types
print(xs)


# In[14]:


xs.append('bar 87') # Add a new element to the end of the list
print(xs)  


# In[15]:


x = xs.pop()     # Remove and return the last element of the list
print(x, xs)


# #### Slicing

# In addition to accessing list elements one at a time, Python provides concise syntax to access sublists; this is known as slicing:

# In[16]:


nums = list(range(5))    # range is a built-in function that creates a list of integers
print(nums)         # Prints "[0, 1, 2, 3, 4]"
print(nums[2:4])    # Get a slice from index 2 to 4 (exclusive)
print(nums[2:])     # Get a slice from index 2 to the end
print(nums[:2])     # Get a slice from the start to index 2 (exclusive)
print(nums[:])      # Get a slice of the whole list
print(nums[:-1])    # Slice indices can be negative
nums[2:4] = [8, 9]; print(nums)  # Assign a new sublist to a slice      


# #### Loops

# One can loop over the elements in a list like this:

# In[17]:


animals = ['dog', 'cat', 'mouse']
for animal in animals:
    print(animal)


# If you want access to the index of each element within the body of a loop, use the built-in `enumerate` function:

# In[18]:


animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#{}: {}'.format(idx + 1, animal))


# #### List comprehensions:

# When programming, frequently we want to transform one type of data into another. As a simple example, consider the following code that computes square numbers:

# In[19]:


nums = [0, 1, 2, 3, 4]
squares = []
for x in nums:
    squares.append(x ** 2)
print(squares)


# You can make this code simpler using a list comprehension:

# In[20]:


nums = [0, 1, 2, 3, 4]
squares = [x ** 2 for x in nums]
print(squares)


# List comprehensions can also contain conditions:

# In[21]:


nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print(even_squares)


# In[22]:


even_squares = [x ** 2 if x % 2 == 0 else -99 for x in nums ] # list comprehension with if/else condition
print(even_squares)


# #### Dictionaries

# A dictionary stores (key, value) pairs

# In[23]:


d = {'cat': 'meow', 'dog': 'bark'}  # Create a new dictionary with some data
print(d['cat'])       # Get an entry from a dictionary
print('cat' in d)     # Check if a dictionary has a given key


# In[24]:


d['fish'] = 'wet'     # Set a new entry in the dictionary
print(d['fish'])      


# In[25]:


print(d)


# In[26]:


del d['fish']        # Remove an element from a dictionary


# In[27]:


print(d)


# It is easy to iterate over the keys in a dictionary:

# In[28]:


d = {'human': 2, 'dog': 4, 'spider': 8}
for animal, legs in d.items():
    print('A {} has {} legs'.format(animal, legs))


# Dictionary comprehensions: These are similar to list comprehensions but allow you to construct dictionaries easily. For example:

# In[29]:


nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)


# #### Sets

# A set is an **unordered** collection of distinct elements. As a simple example, consider the following:

# In[30]:


animals = {'cat', 'dog'}
print('cat' in animals)  
print('fish' in animals)  
print(animals)


# In[31]:


animals.add('fish')      # Add an element to a set
print('fish' in animals)
print(len(animals))       # Number of elements in a set;
print(animals)


# In[32]:


animals.add('cat')       # Adding an element that is already in the set does nothing
print(len(animals))       

animals.remove('cat')    # Remove an element from a set
print(len(animals))    
print(animals)   


# Loops: Iterating over a set has the same syntax as iterating over a list; however, since sets are unordered, you cannot make assumptions about the order in which you visit the elements of the set:

# In[33]:


animals = {'cat', 'dog', 'mouse'}
for idx, animal in enumerate(animals):
    print('#{}: {}'.format(idx + 1, animal))


# Set comprehensions: Like lists and dictionaries, we can easily construct sets using set comprehensions:

# In[34]:


from math import sqrt
print({int(sqrt(x)) for x in range(30)})


# #### Tuples

# A tuple is an immutable ordered list of values. A tuple is similar to a list; one of the most important differences is that tuples can be used as **keys in dictionaries** and as **elements of sets**, **while lists cannot**. Here is a trivial example:

# In[35]:


d = {(x, x + 1): x for x in range(7)}  # Create a dictionary with tuple keys
t = (5, 6)       # Create a tuple
print(type(t))
print(d)
print(d[t])       
print(d[(1, 2)])


# ### Functions

# Python functions are defined using the `def` keyword. For example:

# In[36]:


def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))


# We will often define functions to take optional keyword arguments, like this:

# In[37]:


def hello(name, loud=False):
    if loud:
        print('HELLO, {}'.format(name.upper()))
    else:
        print('Hello, {}!'.format(name))

hello('Bob')
hello('Fred', loud=True)


# ### Classes

# The syntax for defining classes in Python is simple:

# In[38]:


class Greeter:
    # Constructor
    def __init__(self, name):
        self.name = name  # Create an *instance* variable

    # Instance method
    def greet(self, loud=False):
        if loud:
          print('HELLO, {}'.format(self.name.upper()))
        else:
          print('Hello, {}!'.format(self.name))

g = Greeter('Fred')  # Construct an instance of the Greeter class
print(g.name)
g.greet()            # Call an instance method
g.greet(loud=True)   # Call an instance method


# ## Numpy

# Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object and tools for working with these arrays. 

# In[39]:


import numpy as np


# ### Arrays

# A numpy array is a grid of values, all of the same type, indexed by a tuple of non-negative integers. 
# The number of dimensions is the rank of the array. 
# The shape of an array is a tuple of integers giving the size of the array along each dimension.

# We can initialize numpy arrays from nested Python lists and access elements using square brackets:

# In[40]:


a = np.array([1, 2, 3])  # Create a rank 1 array
print(type(a), a.shape, a[0], a[1], a[2])

a[0] = 5                 # Change an element of the array
print(a)                  


# In[41]:


b = np.array([[1,2,3],[4,5,6]])   # Create a rank 2 array
print(b)


# In[42]:


print(b.shape)
print(b[0, 0], b[0, 1], b[1, 0])


# Numpy also provides many functions to create arrays:

# In[43]:


a = np.zeros((2,2))  # Create an array of all zeros
print(a)


# In[44]:


b = np.ones((1,2))   # Create an array of all ones
print(b)


# In[45]:


c = np.full((2,2), 7) # Create a constant array
print(c)


# In[46]:


d = np.eye(2)        # Create a 2x2 identity matrix
print(d)


# In[47]:


e = np.random.random((2,2)) # Create an array filled with random values between 0 and 1
print(e)


# ### Array indexing

# Slicing: Similar to Python lists, numpy arrays can be sliced. Since arrays may be multidimensional, you must specify a slice for each dimension of the array:

# In[48]:


import numpy as np

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)

b = a[:2, 1:3]
print(b)


# A slice of an array is a view into the **same** data, so modifying it will modify the original array.

# In[49]:


print(a[0, 1])
b[0, 0] = 77   
print(a[0, 1]) 


# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:

# In[50]:


row_r1 = a[1, :]    # Rank 1 view of the second row of a  
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
row_r3 = a[[1], :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)
print(row_r3, row_r3.shape)


# Same when accessing columns of an array:

# In[51]:


col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)
print()
print(col_r2, col_r2.shape)


# Integer array indexing: When you index into numpy arrays using slicing, the resulting array view will always be a subarray of the original array. In contrast, integer array indexing allows you to construct arbitrary arrays using the data from another array. Here is an example:

# In[52]:


a = np.array([[1,2], [3, 4], [5, 6]])
print(a)

print(a[[0, 1, 2], [0, 1, 0]])

print(np.array([a[0, 0], a[1, 1], a[2, 0]]))


# One useful trick with integer array indexing is selecting or mutating one element from each row of a matrix:

# In[53]:


a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(a)


# In[54]:


# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  


# In[55]:


a[np.arange(4), b] += 10
print(a)


# Boolean array indexing: Boolean array indexing lets you pick out arbitrary elements of an array. This type of indexing is frequently used to select elements of an array that satisfy some condition. Here is an example:

# In[56]:


import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)  
print(bool_idx)


# In[57]:


# Using boolean array indexing to construct a rank 1 array
print(a[bool_idx])

print(a[a > 2])


# ### Datatypes

# Every numpy array is a grid of elements of the **same** type. Numpy provides a large set of numeric datatypes that you can use to construct arrays. Numpy tries to guess a datatype when you create an array, but functions that construct arrays usually also include an optional argument to specify the datatype explicitly. Here is an example:

# In[58]:


x = np.array([1, 2])  # Let numpy choose the datatype
y = np.array([1.0, 2.0])  # Let numpy choose the datatype
z = np.array([1, 2], dtype=np.int64)  # Force a particular datatype

print(x.dtype, y.dtype, z.dtype)


# ### Array math

# Basic mathematical functions operate elementwise on arrays, and are available both as operator overloads and as functions in the numpy module:

# In[59]:


x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
print(x + y)
print(np.add(x, y))


# In[60]:


x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.int64)  # now an int

# same result as before
print(x + y)
print(np.add(x, y))


# In[61]:


# Elementwise difference; both produce the array
print(x - y)
print(np.subtract(x, y))


# In[62]:


# Elementwise product; both produce the array
print(x * y)
print(np.multiply(x, y))


# In[63]:


# Elementwise division; both produce the array
print(x / y)
print(np.divide(x, y))


# In[64]:


# Elementwise square root;
print(np.sqrt(x))


# The dot function is used to compute the inner products of vectors, multiply a vector by a matrix, and multiply matrices. dot is available both as a function in the numpy module and as an instance method of array objects:

# In[65]:


x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce the same result
print(v.dot(w))
print(np.dot(v, w))


# You can also use the `@` operator which is equivalent to numpy's `dot` operator.

# In[66]:


print(v @ w)


# In[67]:


# Matrix-vector product; all produce a rank 1 array
print(x.dot(v))
print(np.dot(x, v))
print(x @ v)


# In[68]:


# Matrix-matrix product; both produce a rank 2 array
print(x.dot(y))
print(np.dot(x, y))
print(x @ y)


# Numpy provides many useful functions for performing computations on arrays; one of the most useful is `sum`:

# In[69]:


x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of **all** elements
print(np.sum(x, axis=0))  # Compute sum of each **column**, collapsing all rows
print(np.sum(x, axis=1))  # Compute sum of each **row**, collapsing all columns


# Apart from computing mathematical functions using arrays, we frequently need to reshape or otherwise manipulate data in arrays. The simplest example of this type of operation is transposing a matrix; to transpose a matrix, simply use the T attribute of an array object:

# In[70]:


print(x)
print(x.T)


# In[71]:


v = np.array([[1,2,3]])
print(v )
print(v.T)


# ### Broadcasting

# Broadcasting is a powerful mechanism that allows numpy to work with arrays of different shapes when performing arithmetic operations. Frequently we have a smaller array and a larger array, and we want to use the smaller array multiple times to perform some operation on the larger array.
# 
# For example, suppose that we want to add a constant vector to each matrix row. We could do it like this:

# In[72]:


# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(x)

v = np.array([1, 0, 1])
print(v)

y = np.zeros_like(x)   # an array of zeros with the same shape and type as x
print(y)

print(x.shape,v.shape,y.shape)

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

print(y)


# However, when the matrix x is huge, computing an explicit loop in Python can be slow. 
# 
# Note that adding the vector v to each row of the matrix x is equivalent to forming a matrix vv by stacking multiple copies of v vertically, then performing an elementwise summation of x and vv. We could implement this approach like this:

# In[73]:


print(v)
vv = np.tile(v, (4, 1))  # Stack 4 copies of v on top of each other
print(vv)                


# In[74]:


y = x + vv  # Add x and vv elementwise
print(y)


# Numpy broadcasting allows us to perform this computation without actually creating multiple copies of v. In example,

# In[75]:


x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])

y = x + v  # Add v to each row of x using broadcasting
print(y)


# The line y = x + v works even though x has shape (4, 3) and v has shape (3,) due to broadcasting; this line works as if v actually had shape (4, 3), where each row was a copy of v, and the sum gets done elementwise.
# 
# When operating on two arrays, numPy compares their shapes element-wise. It starts with the **rightmost** dimensions and works its way left. Two dimensions are compatible when
# 
# * they are equal, or
# * one of them is 1.
# 
# If these conditions are not met, a ValueError is thrown. Here are a few more examples:
# 
# 
# Ex 1.  | Shape  |    | Ex 2.  | Shape      | | Ex 3.  | Shape         | |
# --     | --     | -- | --     | --         | | --     | --            | |
# A      | 3 x 2  |    | A      | 10 x 3 x 3 | | A      | 6 x 1 x 4 x 1 | |
# B      | 1 x 2  |    | B      | 10 x 1 x 3 | | B      | 1 x 5 x 1 x 3 | |
# A+B    | 3 x 2  |    | A+B    | 10 x 3 x 3 | | A+B    | 6 x 5 x 4 x 3 | |
# 
# Here are some more applications of broadcasting:

# In[76]:


# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
print(v.shape, w.shape)

print(np.reshape(v, (3, 1)) * w)


# In[77]:


# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
v = np.array([1,2,3])  
print(x.shape, v.shape)

print(x + v)


# In[78]:


# Add a vector to each column of a matrix
print(x.shape, w.shape)

print((x.T + w).T)


# In[79]:


# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing:
print(x * 2)


# ## Pandas

# Pandas is a software library in Python for data manipulation and analysis. It offers data structures and operations for manipulating numerical tables and time series.
# The main data structure is the DataFrame, which is an **in-memory** 2D table similar to a spreadsheet, with column names and row labels.

# In[80]:


import pandas as pd


# ### Series objects
# A Series object is 1D array, similar to a column in a spreadsheet, while a DataFrame objects is a 2D table with column names and row labels.

# In[81]:


s = pd.Series([1,2,3,4]); s


# Series objects can be passed as parameters to numpy functions

# In[82]:


np.log(s)


# In[83]:


s = s + s + [1,2,3,4] # elementwise addition with a list
s


# In[84]:


s = s + 1 # broadcasting
s
 


# In[85]:


s >=10


# ### Index labels
# Each item in a Series object has a unique identifier called index label. By default, it is simply the rank of the item in the Series (starting at 0), but you can also set the index labels manually.

# In[86]:


s2 = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
s2


# You can then use the Series just like a **dict** object

# In[87]:


s2['b']


# access items by integer location

# In[88]:


s2[1]


# accessing by **label** (L-oc)

# In[89]:


s2.loc['b'] 


# accessing by **integer location** (IL-oc)

# In[90]:


s2.iloc[1] 


# Slicing a Series

# In[91]:


s2.iloc[1:3]


# In[92]:


s2.iloc[2:]


# ### Initializing from a dict
# Create a Series object from a dict, keys are used as index labels

# In[93]:


d = {"b": 1, "a": 2, "e": 3, "d": 4}
s3 = pd.Series(d)
s3


# In[94]:


s4 = pd.Series(d, index = ["c", "a"])
s4


# In[95]:


s5 = pd.Series(10, ["a", "b", "c"], name="series")
s5


# ### Automatic alignment
# When an operation involves multiple Series objects, pandas automatically **aligns** items by matching index labels.

# In[96]:


print(s2.keys())
print(s3.keys())

s2 + s3


# ### DataFrame objects
# DataFrame object represents a spreadsheet, with cell values, column names and row index labels. 
# 

# In[97]:


d = {
    "feature1": pd.Series([1,2,3], index=["a", "b", "c"]),
    "feature2": pd.Series([4,5,6], index=["b", "a", "c"]),
    "feature3": pd.Series([7,8],  index=["c", "b"]),
    "feature4": pd.Series([9,10], index=["a", "b"]),
}
df = pd.DataFrame(d)
df


# accessing a column

# In[98]:


df["feature3"] 


# accessing ,multiple columns

# In[99]:


df[["feature1", "feature3"]] 


# In[100]:


df


# Constructing a new DataFrame from an existing DataFrame

# In[101]:


df2 = pd.DataFrame(
        df,
        columns=["feature1", "feature2", "feature3"],
        index=["b", "a", "d"]
     )
df2


# Creating a DataFrame from a **list of lists**

# In[102]:


lol = [
        [11, 1,      "a",   np.nan],
        [12, 3,      "b",       14],
        [13, np.nan, np.nan,    15]
      ]
df3 = pd.DataFrame(
        lol,
        columns=["feature1", "feature2", "feature3", "feature4"],
        index=["a", "b", "c"]
     )
df3


# Creating a DataFrame with a **dictionary of dictionaries**

# In[103]:


df5 = pd.DataFrame({
    "feature1": {"a": 15, "b": 1984, "c": 4},
    "feature2": {"a": "sentence", "b": "word"},
    "feature3": {"a": 1, "b": 83, "c": 4},
    "feature4": {"c": 2, "d": 0}
})
df5


# ### Multi-indexing
# If all columns/rows are tuples of the same size, then they are understood as a multi-index

# In[104]:


df5 = pd.DataFrame(
  {
    ("features12", "feature1"):
        {("rows_ab","a"): 444, ("rows_ab","b"): 444, ("rows_c","c"): 666},
    ("features12", "feature2"):
        {("rows_ab","a"): 111, ("rows_ab","b"): 222},
    ("features34", "feature3"):
        {("rows_ab","a"): 555, ("rows_ab","b"): 333, ("rows_c","c"): 777},
    ("features34", "feature4"):
        {("rows_ab", "a"):333, ("rows_ab","b"): 999, ("rows_c","c"): 888}
  }
)

df5


# In[105]:


df5['features12']


# In[106]:


df5['features12','feature1']


# In[107]:


df5['features12','feature1']['rows_c']


# In[108]:


df5.loc['rows_c']


# In[109]:


df5.loc['rows_ab','features12']


# ### Dropping a level
# 

# In[110]:


df5.columns = df5.columns.droplevel(level = 0); 
df5


# ### Transposing
# swaping columns and indices

# In[111]:


df6 = df5.T
df6


# ### Stacking and unstacking levels
# expanding the lowest column level as the lowest index

# In[112]:


df7 = df6.stack()
df7


# In[113]:


df8 = df7.unstack()
df8


# If we call unstack again, we end up with a Series object

# In[114]:


df9 = df8.unstack()
df9


# ### Accessing rows

# In[115]:


df


# In[116]:


df.loc["c"] #access the c row


# In[117]:


df.iloc[2] #access the 2nd column by integer location


# slice of rows

# In[118]:


df.iloc[1:3] # by integer location


# slice rows using boolean array
# 
# 
# 

# In[119]:


df[np.array([True, False, True])]


# with boolean expressions:

# In[120]:


df[df["feature2"] <= 5]


# ### Adding and removing columns

# In[121]:


df


# In[122]:


df['feature5'] = 5 - df['feature2'] #adding a column
df['feature6'] = df['feature3'] > 5
df  


# In[123]:


del df['feature6']
df


# In[124]:


df["feature6"] = pd.Series({"a": 1, "b": 51, "c":1}) 
df


# In[125]:


df.insert(1, "feature1b", [0,1,2]) #insert in the location of the 1st column
df


# ### Assigning new columns
# create a new DataFrame with new columns

# In[126]:


df


# In[127]:


df10 = df.assign(feature0 = df["feature1"] * df["feature2"] )
df10.assign(feature1 = df10["feature1"] +1)
df10


# ### Evaluating an expression

# In[128]:


df = df[['feature1','feature2','feature3','feature4']]
df.eval("feature1 + feature2 ** 2")


# In[129]:


df


# use inplace=True to modify a DataFrame
# 
# 
# 
# 

# In[130]:


df.eval("feature3 = feature1 + feature2 ** 2", inplace=True)
df


# use a local or global variable in an expression by prefixing it with @

# In[131]:


threshold = 30
df.eval("feature3 = feature1 + feature2 ** 2 > @threshold", inplace=True)
df


# ### Querying a DataFrame
# The query method lets you filter a DataFrame

# In[132]:


df.query("feature1 > 2 and feature2 == 6")


# ### Sorting a DataFrame

# In[133]:


df


# In[134]:


df.sort_index(ascending=False)


# In[135]:


df.sort_index(axis=0, inplace=True) #by row
df


# In[136]:


df.sort_index(axis=1, inplace=True) #by column
df


# In[137]:


df.sort_values(by="feature2", inplace=True)
df


# ### Operations on DataFrame

# In[138]:


a = np.array([[1,2,3],[4,5,6],[7,8,9]])
df = pd.DataFrame(a, columns=["q", "w", "e"], index=["a","b","c"])
df


# In[139]:


np.sqrt(df)


# In[140]:


df + 1 #broadcasting


# In[141]:


df >= 5


# In[142]:


df.mean(), df.std(), df.max(), df.sum()


# In[143]:


df


# In[144]:


(df > 2).all() #checks whether all values are True or not


# In[145]:


(df > 2).all(axis = 0) #executed vertically (on each column)


# In[146]:


(df > 2).all(axis = 1) #execute the horizontally (on each row).


# In[147]:


(df == 8).any(axis = 1)


# In[148]:


df - df.mean() 


# In[149]:


df - df.values.mean() # subtracts the global mean elementwise


# ### Handling missing data
# 

# In[150]:


df10


# In[151]:


df11 = df10.fillna(0)
df11


# In[152]:


df11.loc["d"] = np.nan
df11.fillna(0,inplace=True)
df11


# ### Aggregating with groupby
# Similar to SQL, pandas allows to compute over groups.

# In[153]:


df5 = pd.DataFrame({
    "feature1": {"a": 3, "b": 11, "c": 14, 'd':4},
    "feature2": {"a": 2, "b": 2, "c": 4, 'd':4},
    "feature3": {"a": 32, "b": 4, "c": 3, 'd':35},
    "feature4": {"a": 5, "b": 11, "c": 2, 'd':13}
})

df5


# In[154]:


df5.groupby("feature2").mean()


# ### Pivot tables
# pivot tables allows for quick summarization

# In[155]:


df9 = df8.stack().reset_index()
df9


# In[156]:


pd.pivot_table(df9, index="level_0", aggfunc=np.mean)


# In[157]:


pd.pivot_table(df9, index="level_0", values=["rows_ab"], aggfunc=np.max)


# ### Functions

# In[158]:


df = np.fromfunction(lambda x,y: (x+y)%7*11, (1000, 10))
big_df = pd.DataFrame(df, columns=list("1234567890"))
big_df.head(5)


# In[159]:


big_df[big_df % 3 == 0] = np.nan
big_df.insert(3,"feature1", 999)
big_df.head(5)


# In[160]:


big_df.tail(5)


# In[161]:


big_df.describe()


# ### Save and Load
# 

# In[162]:


big_df.to_csv("my_df.csv")
big_df.to_csv("my_df.xlsx")


# In[163]:


df0 = pd.read_csv("my_df.csv", index_col=0)


# ### Combining DataFrames
# 

# In[164]:


city_loc = pd.DataFrame(
    [
        ["CA", "Murrieta", 33.569443, -117.202499],
        ["NY", "Cohoes", 42.774475, -73.708412],
        ["NY", "Rye", 40.981613,	-73.691925],
        ["CA", "Ojai", 34.456936,	-119.254440],
        ["AL", "Jasper", 33.834263,	-87.280708]
    ], columns=["state", "city", "lat", "lon"])

city_loc


# In[165]:


city_pop = pd.DataFrame(
    [
        [112941, "Murrieta", "California"],
        [16684, "Cohoes", "New York"],
        [15820, "Rye", "New York"],
        [13649, "Jasper", "Alabama"]
    ], index=[3,4,5,6], columns=["population", "city", "state"])
city_pop


# In[166]:


pd.merge(left=city_loc, right=city_pop, on="city", how="inner")


# In[167]:


all_cities = pd.merge(left=city_loc, right=city_pop, on="city", how="outer")
all_cities


# In[168]:


pd.merge(left=city_loc, right=city_pop, on="city", how="right")


# ### Concatenation

# In[169]:


result_concat = pd.concat([city_pop, city_loc], ignore_index=True)
result_concat


# In[170]:


pd.concat([city_loc, city_pop], join="inner", ignore_index=True)


# In[171]:


pd.concat([city_loc.set_index("city"), city_pop.set_index("city")], axis=1, ignore_index=True)


# In[172]:


city_loc.append(city_pop) #another way to append, but works on a copy and returns the modified copy.


# ### Categories

# In[173]:


city_eco = city_pop.copy()
city_eco["edu"] = [17, 17, 34, 20]
city_eco


# In[174]:


city_eco["education"] = city_eco["edu"].astype('category')
city_eco["education"].cat.categories


# In[175]:


city_eco["education"].cat.categories = ["College", "High School", "Basic"]
city_eco


# In[176]:


city_eco.sort_values(by="education", ascending=False)

