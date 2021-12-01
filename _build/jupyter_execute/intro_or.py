#!/usr/bin/env python
# coding: utf-8

# #Chapter -3: Python Tutorial
# 
# In the following sections, we will repeatedly use Python scripts. In case you are **less** familiar with Python, here is a short tutorial on what you need to know.
# Also, please take a look here: Google's Python Class- https://developers.google.com/edu/python
# 

# ##Introduction

# Python is a general-purpose programming language that, combining with a few popular libraries (numpy, scipy, matplotlib), becomes a powerful environment for scientific computing.

# In[1]:


get_ipython().system('python --version')


# ##Basics of Python

# Python is a high-level, dynamically typed multiparadigm programming language. Python code is often said to be almost like pseudocode, since it allows you to express very powerful ideas in very few lines of code while being very readable.

# ###Basic data types

# ####Numbers

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


# ####Booleans

# Python implements all of the usual operators for Boolean logic, but uses English words rather than symbols:

# In[6]:


t, f = True, False; print(type(t))


# Now we let's look at the operations:

# In[7]:


print(t and f) # Logical AND;
print(t or f)  # Logical OR;
print(not t)   # Logical NOT;
print(t != f)  # Logical XOR;


# ####Strings

# In[8]:


hello = 'hello'   # String literals can use single quotes
world = "world"   # or double quotes; it does not matter
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


# ####Lists

# A list is the Python equivalent of an array, but is resizeable and can contain elements of different types:

# In[12]:


xs = [3, 1, 2]   # Create a list
print(xs, xs[2])
print(xs[-1])     # Negative indices count from the end of the list; prints "2"


# In[13]:


xs[2] = 'foo'    # Lists can contain elements of different types
print(xs)


# In[14]:


xs.append('bar') # Add a new element to the end of the list
print(xs)  


# In[15]:


x = xs.pop()     # Remove and return the last element of the list
print(x, xs)


# ####Slicing

# In addition to accessing list elements one at a time, Python provides concise syntax to access sublists; this is known as slicing:

# In[16]:


nums = list(range(5))    # range is a built-in function that creates a list of integers
print(nums)         # Prints "[0, 1, 2, 3, 4]"
print(nums[2:4])    # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print(nums[2:])     # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print(nums[:2])     # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print(nums[:])      # Get a slice of the whole list; prints ["0, 1, 2, 3, 4]"
print(nums[:-1])    # Slice indices can be negative; prints ["0, 1, 2, 3]"
nums[2:4] = [8, 9] # Assign a new sublist to a slice
print(nums)         # Prints "[0, 1, 8, 9, 4]"


# ####Loops

# You can loop over the elements of a list like this:

# In[17]:


animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)


# If you want access to the index of each element within the body of a loop, use the built-in `enumerate` function:

# In[18]:


animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print('#{}: {}'.format(idx + 1, animal))


# ####List comprehensions:

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


# ####Dictionaries

# A dictionary stores (key, value) pairs

# In[23]:


d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print(d['cat'])       # Get an entry from a dictionary; prints "cute"
print('cat' in d)     # Check if a dictionary has a given key; prints "True"


# In[24]:


d['fish'] = 'wet'    # Set an entry in a dictionary
print(d['fish'])      # Prints "wet"


# In[25]:


print(d.get('monkey', 'N/A'))  # Get an element with a default; prints "N/A"
print(d.get('fish', 'N/A'))    # Get an element with a default; prints "wet"


# In[26]:


print(d)


# In[27]:


del d['fish']        # Remove an element from a dictionary
print(d.get('fish', 'N/A')) # "fish" is no longer a key; prints "N/A"


# In[28]:


print(d)


# It is easy to iterate over the keys in a dictionary:

# In[29]:


d = {'person': 2, 'cat': 4, 'spider': 8}
for animal, legs in d.items():
    print('A {} has {} legs'.format(animal, legs))


# Dictionary comprehensions: These are similar to list comprehensions, but allow you to easily construct dictionaries. For example:

# In[30]:


nums = [0, 1, 2, 3, 4]
even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
print(even_num_to_square)


# ####Sets

# A set is an unordered collection of distinct elements. As a simple example, consider the following:

# In[31]:


animals = {'cat', 'dog'}
print('cat' in animals)   # Check if an element is in a set; prints "True"
print('fish' in animals)  # prints "False"
print(animals)


# In[32]:


animals.add('fish')      # Add an element to a set
print('fish' in animals)
print(len(animals))       # Number of elements in a set;
print(animals)


# In[33]:


animals.add('cat')       # Adding an element that is already in the set does nothing
print(len(animals))       
animals.remove('cat')    # Remove an element from a set
print(len(animals))    
print(animals)   


# Loops: Iterating over a set has the same syntax as iterating over a list; however since sets are unordered, you cannot make assumptions about the order in which you visit the elements of the set:

# In[34]:


animals = {'cat', 'dog', 'fish'}
for idx, animal in enumerate(animals):
    print('#{}: {}'.format(idx + 1, animal))


# Set comprehensions: Like lists and dictionaries, we can easily construct sets using set comprehensions:

# In[35]:


from math import sqrt
print({int(sqrt(x)) for x in range(30)})


# ####Tuples

# A tuple is an (immutable) ordered list of values. A tuple is in many ways similar to a list; one of the most important differences is that tuples can be used as keys in dictionaries and as elements of sets, while lists cannot. Here is a trivial example:

# In[36]:


d = {(x, x + 1): x for x in range(7)}  # Create a dictionary with tuple keys
t = (5, 6)       # Create a tuple
print(type(t))
print(d)
print(d[t])       
print(d[(1, 2)])


# ###Functions

# Python functions are defined using the `def` keyword. For example:

# In[37]:


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

# In[38]:


def hello(name, loud=False):
    if loud:
        print('HELLO, {}'.format(name.upper()))
    else:
        print('Hello, {}!'.format(name))

hello('Bob')
hello('Fred', loud=True)


# ###Classes

# The syntax for defining classes in Python is straightforward:

# In[39]:


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
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"


# ##Numpy

# Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array object, and tools for working with these arrays. 

# In[40]:


import numpy as np


# ###Arrays

# A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. The number of dimensions is the rank of the array; the shape of an array is a tuple of integers giving the size of the array along each dimension.

# We can initialize numpy arrays from nested Python lists, and access elements using square brackets:

# In[41]:


a = np.array([1, 2, 3])  # Create a rank 1 array
print(type(a), a.shape, a[0], a[1], a[2])
a[0] = 5                 # Change an element of the array
print(a)                  


# In[42]:


b = np.array([[1,2,3],[4,5,6]])   # Create a rank 2 array
print(b)


# In[43]:


print(b.shape)
print(b[0, 0], b[0, 1], b[1, 0])


# Numpy also provides many functions to create arrays:

# In[44]:


a = np.zeros((2,2))  # Create an array of all zeros
print(a)


# In[45]:


b = np.ones((1,2))   # Create an array of all ones
print(b)


# In[46]:


c = np.full((2,2), 7) # Create a constant array
print(c)


# In[47]:


d = np.eye(2)        # Create a 2x2 identity matrix
print(d)


# In[48]:


e = np.random.random((2,2)) # Create an array filled with random values between 0 and 1
print(e)


# ###Array indexing

# Numpy offers several ways to index into arrays.

# Slicing: Similar to Python lists, numpy arrays can be sliced. Since arrays may be multidimensional, you must specify a slice for each dimension of the array:

# In[49]:


import numpy as np

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print(a)

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]
print(b)


# A slice of an array is a view into the same data, so modifying it will modify the original array.

# In[50]:


print(a[0, 1])
b[0, 0] = 77    # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1]) 


# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:

# In[51]:


row_r1 = a[1, :]    # Rank 1 view of the second row of a  
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
row_r3 = a[[1], :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)
print(row_r3, row_r3.shape)


# In[52]:


# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)
print()
print(col_r2, col_r2.shape)


# Integer array indexing: When you index into numpy arrays using slicing, the resulting array view will always be a subarray of the original array. In contrast, integer array indexing allows you to construct arbitrary arrays using the data from another array. Here is an example:

# In[53]:


a = np.array([[1,2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and 
print(a[[0, 1, 2], [0, 1, 0]])

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))


# In[54]:


# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))


# One useful trick with integer array indexing is selecting or mutating one element from each row of a matrix:

# In[55]:


# Create a new array from which we will select elements
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(a)


# In[56]:


# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"


# In[57]:


# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10
print(a)


# Boolean array indexing: Boolean array indexing lets you pick out arbitrary elements of an array. Frequently this type of indexing is used to select the elements of an array that satisfy some condition. Here is an example:

# In[58]:


import numpy as np

a = np.array([[1,2], [3, 4], [5, 6]])

bool_idx = (a > 2)  # Find the elements of a that are bigger than 2;
                    # this returns a numpy array of Booleans of the same
                    # shape as a, where each slot of bool_idx tells
                    # whether that element of a is > 2.

print(bool_idx)


# In[59]:


# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])

# We can do all of the above in a single concise statement:
print(a[a > 2])


# For brevity we have left out a lot of details about numpy array indexing; if you want to know more you should read the documentation.

# ###Datatypes

# Every numpy array is a grid of elements of the same type. Numpy provides a large set of numeric datatypes that you can use to construct arrays. Numpy tries to guess a datatype when you create an array, but functions that construct arrays usually also include an optional argument to explicitly specify the datatype. Here is an example:

# In[60]:


x = np.array([1, 2])  # Let numpy choose the datatype
y = np.array([1.0, 2.0])  # Let numpy choose the datatype
z = np.array([1, 2], dtype=np.int64)  # Force a particular datatype

print(x.dtype, y.dtype, z.dtype)


# ###Array math

# Basic mathematical functions operate elementwise on arrays, and are available both as operator overloads and as functions in the numpy module:

# In[61]:


x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# Elementwise sum; both produce the array
print(x + y)
print(np.add(x, y))


# In[62]:


# Elementwise difference; both produce the array
print(x - y)
print(np.subtract(x, y))


# In[63]:


# Elementwise product; both produce the array
print(x * y)
print(np.multiply(x, y))


# In[64]:


# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))


# In[65]:


# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))


# The dot function is used to compute inner products of vectors, to multiply a vector by a matrix, and to multiply matrices. dot is available both as a function in the numpy module and as an instance method of array objects:

# In[66]:


x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))


# You can also use the `@` operator which is equivalent to numpy's `dot` operator.

# In[67]:


print(v @ w)


# In[68]:


# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))
print(x @ v)


# In[69]:


# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))
print(x @ y)


# Numpy provides many useful functions for performing computations on arrays; one of the most useful is `sum`:

# In[70]:


x = np.array([[1,2],[3,4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"


# Apart from computing mathematical functions using arrays, we frequently need to reshape or otherwise manipulate data in arrays. The simplest example of this type of operation is transposing a matrix; to transpose a matrix, simply use the T attribute of an array object:

# In[71]:


print(x)
print(x.T)


# In[72]:


v = np.array([[1,2,3]])
print(v )
print(v.T)


# ###Broadcasting

# Broadcasting is a powerful mechanism that allows numpy to work with arrays of different shapes when performing arithmetic operations. Frequently we have a smaller array and a larger array, and we want to use the smaller array multiple times to perform some operation on the larger array.
# 
# For example, suppose that we want to add a constant vector to each row of a matrix. We could do it like this:

# In[73]:


# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.zeros_like(x)   # an array of zeros with the same shape and type as x
print(y)
print(x.shape,v.shape,y.shape)

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

print(y)


# This works; however when the matrix `x` is very large, computing an explicit loop in Python could be slow. Note that adding the vector v to each row of the matrix `x` is equivalent to forming a matrix `vv` by stacking multiple copies of `v` vertically, then performing elementwise summation of `x` and `vv`. We could implement this approach like this:

# In[74]:


print(v)
vv = np.tile(v, (4, 1))  # Stack 4 copies of v on top of each other
print(vv)                # Prints "[[1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]]"


# In[75]:


y = x + vv  # Add x and vv elementwise
print(y)


# Numpy broadcasting allows us to perform this computation without actually creating multiple copies of v. Consider this version, using broadcasting:

# In[76]:


import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)


# The line `y = x + v` works even though `x` has shape `(4, 3)` and `v` has shape `(3,)` due to broadcasting; this line works as if v actually had shape `(4, 3)`, where each row was a copy of `v`, and the sum was performed elementwise.
# 
# Broadcasting two arrays together follows these rules:
# 
# 1. If the arrays do not have the same rank, prepend the shape of the lower rank array with 1s until both shapes have the same length.
# 2. The two arrays are said to be compatible in a dimension if they have the same size in the dimension, or if one of the arrays has size 1 in that dimension.
# 3. The arrays can be broadcast together if they are compatible in all dimensions.
# 4. After broadcasting, each array behaves as if it had shape equal to the elementwise maximum of shapes of the two input arrays.
# 5. In any dimension where one array had size 1 and the other array had size greater than 1, the first array behaves as if it were copied along that dimension
# 
# Here are some applications of broadcasting:

# In[77]:


# Compute outer product of vectors
v = np.array([1,2,3])  # v has shape (3,)
w = np.array([4,5])    # w has shape (2,)
# To compute an outer product, we first reshape v to be a column
# vector of shape (3, 1); we can then broadcast it against w to yield
# an output of shape (3, 2), which is the outer product of v and w:

print(np.reshape(v, (3, 1)) * w)
print(v.shape, w.shape)


# In[78]:


# Add a vector to each row of a matrix
x = np.array([[1,2,3], [4,5,6]])
# x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
# giving the following matrix:

print(x + v)


# In[79]:


# Add a vector to each column of a matrix
# x has shape (2, 3) and w has shape (2,).
# If we transpose x then it has shape (3, 2) and can be broadcast
# against w to yield a result of shape (3, 2); transposing this result
# yields the final result of shape (2, 3) which is the matrix x with
# the vector w added to each column. Gives the following matrix:

print((x.T + w).T)


# In[80]:


# Another solution is to reshape w to be a row vector of shape (2, 1);
# we can then broadcast it directly against x to produce the same
# output.
print(x + np.reshape(w, (2, 1)))


# In[81]:


# Multiply a matrix by a constant:
# x has shape (2, 3). Numpy treats scalars as arrays of shape ();
# these can be broadcast together to shape (2, 3), producing the
# following array:
print(x * 2)


# Broadcasting typically makes your code more concise and faster, so you should strive to use it where possible.

# ## Pandas

# Pandas is a software library in Python for data manipulation and analysis. It offers data structures and operations for manipulating numerical tables and time series.
# The main data structure is the DataFrame, which is an in-memory 2D table similar to a spreadsheet, with column names and row labels.

# In[82]:


import pandas as pd


# ### Series objects
# A Series object is 1D array, similar to a column in a spreadsheet, while a DataFrame objects is a 2D table with column names and row labels.

# In[83]:


s = pd.Series([1,2,3,4]); s


# Series objects can be passed as parameters to NumPy functions

# In[84]:


np.log(s)


# In[85]:


s = s + s + [1,2,3,4] # elementwise addition with a list
s


# In[86]:


s = s + 1 #Broadcasting
s
 


# In[87]:


s >=10


# ### Index labels
# Each item in a Series object has a unique identifier called index label. By default, it is simply the rank of the item in the Series (starting at `0`) but you can also set the index labels manually

# In[88]:


s2 = pd.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
s2


# You can then use the Series just like a dict object

# In[89]:


s2['b']


# access items by integer location

# In[90]:


s2[1]


# In[91]:


s2.loc['b'] #accessing by label


# In[92]:


s2.iloc[1] #accessing by integer location


# Slicing a Series

# In[93]:


s2.iloc[1:3]


# In[94]:


s2.iloc[2:]


# ### Initializing from a dict
# Create a Series object from a dict, keys are used as index labels

# In[95]:


d = {"b": 1, "a": 2, "e": 3, "d": 4}
s3 = pd.Series(d)
s3


# In[96]:


s4 = pd.Series(d, index = ["c", "a"])
s4


# In[97]:


s5 = pd.Series(10, ["a", "b", "c"], name="series")
s5


# ### Automatic alignment
# When an operation involves multiple `Series` objects, `pandas` automatically aligns items by matching index labels.

# In[98]:


print(s2.keys())
print(s3.keys())

s2 + s3


# The resulting Series contains the union of index labels from s2 and s3. But, missing index labels get the value of NaN 

# ### DataFrame objects
# DataFrame object represents a spreadsheet, with cell values, column names and row index labels. 
# 

# In[99]:


d = {
    "feature1": pd.Series([1,2,3], index=["a", "b", "c"]),
    "feature2": pd.Series([4,5,6], index=["b", "a", "c"]),
    "feature3": pd.Series([7,8],  index=["c", "b"]),
    "feature4": pd.Series([9,10], index=["a", "b"]),
}
df = pd.DataFrame(d)
df


# You can access columns pretty much as you would expect. They are returned as `Series` objects:

# In[100]:


df["feature3"] #accessing a column


# In[101]:


df[["feature1", "feature3"]] #accessing ,multiple columns


# In[102]:


df


# Constructing a new DataFrame from an existing DataFrame

# In[103]:


df2 = pd.DataFrame(
        df,
        columns=["feature1", "feature2", "feature3"],
        index=["b", "a", "d"]
     )
df2


# Creating a DataFrame from a list of lists

# In[104]:


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


# Creating a DataFrame with a dictionary of dictionaries

# In[105]:


df5 = pd.DataFrame({
    "feature1": {"a": 15, "b": 1984, "c": 4},
    "feature2": {"a": "sentence", "b": "word"},
    "feature3": {"a": 1, "b": 83, "c": 4},
    "feature4": {"c": 2, "d": 0}
})
df5


# ### Multi-indexing
# If all columns/rows are tuples of the same size, then they are understood as a multi-index

# In[106]:


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


# In[107]:


df5['features12']


# In[108]:


df5['features12','feature1']


# In[109]:


df5['features12','feature1']['rows_c']


# In[110]:


df5.loc['rows_c']


# In[111]:


df5.loc['rows_ab','features12']


# ### Dropping a level
# 

# In[112]:


df5.columns = df5.columns.droplevel(level = 0); 
df5


# ### Transposing
# swaping columns and indices

# In[113]:


df6 = df5.T
df6


# ### Stacking and unstacking levels
# expanding the lowest column level as the lowest index

# In[114]:


df7 = df6.stack()
df7


# In[115]:


df8 = df7.unstack()
df8


# If we call unstack again, we end up with a Series object

# In[116]:


df9 = df8.unstack()
df9


# ### Accessing rows

# In[117]:


df


# In[118]:


df.loc["c"] #access the c row


# In[119]:


df.iloc[2] #access the 2nd column


# slice of rows

# In[120]:


df.iloc[1:3]


# slice rows using boolean array
# 
# 
# 

# In[121]:


df[np.array([True, False, True])]


# This is most useful when combined with boolean expressions:

# In[122]:


df[df["feature2"] <=5]


# ### Adding and removing columns

# In[123]:


df


# In[124]:


df['feature5'] = 5 - df['feature2'] #adding a column
df['feature6'] = df['feature3'] > 5
df  


# In[125]:


del df['feature6']
df


# In[126]:


df["feature6"] = pd.Series({"feature2": 1, "feature4": 51, "feature1":1}) 
df


# In[127]:


df.insert(1, "feature1b", [0,1,2])
df


# ### Assigning new columns
# create a new DataFrame with new columns

# In[128]:


df


# In[129]:


df10 = df.assign(feature0 = df["feature1"] * df["feature2"] )
df10.assign(feature1 = df10["feature1"] +1)
df10


# ### Evaluating an expression

# In[130]:


df = df[['feature1','feature2','feature3','feature4']]
df.eval("feature1 + feature2 ** 2")


# In[131]:


df


# use inplace=True to modify the original DataFrame
# 
# ---
# 
# 
# 
# 

# In[132]:


df.eval("feature3 = feature1 + feature2 ** 2", inplace=True)
df


# use a local or global variable in an expression by prefixing it with @

# In[133]:


threshold = 30
df.eval("feature3 = feature1 + feature2 ** 2 > @threshold", inplace=True)
df


# ### Querying a DataFrame
# The query method lets you filter a DataFrame

# In[134]:


df.query("feature1 > 2 and feature2 == 6")


# ### Sorting a DataFrame

# In[135]:


df


# In[136]:


df.sort_index(ascending=False)


# In[137]:


df.sort_index(axis=1, inplace=True)
df


# In[138]:


df.sort_values(by="feature2", inplace=True)
df


# ### Operations on DataFrame

# In[139]:


a = np.array([[1,2,3],[4,5,6],[7,8,9]])
df = pd.DataFrame(a, columns=["q", "w", "e"], index=["a","b","c"])
df


# In[140]:


np.sqrt(df)


# In[141]:


df + 1 #broadcasting


# In[142]:


df >= 5


# In[143]:


df.mean(), df.std(), df.max(), df.sum()


# The All method checks whether all values are True or not

# In[144]:


df


# In[145]:


(df > 2).all()


# In[146]:


(df > 2).all(axis = 0) #executed vertically (on each column)


# In[147]:


(df > 2).all(axis = 1) #execute the horizontally (on each row).


# In[148]:


(df == 8).any(axis = 1)


# In[149]:


df - df.mean() 


# In[150]:


df - df.values.mean() # subtracts the global mean elementwise


# ### Handling missing data
# Pandas offers a few tools to handle missing data (NaN).

# In[151]:


df10


# In[152]:


df11 = df10.fillna(0)
df11


# In[153]:


df11.loc["d"] = np.nan
df11.fillna(0,inplace=True)
df11
#grades + fixed_bonus_points


# ### Aggregating with groupby
# Similar to the SQL language, pandas allows grouping your data into groups to run calculations over each group.

# In[154]:


df5 = pd.DataFrame({
    "feature1": {"a": 3, "b": 11, "c": 14, 'd':4},
    "feature2": {"a": 2, "b": 2, "c": 4, 'd':4},
    "feature3": {"a": 32, "b": 4, "c": 3, 'd':35},
    "feature4": {"a": 5, "b": 11, "c": 2, 'd':13}
})
df5


# In[155]:


df5.groupby("feature2").mean()


# ### Pivot tables
# pivot tables allows for quick data summarization

# In[156]:


df9 = df8.stack().reset_index()
df9


# In[157]:


pd.pivot_table(df9, index="level_0")


# In[158]:


pd.pivot_table(df9, index="level_0", values=["rows_ab"], aggfunc=np.max)


# ### functions
# When dealing with large `DataFrames`, it is useful to get a quick overview of its content. Pandas offers a few functions for this. First, let's create a large `DataFrame` with a mix of numeric values, missing values and text values. Notice how Jupyter displays only the corners of the `DataFrame`:

# In[159]:


df = np.fromfunction(lambda x,y: (x+y)%7*11, (10000, 26))
large_df = pd.DataFrame(df, columns=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
large_df.head(5)


# In[160]:


large_df[large_df % 3 == 0] = np.nan
large_df.insert(3,"feature1", "xxx")
large_df.head(5)


# In[161]:


large_df.tail(4)


# In[162]:


large_df.describe()


# ### saving and loading
# 

# In[163]:


large_df


# ### save and load
# 

# In[164]:


large_df.to_csv("my_df.csv")
large_df.to_csv("my_df.xlsx")


# In[165]:


df0 = pd.read_csv("my_df.csv", index_col=0)


# ### combining DataFrames
# pandas has the ability to perform SQL-like joins: inner joins, left/right outer joins and full joins

# In[166]:


city_loc = pd.DataFrame(
    [
        ["CA", "San Francisco", 37.781334, -122.416728],
        ["NY", "New York", 40.705649, -74.008344],
        ["FL", "Miami", 25.791100, -80.320733],
        ["OH", "Cleveland", 41.473508, -81.739791],
        ["UT", "Salt Lake City", 40.755851, -111.896657]
    ], columns=["state", "city", "lat", "lon"])
city_loc


# In[167]:


city_pop = pd.DataFrame(
    [
        [808976, "San Francisco", "California"],
        [8363710, "New York", "New York"],
        [413201, "Miami", "Florida"],
        [2242193, "Houston", "Texas"]
    ], index=[3,4,5,6], columns=["population", "city", "state"])
city_pop


# In[168]:


pd.merge(left=city_loc, right=city_pop, on="city")


# Note that both `DataFrame`s have a column named `state`, so in the result they got renamed to `state_x` and `state_y`.
# 
# Also, note that Cleveland, Salt Lake City and Houston were dropped because they don't exist in *both* `DataFrame`s. This is the equivalent of a SQL `INNER JOIN`. If you want a `FULL OUTER JOIN`, where no city gets dropped and `NaN` values are added, you must specify `how="outer"`:

# In[169]:


all_cities = pd.merge(left=city_loc, right=city_pop, on="city", how="outer")
all_cities


# Of course `LEFT OUTER JOIN` is also available by setting `how="left"`: only the cities present in the left `DataFrame` end up in the result. Similarly, with `how="right"` only cities in the right `DataFrame` appear in the result. For example:

# In[170]:


pd.merge(left=city_loc, right=city_pop, on="city", how="right")


# If the key to join on is actually in one (or both) `DataFrame`'s index, you must use `left_index=True` and/or `right_index=True`. If the key column names differ, you must use `left_on` and `right_on`. For example:

# In[171]:


city_pop2 = city_pop.copy()
city_pop2.columns = ["population", "name", "state"]
pd.merge(left=city_loc, right=city_pop2, left_on="city", right_on="name")


# ### Concatenation
# Rather than joining `DataFrame`s, we may just want to concatenate them. That's what `concat()` is for:

# In[172]:


result_concat = pd.concat([city_loc, city_pop])
result_concat


# Note that this operation aligned the data horizontally (by columns) but not vertically (by rows). In this example, we end up with multiple rows having the same index (eg. 3). Pandas handles this rather gracefully:

# In[173]:


result_concat.loc[3]


# Or you can tell pandas to just ignore the index:

# In[174]:


pd.concat([city_loc, city_pop], ignore_index=True)


# Notice that when a column does not exist in a `DataFrame`, it acts as if it was filled with `NaN` values. If we set `join="inner"`, then only columns that exist in *both* `DataFrame`s are returned:

# In[175]:


pd.concat([city_loc, city_pop], join="inner")


# You can concatenate `DataFrame`s horizontally instead of vertically by setting `axis=1`:

# In[176]:


pd.concat([city_loc, city_pop], axis=1)


# In this case it really does not make much sense because the indices do not align well (eg. Cleveland and San Francisco end up on the same row, because they shared the index label `3`). So let's reindex the `DataFrame`s by city name before concatenating:

# In[177]:


pd.concat([city_loc.set_index("city"), city_pop.set_index("city")], axis=1)


# This looks a lot like a `FULL OUTER JOIN`, except that the `state` columns were not renamed to `state_x` and `state_y`, and the `city` column is now the index.

# The `append()` method is a useful shorthand for concatenating `DataFrame`s vertically:

# In[178]:


city_loc.append(city_pop)


# As always in pandas, the `append()` method does *not* actually modify `city_loc`: it works on a copy and returns the modified copy.

# ### Categories
# It is quite frequent to have values that represent categories, for example `1` for female and `2` for male, or `"A"` for Good, `"B"` for Average, `"C"` for Bad. These categorical values can be hard to read and cumbersome to handle, but fortunately pandas makes it easy. To illustrate this, let's take the `city_pop` `DataFrame` we created earlier, and add a column that represents a category:

# In[179]:


city_eco = city_pop.copy()
city_eco["eco_code"] = [17, 17, 34, 20]
city_eco


# Right now the `eco_code` column is full of apparently meaningless codes. Let's fix that. First, we will create a new categorical column based on the `eco_code`s:

# In[180]:


city_eco["economy"] = city_eco["eco_code"].astype('category')
city_eco["economy"].cat.categories


# Now we can give each category a meaningful name:

# In[181]:


city_eco["economy"].cat.categories = ["Finance", "Energy", "Tourism"]
city_eco


# Note that categorical values are sorted according to their categorical order, *not* their alphabetical order:

# In[182]:


city_eco.sort_values(by="economy", ascending=False)


# 
# # Chapter -2: Linear Algebra

# ##Scalars, Vectors, Matrices
# 
# Values consisting of one numerical quantity are called **scalars**. 
# For example, the average annual temperature in New York City during is $53^\circ$F. 
# In the following, we adopt the mathematical notation where **scalar** variables are denoted by ordinary lower-cased letters (e.g., $x, y,$ and $z$). 
# 
# We denote the space of real-valued scalars by $\mathbb{R}$. Formally, $x \in \mathbb{R}$ means that $x$ is a real-valued scalar, where the symbol $\in$ is pronounced "in," and simply denotes membership in a set. 
# 
# 
# 

# In[183]:


a = 1 #scalar
b = 2.5 #scalar
a, b


# **Vector** is simply a list (or an array) of scalar values.
# We denote vectors in bold, lower-case letters (e.g., $\mathbf{x}$). 
# Vectors can be presented either as a **column** or as a **row**.
# In example,
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

# In[184]:


import numpy as np
x = np.array([1, 2, 3, 4]) #vector, 1-d array
x


# In[185]:


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

# In[186]:


A = np.array([[1, 2], [3, 4], [5, 6]]) #3x2 matrix 
A


# In[187]:


A.shape


# In[188]:


A[2,1]


# We can exchange a matrix's rows and columns and the result is called the **transpose** of the matrix.
# Formally, we denote the transpose of $\mathbf{A}$ by $\mathbf{A}^\top$.
# Thus, the transpose of $\mathbf{A}$ is
# a $n \times m$ matrix.
# 

# In[189]:


A_T = A.T # Vector transposition
A_T


# In[190]:


A_T.shape


# As a special type of a square matrix is the **symmetric** matrix $\mathbf{A}$ in which its transpose is equal to itself. That is, $\mathbf{A}=\mathbf{A}^\top $.  

# In[191]:


A=np.array([[1,5,0],[5,2,0],[0,0,3]])
print(A==A.T)


# Matrices are useful data structures, and they allow us to organize data efficiently. As an example, it is common to organize data in a way that the **rows** of a matrix correspond to different measurments, while the **columns** correspond to the various features or attributes. 

# ## Arithmetic Operations
# 
# Adding/subtracting scalars, vectors, or matrices of the same shape is performed **elementwise**. It is also possible to multiply/divide scalars, vectors, or matrices elementwise.

# In[192]:


A = np.array([[1, 2], [3, 4], [5, 6]])
A


# In[193]:


B = np.array([[2, 5], [7, 4], [4, 3]])
B


# In[194]:


A+B # elementwise addition


# In[195]:


A-B # elementwise subtraction


# In[196]:


A*B # elementwise multiplication, aka Hadamard product


# In[197]:


A/B # elementwise division


# ##Reduction
# A common operation that can be performed on arbitrary matrices
# is to **sum** elements.
# Formally, we express sums using the $\sum$ symbol.
# To express the sum of the elements in a vector $\mathbf{x}$ of length $n$,
# we write $\sum_{i=1}^n x_i$.
# 

# In[198]:


x = np.arange(10)
x, x.sum()


# In[199]:


A = np.array([[1, 2], [3, 4], [5, 6]])


# The sum of **all** the elements in $\mathbf{A}$ is denoted $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.

# In[200]:


A.sum()


# Summing up elements along **rows** reduces the row dimension (axis 0) $\sum_{i=1}^{m} a_{ij}$.

# In[201]:


A.sum(axis=0)


# Summing up elements along **columns** reduces the row dimension (axis 1) $\sum_{j=1}^{n} a_{ij}$.

# In[202]:


A.sum(axis=1)


# ##Broadcasting

# Broadcasting describes how Numpy handles arrays with different shapes during arithmetic operations. Subject to certain constraints, the smaller array is "broadcast" across the larger array so that they have compatible shapes. 
# NumPy operations are done on **pairs** of arrays on an element-by-element basis. 
# 
# The simplest broadcasting example occurs when an array and a scalar value are combined in an operation

# In[203]:


A = np.array([1, 2,3])
A+4 # broadcasting


# This result is equivalent to 

# In[204]:


A + np.array([4, 4, 4])


# When operating on two arrays, NumPy compares their shapes element-wise. It starts with the **rightmost** dimensions and works its way left. Two dimensions are compatible when
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
# 
# 
# 
# 

# ##Multiplying Vectors and Matrices
# 
# One of the fundamental operations in Linear Algebra is the **dot product**.
# Given two vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$, their dot product $\mathbf{x} \cdot \mathbf{y}=\mathbf{x}^\top \mathbf{y}$ is a sum over the elements at the same positions. In example, $\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{n} x_i y_i$ resulting in a scalar.

# In[205]:


x = np.ones(4); y = 2*np.ones(4)
x, y, np.dot(x, y)


# The dot product has a geometric interpretation related to the **angle** between the two vectors
# 
# $$
# \mathbf{x}\cdot\mathbf{y} = \mathbf{x}^\top\mathbf{y} = \|\mathbf{x}\|\|\mathbf{y}\|\cos(\theta),
# $$
# 
# where $\| \cdot \|$ is the **norm** operator applied to tha vector. Norms will be discussed later on. 
# 
# In the Machine Learning contexts, the angle $\theta$ is often used to measure the closeness, or similarity, of two vectors. This can be computed simply via
# 
# $$
# \cos(\theta) = \frac{\mathbf{x}^\top\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}.
# $$
# 
# This (cosine) similarity takes values between -1 and 1. A maximum value of $1$
# when the two vectors point in the same direction, a minimum value of $-1$ when pointing in opposite directions, and a value of $0$ when orthogonal.

# In[206]:


np.arccos(x.dot(y) / (np.linalg.norm(x) * np.linalg.norm(y)))


# Consider now the dot product of a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ and a vector $\mathbf{x} \in \mathbb{R}^n$.
# We can write the matrix $\mathbf{A}$ in terms of its **row** vectors
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
# whose $i^\mathrm{th}$ element is the dot product $\mathbf{a}^\top_i \mathbf{x}$
# 
# $$
# \mathbf{A}\mathbf{x}
# = \begin{bmatrix}
# \mathbf{a}^\top_{1} \\
# \mathbf{a}^\top_{2} \\
# \vdots \\
# \mathbf{a}^\top_m \\
# \end{bmatrix}\mathbf{x}
# = \begin{bmatrix}
#  \mathbf{a}^\top_{1} \mathbf{x}  \\
#  \mathbf{a}^\top_{2} \mathbf{x} \\
# \vdots\\
#  \mathbf{a}^\top_{m} \mathbf{x}\\
# \end{bmatrix}.
# $$
# 
# To make it clear, in the above example there are $m$ dot products, each $\mathbf{a}^\top_i \mathbf{x}$ which is a multiplcation on $n$-sized $\mathbf{a}^\top_i$ by $n$-sized $\mathbf{x}$. The result of each dot product is a scalar.
# 
# We can think of the multiplication of $\mathbf{A}\in \mathbb{R}^{m \times n}$ by $\mathbf{x}\in \mathbb{R}^n$ as a transformation that projects vectors, $\mathbf{x}\in \mathbb{R}^n$,
# from $\mathbb{R}^{n}$ to $\mathbb{R}^{m}$.
# Note that the column dimension of $\mathbf{A}$ (its length along axis 1)
# must be the same as the dimension of $\mathbf{x}$.
# 

# In[207]:


A = np.array([[1, 2], [3, 4], [5, 6]])
x = np.arange(1,3)
np.dot(A, x)


# Next we consider matrix-matrix multiplication.
# Given two matrices $\mathbf{A} \in \mathbb{R}^{n \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times m}$:
# 
# $$\mathbf{A}=\begin{bmatrix}
#  a_{11} & a_{12} & \cdots & a_{1k} \\
#  a_{21} & a_{22} & \cdots & a_{2k} \\
# \vdots & \vdots & \ddots & \vdots \\
#  a_{n1} & a_{n2} & \cdots & a_{nk} \\
# \end{bmatrix},\quad
# \mathbf{B}=\begin{bmatrix}
#  b_{11} & b_{12} & \cdots & b_{1m} \\
#  b_{21} & b_{22} & \cdots & b_{2m} \\
# \vdots & \vdots & \ddots & \vdots \\
#  b_{k1} & b_{k2} & \cdots & b_{km} \\
# \end{bmatrix}.$$
# 
# 
# Denote by $\mathbf{a}^\top_{i} \in \mathbb{R}^k$
# the row vector representing the $i^\mathrm{th}$ row of the matrix $\mathbf{A}$,
# and let $\mathbf{b}_{j} \in \mathbb{R}^k$
# be the column vector of the $j^\mathrm{th}$ column of the matrix $\mathbf{B}$.
# The matrix product $\mathbf{C} = \mathbf{A}\mathbf{B}$, is easiest to understand with $\mathbf{A}$ specified by its row vectors and $\mathbf{B}$ specified by its column vectors,
# 
# $$\mathbf{A}=
# \begin{bmatrix}
# \mathbf{a}^\top_{1} \\
# \mathbf{a}^\top_{2} \\
# \vdots \\
# \mathbf{a}^\top_n \\
# \end{bmatrix},
# \quad \mathbf{B}=\begin{bmatrix}
#  \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
# \end{bmatrix}.
# $$
# 
# 
# The matrix product $\mathbf{C}\in \mathbb{R}^{n \times m}$ is produced as we simply compute each element $c_{ij}$ as the dot product $\mathbf{a}^\top_i \mathbf{b}_j$:
# 
# $$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
# \mathbf{a}^\top_{1} \\
# \mathbf{a}^\top_{2} \\
# \vdots \\
# \mathbf{a}^\top_n \\
# \end{bmatrix}
# \begin{bmatrix}
#  \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
# \end{bmatrix}
# = \begin{bmatrix}
# \mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
#  \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
#  \vdots & \vdots & \ddots &\vdots\\
# \mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
# \end{bmatrix}.
# $$
# 
# 

# In[208]:


A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[5, 3], [2, 2]])
A.dot(B) 


# Matrix mutliplication is **distributive**, i.e. $\mathbf{A}(\mathbf{B}+\mathbf{C})=\mathbf{A}\mathbf{B}+\mathbf{A}\mathbf{C}$:

# In[209]:


A = np.array([[2, 3], [1, 4], [7, 6]])
B = np.array([[5], [2]])
C = np.array([[4], [3]])
A.dot(B+C) == A.dot(B)+A.dot(C)


# Matrix mutliplication is **associative**, i.e. $\mathbf{A}(\mathbf{B}\mathbf{C})=(\mathbf{A}\mathbf{B})\mathbf{C}$:

# In[210]:


A = np.array([[2, 3], [1, 4], [7, 6]])
B = np.array([[5, 3], [2, 2]])
C = np.array([[4], [3]])
A.dot(B.dot(C)) == (A.dot(B)).dot(C)


# Vector multiplication is commutative, i.e. $\mathbf{x}^\top\mathbf{y}=\mathbf{y}^\top\mathbf{x}$

# In[211]:


x = np.array([[2], [6]])
y = np.array([[5], [2]])
x.T.dot(y) == y.T.dot(x)


# However, matrix multiplication is **not commutative**, i.e. $\mathbf{A}\mathbf{B}\ne\mathbf{B}\mathbf{A}$:

# In[212]:


A = np.array([[2, 3], [6, 5]])
B = np.array([[5, 3], [2, 2]])
np.dot(A, B) == np.dot(B, A)


# ##Norms
# 
# One of the important operators in linear algebra are the **norms**.
# The norm of a vector tells you how large a vector is by maping a vector
# to a scalar. This scalar measures, not the dimensions, but the magnitude of the components.
# 
# Formally, given any vector $\mathbf{x}$, the vector norm is a function $f$ that has multiple propeerties: 
# 
# The first is a scaling property that says
# that if we scale all the elements of a vector
# by a constant factor $\alpha$,
# its norm also scales by the *absolute value*
# of the same constant factor. In example,
# 
# $$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$
# 
# The second property is the triangle inequality:
# 
# $$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$
# 
# 
# The third property simply says that the norm must be non-negative:
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

# In[213]:


import numpy as np
u = np.array([1,2,3]) # L2 norm
np.linalg.norm(u, ord=2)


# The $L_1$ norm is defined as the sum of the absolute values of the vector elements:
# $$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$
# In more generality, the $L_p$ norm is defined as 
# $$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

# In[214]:


u = np.array([1,2,3])
np.linalg.norm(u, ord=1) # L1 norm


# In[215]:


u = np.array([1,2,3])
np.linalg.norm(u, ord=np.inf) # L-inf norm, basically choosing the maximum value within the array


# ##Identity matrices

# In[216]:


A = np.eye(3)
A


# In[217]:


x = np.array([[2], [6], [3]])
x


# In[218]:


b = A.dot(x)
b


# ##Inverse matrices¶
# 

# In[219]:


A = np.array([[3, 0, 2], [2, 0, -2], [0, 1, 1]])
A


# In[220]:


A_inv = np.linalg.inv(A)
A_inv


# In[221]:


A_inv.dot(A)


# ## Hyperplanes
# 
# A key object in linear algebra is the **hyperplane**.
# In two dimensions the hyperplane is a **line**, and in three dimensions it is a **plane**. In more generality, in an $d$-dimensional vector space, a hyperplane has $d-1$ dimensions
# and **divides the space into two half-spaces**. 
# In the Machine Learning language in the contex of classification hyperplanes that separate the space are named **decision planes**. 
# 
# In example, $\mathbf{a}=(1,2)$ and $\mathbf{a} \cdot \mathbf{b}=1$ define the line $2y = 1 - x$ in the two dimensional space, while $\mathbf{a}=(1,2,3)$ and $\mathbf{a} \cdot \mathbf{b}=1$ defines the plane $ z= 1 - x-2y$ in the three dimensional space.
# 
# 
# 

# ##solving a system of linear equations¶
# 

# In[222]:


A = np.array([[2, -1], [1, 1]])
b = np.array([[0], [3]])
x = np.linalg.inv(A).dot(b)
x


# Singular matrices are not invertible and are called singular.

# ## Linear equations

# The set of equations
# 
# $$\mathbf{A}\mathbf{x}=\mathbf{b}$$
# 
# corresponds
# 
# $$
# \begin{align}
# a_{1,1}x_1 + a_{1,2}x_2 + \cdots + a_{1,n}x_n = b_1 \\\\
# a_{2,1}x_1 + a_{2,2}x_2 + \cdots + a_{2,n}x_n = b_2 \\\\
# \cdots \\\\
# a_{m,1}x_1 + a_{m,2}x_2 + \cdots + a_{m,n}x_n = b_n
# \end{align}
# \quad\text{or}\quad
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
# $$
# 
# Here we have $m$ equations representing..., with $n$ unknowns representing the **dimension space**. 
# Commonly, we are given $\mathbf{A}$ and $\mathbf{b}$, and we solve for $\mathbf{x}$ that satisfy the above equations.
# 
# How many solutions does $\mathbf{A}\mathbf{x}=\mathbf{b}$ have? There are three possibilites:
# 
#  1. No solution
#  2. One solution
#  3. Infinite number of solutions
#  
# Why can't there be more than one solution and less than an infinite number of solutions ?
# 
# On the intuition level, it is because we are working with linear equations.
# It is clear if we consider the simple scenario of two equations and two unknowns: the solution of this system corresponds to the intersection of the lines. One possibility is that the two lines never cross (parallel). Another possibility is that they cross exactly once. And the last possibility is that they cross everywhere (superimposed).

# There are two ways to describe a system of linear equations: the **row view** and the **column view**.

# ## Row view
# 
# The row view is probably more intuitive because it is the representation used when we have only one equation. It can now be extended to an infinite number of equations and unknowns.
# 
# We said that the solutions of the linear system of equations are the sets of values of $x_1...x_n$ that satisfies all equations. For instance, in the case of $\mathbf{A}\in \mathbb{R}^{m \times n}$ with $m=n=2$ the equations correspond to 2 lines in the 2-dimensional space and the solution of the system is the intersection of these lines.
# 
# As mentioned before, a linear system can be viewed as a set of $(n-1)$-dimensional hyperplanes in a $n$-dimensional space. The system can also be characterized by its number of equations $m$, and the number of unknown variables $n$.
# 
# * If $m \gt n$, that is there are more equations than unknows the system is called **overdetermined**. In example, a system of 3 equations (represented by 3 lines) and 2 unknowns (corresponding to 2 dimensions). In this example, if the lines are independent, there is no solution since there is no point belonging to the three lines.
# 
# * If $m\lt n$, that is there are more unknowns than equations the system is named **underdetermined**. In example, there is only 1 equation (1 line) and 2 dimensions. In this case, each point along the line is a solution to the system and the system has infinite number of solutions.
# 

# ### Example 1: 1 equation and 2 variables
# 
# Let us start with an **underdetermined** example with  $m=1$ and  $n=2$:
# 
# $$
# a_{1,1}x_1 + a_{1,2}x_2 = b_1
# $$
# 
# The graphical interpretation of $n=2$ is that we have a 2-dimensional space. So we represent it with 2 axes. Since our hyperplane is of $n-1$-dimensional, we have a 1-dimensional hyperplane. This is simply a line. As $m=1$, we have only one equation. This means that we have only one line characterizing our linear system.
# 
# Note that the last equation can also be written in a way that may be more usual:
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
# The graphical interpretation of $n=2$ is as before. That is, we have a 2-dimensional space, represented with 2 axes, and each equation represents an $1$-dimensional hyperplane which is simply a line. 
# 
# But, here $m=2$, which means that we have two equations, or 2 lines in a 2-dimensional space.
# There are multiple scenarios:
# 
# * The two lines run in parallel and there is no solution.
# * The two lines intersect and we have one unique solution.
# * The two lines superimpose: it is the same equation or linearily dependant. In this case there are infinite number of solutions since each and every point on the lines corresponds to a solution.
# 
# Same conlusioned can be drawn with higher values of $m$ (number of equations) and $n$ (number of dimensions). For example, two 2-dimensional plane in a 3-dimensional space can be 
# * parallel (no solution),
# * cross (infinitely many solutions since their crossing is a line), or
# * superimposed (infinitely many solutions). 

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
# The graphical interpretation of $n=2$ is as before. That is, we have a 2-dimensional space, represented by 2 axes, and each equation represents an $1$-dimensional hyperplane which is simply a line. 
# 
# Here $m=3$, which means that we have three equations, or 3 lines in a 2-dimensional space.
# 
# There are multiple scenarios:
# * The three lines run in parallel and there is no solution.
# * The three lines intersect. If the three lines are independent, there are multiple intersections with no unique solution.
# * Two lines are dependent and the third is independent: we have a unique solution at the intersection of them with the third line.
# * The three lines superimpose: there are infinite number of solutions.
# 
# As before, these conlusioned can be generalized to higher spaces.

# ## Linear combination
# 
# The linear combination of 2 vectors corresponds to their weighted sum.
# In example:
# 
# $$
# \mathbf{x}^\top=(1,2,3),~\mathbf{y}^\top=(4,5,6),\quad \text{and}\quad \mathbf{y}^\top=a\mathbf{x}^\top+b\mathbf{y}^\top,\text{with}~ a=1,~b=2.
# $$
# 

# In[223]:


a=1.;b=2.;
a*np.array([1,2,3])+b*np.array([4,5,6])


# ## Span
# 
# Consider the above $\mathbf{x}$ and $\mathbf{y}$: ALL the points that can reached by their combination via some $a$ and $b$ are called the set of points is the span of the vectors $\{\mathbf{x}$, $\mathbf{y}\}$.
# Vector **space** means all the points that can be reached by this vector. In example, the space of all real numbers is $\mathbb{R}$ and for $n$-dimensions the space is $\mathbb{R}^n$. 
# Vector **subspace** means that the space occupies only part of the space. In example, a 2-dimensional plane is a subspace in $\mathbb{R}^3$ and a line is a subspace in $\mathbb{R}^2$.
# 
# In general, we will say that a collection of vectors
# $\mathbf{x}_1,\mathbf{x}_2, \ldots, \mathbf{x}_n$ are linearly **dependent**
# if there exist coefficients $a_1,a_2 \ldots, a_n$ not all zero such that
# 
# $$
# \sum_{i=1}^n a_i\mathbf{x_i} = 0.
# $$
# 
# In this case, we can solve for one of the vectors in terms of a combination of the others, and thus this vector is redundant.
# Similarly, a linear dependence in the columns of a matrix means that this matrix can be compresed down to a lower dimension. If there is no linear dependence we say the vectors are linearly independent. In a matrix, if the columns are linearly independent, the matrix cannot be compressed to a lowed dimension.
# 
# A set of vectors which is linearly **independent** and **spans** some vector space, forms a **basis** for that vector space. 
# The **standard basis** (also called natural basis or canonical basis) in $\mathbb{R}^n$ is the set of vectors whose components are all zero, except one that equals 1. For example, the standard basis in $\mathbb{R}^2$ is formed by the vectors $\mathbf{x}^\top=(1,0)^\top$ and $\mathbf{y}^\top=(0,1)^\top$.
# 
# The linear combination of vectors in a space stay in the same space. For instance, any linear combination of two lines in a $\mathbb{R}^2$ result in another vector in $\mathbb{R}^2$.

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
# $$
# 
# In this view, the solution $\mathbf{b}$ is a linear combination of the columns of $\mathbf{A}$, weighted by the components of $\mathbf{x}$.

# * For the **underdetermined** system, $n \gt m$ and the system has **infinite number of solutions**. As an example in the column view,
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
# But, if the columns of $\mathbf{A}$ are independent, two of them are enough to reach any point in $\mathbb{R}^2$. So, two components out of the three in $\mathbf{x}$ are set and the third is free, meaning that there are infinite number of solutions.
# 
# * For the **overdetermined** system $m\gt n$ and there is **no solution**. For example using the column view,
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
# In the row view we have 3 lines and we are looking for unique intersection in the 2-dimensional plan. The three lines (if independent) will intesect in multiple points resulting in no solution. In the column view, becuase $n=2$ the linear combination of two 3-dimensional vectors is not enough to span the 3-dimensional space, unless the vecotr $\mathbf{b}$ lies, for some reason, in the subspace formed by these two vectors.

# ### Linear dependency
# 
# The number of columns can thus provide information on the number of solutions. But the number that we have to take into account is the number of linearly **independent** columns. Columns are linearly **dependent** if one of them is a linear combination of the others. In the column view, the direction of two linearly dependent vectors is the same and this doesn't add value in spanning the space.
# 
# As an example,
# 
# $$
# \begin{align}
# x_1+2x_2=b_1\\
# 2x_1+4x_2=b_2
# \end{align}
# \quad\text{which in the column view is}\quad
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
# \end{bmatrix}
# $$.
# 
# The columns $(1,2)^\top$ and $(2,4)^\top$ are **dependent** and hence their linear combination is not enough to span the full $\mathbb{R}^2$ and reach all points in this space. If $\mathbf{b}=(3,6)^\top$ there is a solution $\mathbf{x}^\top=(1,1)^\top$ because the vector $(1,2)^\top$ spans a subspace of $\mathbb{R}^2$ that contains the vector $\mathbf{b}=(3,6)^\top$.
# But for more general solution, as say $\mathbf{b}=(3,7)$, there is no solution 
# as linear combinations of $(1,2)^\top$ are not enough to reach all points in $\mathbb{R}^2$. This is an example of an **overdetermined** system with $m=2\gt n=1$ because of the linear dependency between the columns.
# 

# ### Square matrix
# 
# When $\mathbf{A}\in\mathbb{R}^{m\times n}$ and $m=n$ the matrix $\mathbf{A}$ is called square matrix and if the columns are linearly **independant** there is a unique solution to $\mathbf{A}\mathbf{x}=\mathbf{b}$.
# The solution is simply $\mathbf{x}=\mathbf{A}^{-1}\mathbf{b}$, where $\mathbf{A}^{-1}$ is the inverse of $\mathbf{A}$.
# 
# When $\mathbf{A}^{-1}$ exist, we say that $\mathbf{A}$ is invertible (aka, nonsingular, or nondegenerate). 
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

# ### Rank
# 
# The **column rank** of $\mathbf{A}$ is the dimension of the column space of $\mathbf{A}$ (i.e., the number of linearly independent columns), while the **row rank** of $\mathbf{A}$ is the dimension of the row space of $\mathbf{A}$ (i.e., the number of linearly independent rows).
# 
# A fundamental result in linear algebra is that the column rank and the row rank are **always equal** and this number is simply called the rank of $\mathbf{A}$.
# A matrix $\mathbf{A}\in\mathbb{R}^{m\times n}$ is said to have **full rank** if it's rank equals to $\min(m, n$); otherwise, the matrix is **rank deficient**.

# In[224]:


A=np.array([[1,2],[3,4]]); B=np.array([[1,2],[2,4]]); C=np.array([[1,0,0],[0,1,0],[0,0,1]])
np.linalg.matrix_rank(A), np.linalg.matrix_rank(B), np.linalg.matrix_rank(C)


# ### Invertibility
# 
# An $n\times n$ square matrix $\mathbf{A}$ is called **invertible** (also **nonsingular** or **nondegenerate**), if there exists an $n\times n$ square matrix $\mathbf{B}$ such that
# $$
# \mathbf{A}\mathbf{B}=\mathbf{B}\mathbf{A}=\mathbf{I}_n
# $$
# 
# where $\mathbf{I}_n$ denotes the $n\times n$ identity matrix. In this case, then the matrix $\mathbf{B}$ is uniquely determined by $\mathbf{A}$, and is called the inverse of $\mathbf{A}$, denoted by $\mathbf{A}^{-1}$.
# 
# A square matrix that is not invertible is called **singular** or **degenerate**. Non-square matrices ($m\times n$ matrices with $n\ne n$) do not have an inverse (but they may have a left inverse or right inverse). 
# A square matrix is singular if and only if its determinant is zero. In practice, singular square matrices are quite rare. 
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
# We can test to see this by seeing that multiplying
# by the inverse given by the formula above works in practice.
# 

# # Chapter -1: Probability and Statistics
# 
# Probability and statistics and constitute of dealing with uncertainty in desicion making. The theory of probability provides the basis for the learning from data.

# In[225]:


import numpy as np
from scipy import stats
import scipy


# ## Fundamentals
# Starting with a few definitions. A **sample space** is the set of all the possible experimental outcomes. And, **experiment**, is defined to be any process for which more than one **outcome** is possible.
# 
# Imagine a sample space of $n$ possible outcomes $S = \{ O_1, O_2, \cdots, O_n \}$. We assign a probability $p_i$ to each outcome $O_i$
# 
# \begin{equation}
#     P(O_i) = p_i.
# \end{equation}
# 
# All $p_i$ must satisfy 
# \begin{equation}
#    0 \leq p_i \leq 1, \hspace{0.5cm} \forall i= 1,2, \cdots, n.
# \end{equation}
# 
# If all $p_i$ are equal to a constant $p$ we say that all $n$ outcomes are equally likely, and each probability has a value of $1/n$.
# 
# An **event** is defined to be a subset of the sample space. The probability of an event $A$, $P(A)$, is obtained by summing the probabilities of the outcomes contained withing the event $A$. An event is said to occur if one of the outcomes contained within the event occurs. The complement of event $A$ is the event $A^C$ and is defined to be the event consisting of everything in the sample space $S$ that is not contained within $A$. In example,
# 
# \begin{equation}
#     P(A) + P(A^C) = 1
# \end{equation}
# 
# **Intersections** of events, $A \cap B$, consists of the outcomes contained within both events $A$ and $B$. The probability of the intersection, $P(A \cap B)$, is the probability that both events occur simultaneously.
# A few known properties of the intersections of events:
# \text{For mutually exclusive events:}~
# 
# \begin{align}
#  & P(A \cap B) +P(A \cap B^C) = P(A)\\
# &A \cap (B \cap C) = (A \cap B) \cap C\\
# & A \cap B = \emptyset \quad\text{(for mutually exclusive events)}\\
# \end{align}
# 
# The **union** of events, $ A\cup B$, consists of the outcomes that are contained within at least one of the events $A$ and $B$. The probability of this event, $P (A \cup B)$ is the probability that at least one of these events $A$ and $B$ occurs.
# A few known properties of the union of events:
# 
# \begin{align}
# & A \cup A^C = S \\
# & (A \cup B)^C = A^C \cap B^C\\
# & (A \cap B)^C = A^C \cup B^C\\
# & A \cup (B \cup C) = (A \cup B) \cup C \\
# & P(A \cup B) = P(A) + P(B) = \emptyset \quad\text{(for mutually exclusive events)}\\
# & P( A \cup B) = P(A \cap B^C) + P(A^C \cap B^C) + P(A \cap B)\\
# \end{align}
# 
# The union of three events is equal to 
# \begin{align}
# P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P( B \cap C) - P( A \cap C) + P(A \cap B \cap C).
# \end{align}
# If the events union is **mutually exclusive** then
# \begin{align}
#     P(A_1 \cup A_2 \cup \cdots \cup A_n) = P(A_1) + \cdots + P(A_n),
# \end{align}
# where the sequence $A_1, A_2, \cdots , A_n$ are called the **partition** of $S$.
# 
# **Conditional Probability** is defined as an event $B$ that is conditioned on another event $A$. In this case,
# \begin{align}
#     P(B \mid A) = \frac{P(A \cap B)}{P(A)} \hspace{0.5cm}  \text{for } P(A) >0.
# \end{align}
# 
# From the above equation, it follows that 
# \begin{align}
# P(A \cap B) = P (B \mid A) P(A).
# \end{align}
# 
# It's not hard to see that conditioning on more evets (e.g. two) results in
# \begin{align}
# P(A \cap B\cap C) = P (C \mid B\cap A) P(B\cap A).
# \end{align}
# 
# In general, for a sequence of events $A_1, A_2, \cdots, A_n$:
# \begin{align}
# \mathrm {P} (A_{n}\cap \ldots \cap A_{1})=\mathrm {P} (A_{n}|A_{n-1}\cap \ldots \cap A_{1})\cdot \mathrm {P} (A_{n-1}\cap \ldots \cap A_{1}).
# \end{align}
# 
# If the two events $A$ and $B$ are independent, knowledge about one event does not affect the probability of the other event. The following conditions are equivalent:
# \begin{align}
# P(A \mid B) &= P(A)\\
# P(A \cap B) &= P(A)P(B).\\
# \end{align}
# 
# In general, if $A_1, A_2, \cdots, A_n$ are independent then
# \begin{align}
# P(A_1 \cap A_2  \ldots \cap A_n) = P(A_1)P(A_2) \cdots P(A_n).
# \end{align}
# 

# The law of total probability states that given a partition of the sample space $B$ to $n$ non-overlapping segments $\{ A_1, A_2, \cdots, A_n \}$ the probability of an event $B$, $P(B)$ can be expressed as:
# \begin{align}
#     P(B) = \sum_{i=1}^n P(A_i)P(B \mid A_i)
# \end{align}
# 
# And finally, Bayes' theorem is infered from the conditional probability equations $P(A|B)=P(A\cap B)/P(B)$ and $P(B|A)=P(B\cap A)/P(A)$. Because, $P(A\cap B)=P(B\cap A)$ it follows that
# \begin{align}
#     P(A \mid B) = \frac{P(B \mid A) P(A) }{ P(B)}.
# \end{align}
# If $B$
# - Given $\{ A_1, A_2, \cdots, A_n \}$ a partition of a sample space, then the posterior probabilities of the event $A_i$ conditional on an event $B$ can be obtained from the probabilities $P(A_i)$ and $P(A_i \mid B)$ using the formula:
# \begin{equation}
#     P(A_i \mid B) = \frac{P(A_i)P(B \mid A_i)}{\sum_{j=1}^n P(A_j)P(B \mid A_j)}
# \end{equation}

# ### Bayes' Theorem

# - Given $\{ A_1, A_2, \cdots, A_n \}$ a partition of a sample space, then the posterior probabilities of the event $A_i$ conditional on an event $B$ can be obtained from the probabilities $P(A_i)$ and $P(A_i \mid B)$ using the formula:
# \begin{equation}
#     P(A_i \mid B) = \frac{P(A_i)P(B \mid A_i)}{\sum_{j=1}^n P(A_j)P(B \mid A_j)}
# \end{equation}

# In[ ]:





# 

# 
# # Chapter 0: Introduction
# 
# 

# Welcome to our short introductory course on Operation Research using Python!
# 
# Operations research (OR), (a.k.a., Management Science) is an interdisciplinary field that uses analytical methods for quantitative decision-making.
# The emphasis of OR is on practical applications, and as such, it has substantial overlap with many other disciplines, like industrial engineering and business management.
# It uses tools from mathematical sciences, such as modeling, statistics, and optimization, to arrive at optimal or near-optimal solutions to complex decision-making problems. Operations research is often concerned with determining the extreme values of real-world objectives, such as maximizing profit or minimizing cost.
# 
# Operations research is used to deal with real-world problems. For example:
# * Scheduling: hospital patients, classes, buses, planes, sporting events.
# * Marketing: store layout, advertising, social media, online ad placement, recommendations on a website.
# * Product development: product features, pricing, sales forecasts.
# * Inventory: how many to build, how many touchpads the store should have in stock.
# * Organizations: business management, cross-cultural issues, social networks.
# * Queueing: waiting for lines at amusement parks, banks, movie theaters, the line at the store to buy new electronic gadgets, traffic.
# * Environment: managing sustainable resources, reducing materials needed to manufacture a product.
# * Optimizing: internet search engines, product design.
# * Decision making: security, investment, what college to attend.
# 
# <!-- The following set of notes will emphasize the use of Python in solving OR-related problems. -->
# 
# <!-- ## Python and Jupyter Resources -->
# <!-- 
# * [Google Colab Introduction](https://colab.research.google.com/notebooks/welcome.ipynb)
# * [Google's Python Class](https://developers.google.com/edu/python/)
# * [Stanford Numpy Tutorial](https://cs231n.github.io/python-numpy-tutorial/)
# * [Linear algebra](https://github.com/ageron/handson-ml2/blob/master/math_linear_algebra.ipynb)
# * [Numpy](https://github.com/ageron/handson-ml2/blob/master/tools_numpy.ipynb)
# * [Matplotlib](https://github.com/ageron/handson-ml2/blob/master/tools_matplotlib.ipynb)
# * [Pandas](https://github.com/ageron/handson-ml2/blob/master/tools_pandas.ipynb)
# 
# <!-- * [Jupyter Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook) -->
# <!-- * [Calculus](https://github.com/ageron/handson-ml2/blob/master/math_differential_calculus.ipynb) -->
# <!-- * [Matplotlib Tutorial Notebook](https://nbviewer.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-4-Matplotlib.ipynb) -->
# <!-- * [Python Data Science Handbook](https://github.com/jakevdp/PythonDataScienceHandbook) -->
# <!-- * [Scikit-Learn Documentation](https://scikit-learn.org/stable/index.html) --> 
# 

# # Chapter 1: Linear Programming

# Objectives of business decisions frequently involve maximizing profit or minimizing costs.
# Linear programming uses linear algebraic relationships to represent a firm's decisions given a business objective and resource constraints.
# Steps in application:
# 1. Identify the problem as solvable by linear programming
# 2. Formulate a mathematical model of the unstructured problem
# 3. Solve the model
# 
# The model has multiple components:
# * Decision variables - mathematical symbols representing levels of activity by the firm
# * Objective function - a linear mathematical relationship describing an objective of the firm, in terms of decision variables - this function is to be maximized or minimized
# * Constraints - requirements or restrictions placed on the firm by the operating environment, stated in linear relationships of the decision variables
# * Parameters - numerical coefficients and constants used in the objective function and constraints
# 
# To make things more concrete let's consider and example-

# ## The Factory Problem

# A buisness manager develops a production plan for a factory that produces **chairs** and **tables**. The factory wants to sell a chair for **\$40** and a table for **\$50**.
# There are two critical resources in the production of chairs and tables: wood (measured in **m$^2$**), and labor (measured in working **hours** per week).
# At the beginning of each week there are **40** units of wood and **50** units of labor available.
# 
# The manager estimates that 
# * **one chair** requires **1** unit of wood and **4** units of labor,
# * while **one table** requires **2** units of wood and **3** units of labor. 
# 
# product | profit (\$/unit) | wood (m$^2$/unit) | labor (hour/unit)
# -|-|-|-
# chair | 40 | 1 | 2
# table | 50 | 4 | 3
# 
# The marketing department has told the manager that ALL chairs and tables can be sold.
# 
# **Problem statement:
# what is the production plan that maximizes total revenue?**
# 
# If the factory produces $x_1$ chairs, then since each chair generates \$40, the *total revenue* generated by the production of chairs is $40x_1$.
# Similarly, the *total revenue* generated by the production of tables is $50x_2$. 
# Consequently, the total revenue generated by the production plan can be determined by the following equation:
# 
# \begin{align}
# \text{total revenue per week} = 40x_1 + 50x_2.
# \end{align}
# 
# The production plan is constrained by the amount of resources available per week. How do we ensure that the production plan does not consume more wood or labor than the amount available?
# If we decide to produce $x_1$ number of chairs, then the total amount of wood consumed by the production of chairs is $1x_1$ m$^2$.
# Similarly, if we decide to produce $x_2$ number of tables,
# then the total amount of wood consumed by the
# production of tables is $2x_2$ m$^2$.
# Hence, the total consumption of wood by the
# production plan determined by the values of $x_1$ and $x_2$
# is $1x_1 + 2x_2$  m$^2$. 
# However, the consumption of wood by the production plan cannot exceed the amount of wood available. We can expressed these ideas in
# the following constraint:
# 
# \begin{align}
# 1x_1 + 2x_2 \le 40.
# \end{align}
# 
# Similarly, we can formulate the constraint for
# labor resources per week.
# The total amount of labor resources consumed by the
# production of chairs is 4 labor units multiplied by the
# number of chairs produced, that is $4x_1$ hours.
# The total amount of labor resources consumed by the
# production of tables is 3 labor units multiplied by the
# number of tables produced, that is $3x_2$ hours.
# Therefore, the total consumption of labor resources by
# the production plan determined by the values of $x_1$ and
# $x_2$ is $4x_1 + 3x_2$ hours. This labor consumption cannot
# exceed the labor capacity available. Hence, this
# constraint can be expressed as follows:
# 
# \begin{align}
# 4x_1 + 3x_2 \le 120.
# \end{align}
# 
# In a linear programming (LP) model we have objective (goal) and constraints that must meet.
# Here, the objective is to maximize the total revenue (denoted by $z$) generated by the buisness plan and constraints that limit the number of chairs and tables that can be produced. That is,
# 
# \begin{align}
# &\text{max}\\
# &\qquad z=40𝑥_1 + 50𝑥_2&\quad&\text{(revenue)}\\
# &\text{subject to}\\
# &\qquad 1x_1 + 2𝑥_2 \le 40&\quad&\text{(wood, m$^2$)}\\
# &\qquad 4𝑥_1 + 3𝑥_2 \le 120&\quad&\text{(labor, hours)}\\
# &\qquad x_1, x_2 \ge 0&\quad&\text{(non negativity)}\\
# \end{align} 
# 
# where $𝑥_1$ is a **decision variable** that represent the number of
# chairs to produce, and $𝑥_2$ is a **decision variable** that represent the number of tables to produce. 
# 
# Abstraction of factory problem allows
# us to generalize the model. That is, we are able to change the values of the data without changing the
# model structure:
# 
# \begin{align}
# &\text{max}\\
# &\qquad z=c_1𝑥_1 + c_2𝑥_2&\quad&\text{(revenue)}\\
# &\text{subject to}\\
# &\qquad a_{1,1}x_1 + a_{1,2}𝑥_2 \le b_1&\quad&\text{(wood, $m^2$)}\\
# &\qquad a_{2,1}𝑥_1 + a_{2,2}𝑥_2 \le b_2&\quad&\text{(labor, hours)}\\
# &\qquad x_1, x_2 \ge 0&\quad&\text{(non negativity)}\\
# \end{align} 
# 
# where $c_1=40$, $c_2=50$, $a_{1,1}=1$, $a_{1,2}=2$,
# $a_{2,1}=4$, $a_{2,2}=3$, $b_1=40$, and $b_2=120$. 
# Summation notation allows us to write the above set of equations in more concise form:
# 
# \begin{align}
# &\text{max}\\
# &\qquad z=\sum_{i=1}^{n} c_i x_i  \\
# &\text{subject to} \\
# &\qquad \sum_{j=1}^{m} a_{i,j} x_j \le b_i, \quad(\forall i = 1,2,.. m) \\
# &\qquad x_i \ge 0, \quad(\forall i = 1,2,.. n). \\
# \end{align} 
# 
# where $n$ is the number of decision variables (here $n=2$), and
# $m$ is the number of constraining resources (here $m=2$). In example, 
# 
# \begin{align}
# &\text{max}\\
# &z=\sum_{i=1}^{n} c_i x_i= c_1x_1+c_2x_2=40x_1+50x_2,\\
# &\text{subject to} \\
# &\sum_{j=1}^{m} a_{i,j} x_j = \left\{
#         \begin{array}{cl}
#         a_{1,1}x_1+a_{1,2}x_2&\le b_1&\Rightarrow 1x_1+2x_2&=40,  \\
#         a_{2,1}x_1+a_{2,2}x_2&\le b_2&\Rightarrow 4x_1+3x_2&=120,
#         \end{array}
#         \right.\\
# &x_1\ge 0, x_2\ge 0.\\
# \end{align} 
# 
# 
# Another way of writing this is using a matrix form:
# 
# \begin{align}
# &\text{max}\\
# &\qquad z=\mathbf{c}^T\mathbf{x}  \\
# &\text{subject to}\\
# &\qquad\mathbf{A}\mathbf{x} \le \mathbf{b},\\ 
# &\qquad\mathbf{x} \ge 0.\\
# \end{align} 
# 
# Here, $\mathbf{c}=(c_1,c_2)=(40,50),~\mathbf{x}=(x_1,x_2), \mathbf{b}=(b_1,b_2)=(40,120)$, and 
# $\mathbf{A} = 
#   \begin{pmatrix}
#   a_{1,1} & a_{1,2} \\
#   a_{2,1} & a_{2,2} 
#   \end{pmatrix}
#    = 
#   \begin{pmatrix}
#   1 & 2 \\
#   4 & 3 
#   \end{pmatrix}.
# $
# 
# Lastely, in a one line format:
# 
# \begin{align} 
# {\displaystyle \max\{\,\mathbf {c} ^{\mathrm {T} }\mathbf {x} \mid \mathbf {x} \in \mathbb {R} ^{n}\land A\mathbf {x} \leq \mathbf {b} \land \mathbf {x} \geq 0\,\}}.
# \end{align} 
# 
# In operation research, objective functions are *maximized* or *minimized*. For example, maximizing revenue or minimizing cost.
# 
# Constraints typically involve inequalities:
# * $\le$ constraints are typically considered for
# **capacity/supply** constraints that can't exceed some threshold.
# * $\ge$ constraints are typically used to model **demand**
# requirements where we want to ensure that
# at least certain level is satisfied.
# * $=$ constraints are used when we want to
# match exactly certain activities with a given
# requirement. 
# 
# A solution can be feasible or infeasible. A feasible solution does not violate any of the constraints, while an infeasible solution violates **at least one** of the constraints. In example, for the factury example, $\mathbf{x}=(5,10)$ is feasible, while $\mathbf{x}=(10,20)$ is not.
# 

# ## Graphical Solution of Linear Programming Models

# Graphical solution is limited to LP models containing only two decision variables.
# Graphical **methods** provide an illustration of how a solution for a linear programming problem is obtained.
# 
# Panel a) of the figure below shows the region of the wood constraint $1x_1+2x_2 <= 40$ using the fact that the decisions variables are non-negative. This is the feasible area that contains the possible solutions for the decisions variables. Panel b) shows the region of the labor constraint $4x_1+3x_2 <= 120$, again, using the fact that the decisions variables are non-negative. 
# 
# The goal is to maximize the objective $40x_1+50x_2$. In panel c) we show multiple lines for the various values that the objective *can* take.
# 
# The two contraints must meet at the same time, and so on panel d) we intersect both contraints and the resulting common area is the new feasible area that is considered in this problem. Panel e) shows multiplte objective values with respect to the new feasible region. In linear programming the
# feasible region is called a **polyhedron**. Panel f) is the outcome of the graphical method by which we identify the optimal solution $(x_1,x_2)=(24,8)$ to the problem. At this point the objective maximized with $z=\$1360$ and satisfy all the contraints of the question.
# 
# Hence, the optimal solution point is the last point the objective function touches as it leaves the feasible solution area.

# In[226]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.figure(figsize=(16,8))
#-------------------
plt.subplot(2,3,1)
plt.plot([40,0],[0,20],'-b')
plt.text(35,5, '$1x_1+2x_2 <= 40$', color='b')
P = np.array([
        [0, 40, 0],
        [0, 0, 20]
    ])
plt.gca().add_artist(Polygon(P.T, alpha=0.3, color="b"))
plt.grid()
plt.axis([0, 60, 0, 60])
plt.xlabel('$x_1$');plt.ylabel('$x_2$')
plt.title('a) feasiblewood contraint: $1x_1+2x_2 <= 40$')
#-------------------
plt.subplot(2,3,2)
plt.plot([30,0],[0,40],'-r')
plt.text(5,35, '$4x_1+3x_2 <= 120$', color='r')
P = np.array([
        [0, 30, 0],
        [0, 0, 40]
    ])
plt.gca().add_artist(Polygon(P.T, alpha=0.3, color="r"))
plt.grid()
plt.axis([0, 60, 0, 60])
plt.xlabel('$x_1$');plt.ylabel('$x_2$')
plt.title('b) feasible labor contraint: $4x_1+3x_2 <= 120$')
#-------------------
plt.subplot(2,3,3)
plt.plot([800/40,0],[0,800/50],'--g')
plt.text(20,5, '$40x_1+50x_2=800$', color='g')

plt.plot([2000/40,0],[0,2000/50],'--k')
plt.text(40,15, '$40x_1+50x_2=2000$', color='k')
plt.grid()
plt.axis([0, 60, 0, 60])
plt.xlabel('$x_1$');plt.ylabel('$x_2$')
plt.title('c) multiple objective $40x_1+50x_2$ lines')
#-------------------
plt.subplot(2,3,4)
#plt.plot([40,0],[0,20],'-b')
#plt.text(35,5, '$1x_1+2x_2 <= 40$', color='b')
P = np.array([
        [0, 40, 0],
        [0, 0, 20]
    ])
plt.gca().add_artist(Polygon(P.T, alpha=0.3, color="b"))
P = np.array([
        [0, 30, 0],
        [0, 0, 40]
    ])
plt.gca().add_artist(Polygon(P.T, alpha=0.3, color="r"))
plt.grid()
plt.axis([0, 60, 0, 60])
plt.xlabel('$x_1$');plt.ylabel('$x_2$')
plt.title('d) intersecting the two feasible constraint regions')
#-------------------
plt.subplot(2,3,5)
#plt.plot([40,0],[0,20],'-b')
#plt.text(35,5, '$1x_1+2x_2 <= 40$', color='b')
P = np.array([
        [0, 30, 24,0],
        [0, 0, 8 ,20]
    ])
plt.gca().add_artist(Polygon(P.T, alpha=0.3, color="b"))
P = np.array([
        [0, 30, 24,0],
        [0, 0, 8 ,20]
    ])
plt.gca().add_artist(Polygon(P.T, alpha=0.3, color="r"))
plt.plot([600/40,0],[0,600/50],'--k')
plt.plot([1360/40,0],[0,1360/50],'--k')
plt.plot([2000/40,0],[0,2000/50],'--k')
plt.grid()
plt.axis([0, 60, 0, 60])
plt.xlabel('$x_1$');plt.ylabel('$x_2$')
plt.title('e) the feasible area and multiple objective lines')

#-------------------
plt.subplot(2,3,6)
#plt.plot([40,0],[0,20],'-b')
#plt.text(35,5, '$1x_1+2x_2 <= 40$', color='b')
P = np.array([
        [0, 30, 24,0],
        [0, 0, 8 ,20]
    ])
plt.gca().add_artist(Polygon(P.T, alpha=0.3, color="b"))
P = np.array([
        [0, 30, 24,0],
        [0, 0, 8 ,20]
    ])
plt.gca().add_artist(Polygon(P.T, alpha=0.3, color="r"))
plt.plot([1360/40,0],[0,1360/50],'--k')
plt.text(25,15, '$z=40x_1+50x_2=1360$', color='k')
plt.scatter(24,8,s=50,c='k')
plt.grid()
plt.axis([0, 60, 0, 60])
plt.xlabel('$x_1$');plt.ylabel('$x_2$')
plt.title('f) the feasible area and optimal solution')

plt.tight_layout()
plt.show()


# The **standard form** of LP requires that all constraints be in the form of equations (equalities).
# For that a **slack** variable is added to a $\le$ constraint (weak inequality) to convert it to an equation ($=$).
# A slack variable typically represents an *unused* resource.
# A slack variable contributes nothing to the objective function value.
# 
# In example, the standard form of the above factory LP problem is:
# 
# \begin{align}
# &\text{max}\\
# &\qquad z=40𝑥_1 + 50𝑥_2+0s_1+0s_2\\
# &\text{s.t.}\\
# &\qquad 1x_1 + 2𝑥_2+s_1 = 40\\
# &\qquad 4𝑥_1 + 3𝑥_2+s_2 = 50\\
# &\qquad x_1, x_2, s_1, s_2 \ge 0\\
# \end{align} 
# 
# where $s_1$ and $s_2$ are the slack variables.

# **Minimization LP** problems work similarly with contraints equations bearing the opposite inequlity sign. In example,
# 
# \begin{align}
# &\text{min}\\
# &\qquad z=6𝑥_1 + 3𝑥_2\\
# &\text{s.t.}\\
# &\qquad 2x_1 + 4𝑥_2 \ge 18\\
# &\qquad 4𝑥_1 + 3𝑥_2 \ge 24\\
# &\qquad x_1, x_2\ge 0\\
# \end{align} 
# 
# The solution can be found using the graphical method resulting in $(x_1,x2)=(4.8,1.6)$ and $z=33.6$.

# In[227]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

plt.figure(figsize=(16,3))
#-------------------
plt.subplot(1,2,1)
P = np.array([
        [8, 12, 12, 0,  0, 4.8],
        [0,  0, 12, 12, 8, 1.6]
    ])
plt.gca().add_artist(Polygon(P.T, alpha=0.3, color="r"))
plt.grid()
plt.axis([0, 12, 0, 12])
plt.xlabel('$x_1$');plt.ylabel('$x_2$')
plt.title('a) feasible area for the minimiztion problem')
#-------------------
plt.subplot(1,2,2)
P = np.array([
        [8, 12, 12, 0,  0, 4.8],
        [0,  0, 12, 12, 8, 1.6]
    ])
plt.gca().add_artist(Polygon(P.T, alpha=0.3, color="r"))
plt.plot([33.6/6,0],[0,33.6/3],'--k')
plt.scatter(4.8,1.6,s=50,c='k')
plt.text(1,2, '$z=6x_1+3x_2=33.6$', color='k')

plt.grid()
plt.axis([0, 12, 0, 12])
plt.xlabel('$x_1$');plt.ylabel('$x_2$')
plt.title('b) feasible area for the minimiztion problem + optimal solution')
plt.tight_layout()
plt.show()


# **Surplus** variables correspond to slack variables but represent an excess above a constraint requirement level.
# As with slack, surplus variables contribute nothing to the calculated value of the objective function. For the minimization problem, adding surplus variables results in,
# 
# \begin{align}
# &\text{min}\\
# &\qquad z=6𝑥_1 + 3𝑥_2+0s_1+0s_2\\
# &\text{s.t.}\\
# &\qquad 2x_1 + 4𝑥_2 -s_1 = 18\\
# &\qquad 4𝑥_1 + 3𝑥_2 -s1 =  24\\
# &\qquad x_1, x_2, s_1, s_2\ge 0\\
# \end{align} 
# 

# There are **irregular** types of LP problems for which the general rules do not apply. Special types of problems include those with:
# * **Multiple optimal solutions**, i.e., where the objective function is parallel to a constraint line.
# * **Infeasible solutions**, i.e., where every possible solution violates at least one constraint.
# * **Unbounded solutions**, i.e., where the value of the objective
# function increases indefinitely.
# 
# 

# # Chapter 2: Solving problems using Python

# In[228]:


get_ipython().system('pip install gurobipy')

# Import gurobi library
from gurobipy import * # This command imports the Gurobi functions and classes.

# Create new model
m = Model('Factory') # The Model() constructor creates a model object m. The name of this new model is 'Factory'.
                     # This new model m initially contains no decision variables, constraints, or objective function.

# Create decision variables
# This method adds a decision variable to the model object m, one by one; i.e. x1 and then x2. 
# The argument of the method gives the name of added decision variable. 
# The default values are applied here; i.e. the decision variables are of type continuous and non-negative, with no upper bound.
x1 = m.addVar(lb=0, vtype = GRB.CONTINUOUS, name='chairs') 
x2 = m.addVar(lb=0, vtype = GRB.CONTINUOUS, name='tables')

#Define objective function
#This method adds the objective function to the model object m. The first
#argument is a linear expression (LinExpr) and the second argument defines
#the sense of the optimization.
m.setObjective(40*x1+50*x2, GRB.MAXIMIZE)

#Add constraints
#This method adds a constraint to the model object m and considers a linear of coefficient-variables elements
m.addConstr(1*x1+2*x2<=40, name='wood')
m.addConstr(4*x1+3*x2<=120, name= 'labor')

#Run optimization engine
#This method runs the optimization engine to solve the LP problem in the model object m
m.optimize()

#display optimal production plan
for v in m.getVars():
  print(v.varName, v.x)
print('optimal total revenue:', m.objVal)


# **What if our LP problem has hundreds of thousands varibales and constraints?** 
# 
# The Gurobi python code just presented is too manual and would take too long too build a large scale LP problem. 
# We should use appropriate data structures and Gurobi python functions and objects to abstract the problem and have the Gurobi python code build the LP problem of any size.

# In[ ]:


#Python list comprehension
#List comprehension is compact way to create lists
sqrd = [i*i for i in range(5)]
print(sqrd) 

#Can be used to create subsequences that satisfy certain conditions (ex: filtering a list)
bigsqrd = [i*i for i in range(5) if i*i >= 5]
print(bigsqrd) 

#Can be used with multiple for loops (ex: all combinations)
prod = [i*j for i in range(3) for j in range(4)]
print(prod) 

#Generator expression is similar, but no brackets (ex: argument to aggregate sum)
sumsqrd = sum(i*i for i in range(5))
print(sumsqrd)


# In[ ]:


from gurobipy import * 

#resource data
#The multidict function returns a list which maps each resource (key) to its capacity value.
resources, capacity = multidict({ 
    'wood':  40,
    'labor': 120 })
print(resources, capacity)

#products data
#This multidict function returns a list which maps each product (key) to its price value.
products, price = multidict({
    'chair': 40,
    'table': 50 })
print(products, price)

#bill of materials: resources required by each product
#This dictionary has a 2-tuple as a key, mapping the resource required by a product with its quantity per.
bom={
('wood','chair'):1,
('wood','table'):2,
('labor','chair'):4,
('labor','table'):3
}
print(bom)

m = Model('Factory')

#This method adds decision variables to the model object m
make = m.addVars(products, name='make')

#This method adds constraints to the model object m
res = m.addConstrs(((sum(bom[r,p]*make[p] for p in products) <= capacity[r]) for r in resources),name='R')

#This method adds the objective function to the model object m.
#The first argument is a linear expression which is generated by the 'prod' method. 
#The 'prod' method is the product of the object (revenue) with the object (make) 
#for each product p in the set (products). The second argument defines the sense of the optimization.
m.setObjective(make.prod(price), GRB.MAXIMIZE)

#save model for inspection
m.write('factory.lp')


# In[ ]:


cat factory.lp


# In[ ]:


# run optimization engine
m.optimize()

#display optimal production plan
for v in m.getVars():
  print(v.varName, v.x)
print('optimal total revenue:', m.objVal)


# ## Sensitivity analysis of LP problems

# Solving LP problems provides more information than only the values of the decision variables and the value of the objective function.

# In[ ]:


for v in m.getVars():
  print(v.varName, v.x)
print('optimal total revenue:', m.objVal)

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, (var.obj,var.SAObjLow, var.SAObjUp, var.RC))

for con in m.getConstrs(): # constraints
    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal)


# ### Objective value coefficients
# * Optimally, we produce 24 chairs ($x1$ has the optimal value of 24). We sell a chair for $40 (its objective coefficient is 40). While holding the other objective coefficients fix, the values of 40 can change within the range of (25.0, 66.67) without affecting the optimal solution of(24,8,0). However, changing the objective coefficient will change the objective value!
# * Similarly, for tables, the objective coefficient is 50 but can vary (42.5, 80.0) without affecting the optimal solution point.
# * Similarily, or benches. Although here, the value is just 0.

# ### Constraint quantity values
# The constraint quantity values are 40 m$^2$ and 120 hours. Modifying these values will change the feasible area. Here, we are looking for the range of values over which the quantity values can change **without changing the solution variable mix** including slack. 
# * For the wood constraint, the RHS value is 40 m$^2$ and it can change within the range of (30, 80) without changing the solution variable mix.
# * For the labor constraint, the RHS value is 120 hours, and it can change within the range of (60, 160) without changing the solution variable mix.

# ### Shadow prices
# Associated with an LP optimal solution there are **shadow prices** (also known as: **dual values**, or **marginal values**) for the constraints.
# The **shadow price** of a constraint associated with the optimal solution represents the change in the value of the objective function per unit of increase in the RHS value of that constraint.
# 
# * Suppose the wood capacity is increased from 40 m$^2$ to 41 m$^2$, then the objective function value will increase from the optimal value of 1360 to 1360+**16**. The shadow price of the wood constraint is 16.
# * Similarly, suppose the labor capacity is increased from 120 hours to 121 hours, then the objective function value will increase from the optimal value of 1360 to 1360+**6**. The shadow price of the labor constraint is 6. 

# 
# ### Sensitivity of the shadow price
# The sensitivity range for a constraint quantity value is also the range over which the shadow price is valid (i.e., before a slack/surplus value is added to the mix). For example, the shadow price of 16 hours is valid over the range of (60, 160) hours for labor.

# ### Adding new variable and/or new constraint
# Is it profitable to make a third product, like benches?
# Assume that the price of a bench is $30, and a bench consumes 1.2 units of wood and 2 units of labor, then we can formulate the LP model using the previous resources constraints on wood and labor as follows:
# 
# What about adding a new constraint as packaging, and varying costs for chairs and tables?

# In[ ]:


# Adding new variable

products, price = multidict({
    'chair': 40,
    'table': 50,
    'bench' : 30})
bom={
('wood','chair'):1,
('wood','table'):2,
('wood','bench'):1.2,
('labor','chair'):4,
('labor','table'):3,
('labor','bench'):2,
}

m = Model('Factory')

make = m.addVars(products,name='make')
res = m.addConstrs(((sum(bom[r,p]*make[p] for p in products) <= capacity[r]) for r in resources),name='R')
m.setObjective(make.prod(price), GRB.MAXIMIZE)
m.write('factory.lp')


# In[ ]:


cat factory.lp


# In[ ]:


m.optimize()

#display optimal production plan
for v in m.getVars():
  print(v.varName, v.x)
print('optimal total revenue:', m.objVal)

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, (var.obj,var.SAObjLow, var.SAObjUp, var.RC))

for con in m.getConstrs(): # constraints
    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal)


# In[ ]:


# Adding new constraint

resources, capacity = multidict({ 
    'wood':  40,
    'labor': 120,
    'packaging': 5 })


products, price = multidict({
    'chair': 40,
    'table': 50})
bom={
('wood','chair'):1,
('wood','table'):2,
('labor','chair'):4,
('labor','table'):3,
('packaging','chair'):0.2,
('packaging','table'):0.1,
}

m = Model('Factory')

make = m.addVars(products,name='make')
res = m.addConstrs(((sum(bom[r,p]*make[p] for p in products) <= capacity[r]) for r in resources),name='R')
m.setObjective(make.prod(price), GRB.MAXIMIZE)
m.write('factory.lp')


# In[ ]:


cat factory.lp


# In[ ]:


m.optimize()

#display optimal production plan
for v in m.getVars():
  print(v.varName, v.x)
print('optimal total revenue:', m.objVal)

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, (var.obj,var.SAObjLow, var.SAObjUp, var.RC))

for con in m.getConstrs(): # constraints
    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal)


# ## Problem example: C2Q1
# Solve the following LP problem:
# 
# \begin{align}
# &\text{max}\\
# &\qquad z=10x_1+6x_2\\
# &\text{s.t.}\\
# &\qquad 3x_1+8x_2\le 20\\
# &\qquad 45x_1+30x_2\le 180\\
# &\qquad x_1, x_2\ge 0\\
# \end{align} 
# 

# In[ ]:


c = [10, 6]    
A = [[3,   8 ],
     [45, 30 ]]
b = [20, 180]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("C2Q1")

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


# ## Problem example: C2Q2
# Solve the following LP problem:
# 
# \begin{align}
# &\text{min}\\
# &\qquad z=0.5x_1+0.03x_2\\
# &\text{s.t.}\\
# &\qquad 8x_1+6x_2\ge 48\\
# &\qquad x_1+2x_2\ge 12\\
# &\qquad x_1, x_2\ge 0\\
# \end{align} 

# In[ ]:


c = [0.5, 0.03]    
A = [[8,6 ],
     [1,2 ]]
b = [48,12]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("C2Q1")

x = []
for i in decision_variables:
    x.append(m.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = 'x' + str(i)))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MINIMIZE) 

m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           >= b[j] for j in constraints), "constraints")

m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, (var.obj,var.SAObjLow, var.SAObjUp, var.RC))

for con in m.getConstrs(): # constraints
    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal)


# ## Problem example: C4Q8
# Solve the following LP problem:
# 
# \begin{align}
# &\text{min}\\
# &\qquad z=4x_1+3x_2+2x_3\\
# &\text{s.t.}\\
# &\qquad 2x_1+4x_2+x_3\ge 16\\
# &\qquad 3x_1+2x_2+x_3\ge 12\\
# &\qquad x_1, x_2, x_3\ge 0\\
# \end{align} 

# In[ ]:


c = [4,3,2]    
A = [[2,4,1 ],
     [3,2,1 ]]
b = [16,12]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("C4Q8")

x = []
for i in decision_variables:
    x.append(m.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = 'x' + str(i)))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MINIMIZE) 

m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           >= b[j] for j in constraints), "constraints")

m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, (var.obj,var.SAObjLow, var.SAObjUp, var.RC))

for con in m.getConstrs(): # constraints
    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal)


# ## Problem example: C4Q32
# Solve the following LP problem:
# 
# \begin{align}
# &\text{min}\\
# &\qquad z=22x_1+18x_2+35x_3+41x_4+30x_5+28x_6+25x_7+36x_8+18x_9\\
# &\text{s.t.}\\
# &\qquad x_1+x_2+x_3=1\\
# &\qquad x_4+x_5+x_6=1\\
# &\qquad x_7+x_8+x_9=1\\
# &\qquad x_1+x_4+x_7=1\\
# &\qquad x_2+x_5+x_8=1\\
# &\qquad x_3+x_6+x_9=1\\
# &\qquad x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9\ge 0\\
# \end{align} 

# In[ ]:


c = [22,18,35,41,30,28,25,36,18]    
A = [[1,1,1,0,0,0,0,0,0 ],
     [0,0,0,1,1,1,0,0,0 ],
     [0,0,0,0,0,0,1,1,1 ],
     [1,0,0,1,0,0,1,0,0 ],
     [0,1,0,0,1,0,0,1,0 ],
     [0,0,1,0,0,1,0,0,1 ]]
b = [1,1,1,1,1,1]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("C4Q32")

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


# This would require the model to be reformulated with three new variables, $x_{10}, x_{11}, x_{12}$, representing Kelly's assignment to the press, lathe, and grinder. The model would be reformulated as,
# 
# \begin{align}
# &\text{min}\\
# &\qquad z=22x_1+18x_2+35x_3+41x_4+30x_5+28x_6+25x_7+36x_8+18x_9+20x_{10}+20x_{11}+20x_{12}\\
# &\text{s.t.}\\
# &\qquad x_1+x_2+x_3=1\\
# &\qquad x_4+x_5+x_6=1\\
# &\qquad x_7+x_8+x_9=1\\
# &\qquad x_1+x_4+x_7=1\\
# &\qquad x_2+x_5+x_8=1\\
# &\qquad x_3+x_6+x_9=1\\
# &\qquad x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9\ge 0
# \end{align} 

# In[ ]:


c = [22,18,35,41,30,28,25,36,18,20,20,20]    
A = [[1,1,1,0,0,0,0,0,0,0,0,0 ],
     [0,0,0,1,1,1,0,0,0,0,0,0 ],
     [0,0,0,0,0,0,1,1,1,0,0,0 ],
     [0,0,0,0,0,0,0,0,0,1,1,1 ],
     [1,0,0,1,0,0,1,0,0,1,0,0 ],
     [0,1,0,0,1,0,0,1,0,0,1,0 ],
     [0,0,1,0,0,1,0,0,1,0,0,1 ]]
b =  [1,1,1,1,1,1,1]
print(np.array(c).shape,np.array(A).shape,np.array(b).shape)
decision_variables = range(len(c))     
constraints = range(np.array(A).shape[0])

m = Model("C4Q32")

x = []
for i in decision_variables:
    x.append(m.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = 'x' + str(i)))

m.setObjective(quicksum(c[i] * x[i] for i in decision_variables) , GRB.MINIMIZE) 

m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           <= b[j] for j in range(4)), "constraints")

m.addConstrs((quicksum(A[j][i] * x[i] for i in decision_variables) 
                           == b[j] for j in range(4,7)), "constraints")
m.optimize()

for var in m.getVars(): # descision variable
    print(var.varName, '=', var.x, (var.obj,var.SAObjLow, var.SAObjUp, var.RC))

for con in m.getConstrs(): # constraints
    print(con.ConstrName, ': slack =', con.slack,', shadow price=',
          con.pi,',', (con.RHS, con.SARHSLow, con.SARHSUp))
    
print("objective value =", m.objVal)


# ## Problem example: C4Q33
# 
# Solve the following LP problem:
# 
# This is a transportation problem
# 
# \begin{align}
# &\text{min}\\
# &\qquad z=40x_1 + 65x_2 + 70x_3 + 30x_4\\
# &\text{s.t.}\\
# &\qquad x_1+x_2=250\\
# &\qquad x_3+x_4=400\\
# &\qquad x_1+x_3=300\\
# &\qquad x_2+x_4=350\\
# &\qquad x_1,x_2,x_3,x_4\ge 0\\
# \end{align} 

# In[ ]:


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


# # Chapter 3: Theory of LP and the Simplex method
# 

# To be done later on
# 
# To place here a python code that follows the Simplex algorithm

# In[ ]:


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


# # Chapter 4: Integer Programming
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

# In[ ]:


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

# In[ ]:


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

# In[ ]:


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

# In[ ]:


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


# In[ ]:


y=[]
for var in m.getVars():
    y.append(np.floor(var.x))

np.array(y).dot(np.array((50,40)))


# In[ ]:


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

# In[ ]:


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

# In[ ]:


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

# In[ ]:


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

# In[ ]:


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


# In[ ]:


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

# In[ ]:


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

# In[ ]:


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


# In[ ]:


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

# In[ ]:


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


# ##Problem Example: C6Q43
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

# In[ ]:


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


# # Chapter 5: Multicriteria Decision Making

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

# In[ ]:


from gurobipy import *
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

# In[ ]:


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

# In[ ]:


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

# In[ ]:


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


# # Chapter 6: Nonlinear Programming

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

# In[ ]:


# import gurobi library
from gurobipy import * # This command imports the Gurobi functions and classes.

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

# In[ ]:


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

# In[ ]:


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

# In[ ]:


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


# In[ ]:


import numpy as np
x=1658.8;y=1416.7;
d=7000*np.sqrt((x-1000)**2+(y-1250)**2)+9000*np.sqrt((x-1500)**2+(y-2700)**2)+11500*np.sqrt((x-2000)**2+(y-700)**2)+4300*np.sqrt((x-2200)**2+(y-2000)**2)
print(x,y,d)

x=1665.4;y=1562.9;
d=7000*np.sqrt((x-1000)**2+(y-1250)**2)+9000*np.sqrt((x-1500)**2+(y-2700)**2)+11500*np.sqrt((x-2000)**2+(y-700)**2)+4300*np.sqrt((x-2200)**2+(y-2000)**2)
print(x,y,d)


# # Chapter 7: Probability and Statistics

# ##Basic Probability Theory
# 
# **Deterministic** techniques assume that no uncertainty exists in model parameters. Previous topics assumed no uncertainty or variation to the specified paramaters.
# **Probabilistic** techniques on the other side include uncertainty and assume that there can be more than one model solution or variation in the parameter values. Also, there may be some doubt about which outcome will occur.
# 

# ##Bayesian analysis

# ##Probability Distributions

# ##Chi-Square Test 

# 

# ##Problem Example: C11Q5

# In[ ]:


from scipy.stats import binom
binom.cdf(k=4, n=20, p=.1), 1-binom.cdf(k=4, n=20, p=.1)


# ## Problem Example: C11Q8

# In[ ]:


1-binom.cdf(k=2, n=7, p=.2)


# In[ ]:




