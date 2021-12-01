#!/usr/bin/env python
# coding: utf-8

# # Python Tutorial
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

