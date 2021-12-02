#!/usr/bin/env python
# coding: utf-8

# # Probability and Statistics

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

# In[1]:


from scipy.stats import binom
binom.cdf(k=4, n=20, p=.1), 1-binom.cdf(k=4, n=20, p=.1)


# ## Problem Example: C11Q8

# In[2]:


1-binom.cdf(k=2, n=7, p=.2)


# In[2]:




