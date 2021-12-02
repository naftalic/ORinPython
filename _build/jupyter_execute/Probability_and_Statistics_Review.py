#!/usr/bin/env python
# coding: utf-8

# # Probability and Statistics Review
# 
# Probability and statistics and constitute of dealing with uncertainty in desicion making. The theory of probability provides the basis for the learning from data.

# In[1]:


import numpy as np
from scipy import stats
import scipy


# ## Fundamentals
# Starting with a few definitions. A **sample space** is the set of all the possible experimental outcomes. And, **experiment**, is defined to be any process for which more than one **outcome** is possible.
# 
# Imagine a sample space of $n$ possible outcomes $S = \{ O_1, O_2, \cdots, O_n \}$. We assign a probability $p_i$ to each outcome $O_i$
# $
# \begin{equation}
#     P(O_i) = p_i.
# \end{equation}
# $
# All $p_i$ must satisfy 
# $
# \begin{equation}
#    0 \leq p_i \leq 1, \hspace{0.5cm} \forall i= 1,2, \cdots, n.
# \end{equation}
# $
# If all $p_i$ are equal to a constant $p$ we say that all $n$ outcomes are equally likely, and each probability has a value of $1/n$.
# 
# An **event** is defined to be a subset of the sample space. The probability of an event $A$, $P(A)$, is obtained by summing the probabilities of the outcomes contained withing the event $A$. An event is said to occur if one of the outcomes contained within the event occurs. The complement of event $A$ is the event $A^C$ and is defined to be the event consisting of everything in the sample space $S$ that is not contained within $A$. In example,
# $
# \begin{equation}
#     P(A) + P(A^C) = 1
# \end{equation}
# $
# **Intersections** of events, $A \cap B$, consists of the outcomes contained within both events $A$ and $B$. The probability of the intersection, $P(A \cap B)$, is the probability that both events occur simultaneously.
# A few known properties of the intersections of events:
# $
# \begin{align}
#  & P(A \cap B) +P(A \cap B^C) = P(A)\\
# &A \cap (B \cap C) = (A \cap B) \cap C\\
# & A \cap B = \emptyset \quad\text{(for mutually exclusive events)}\\
# \end{align}
# $
# The **union** of events, $ A\cup B$, consists of the outcomes that are contained within at least one of the events $A$ and $B$. The probability of this event, $P (A \cup B)$ is the probability that at least one of these events $A$ and $B$ occurs.
# A few known properties of the union of events:
# $
# \begin{align}
# & A \cup A^C = S \\
# & (A \cup B)^C = A^C \cap B^C\\
# & (A \cap B)^C = A^C \cup B^C\\
# & A \cup (B \cup C) = (A \cup B) \cup C \\
# & P(A \cup B) = P(A) + P(B) = \emptyset \quad\text{(for mutually exclusive events)}\\
# & P( A \cup B) = P(A \cap B^C) + P(A^C \cap B^C) + P(A \cap B)\\
# \end{align}
# $
# The union of three events is equal to
# $ 
# \begin{align}
# P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P( B \cap C) - P( A \cap C) + P(A \cap B \cap C).
# \end{align}
# $
# If the events union is **mutually exclusive** then
# $
# \begin{align}
#     P(A_1 \cup A_2 \cup \cdots \cup A_n) = P(A_1) + \cdots + P(A_n),
# \end{align}
# $
# where the sequence $A_1, A_2, \cdots , A_n$ are called the **partition** of $S$.
# 
# **Conditional Probability** is defined as an event $B$ that is conditioned on another event $A$. In this case,
# $
# \begin{align}
#     P(B \mid A) = \frac{P(A \cap B)}{P(A)} \hspace{0.5cm}  \text{for } P(A) >0.
# \end{align}
# $
# From the above equation, it follows that 
# $
# \begin{align}
# P(A \cap B) = P (B \mid A) P(A).
# \end{align}
# $
# It's not hard to see that conditioning on more evets (e.g. two) results in
# $
# \begin{align}
# P(A \cap B\cap C) = P (C \mid B\cap A) P(B\cap A).
# \end{align}
# $
# In general, for a sequence of events $A_1, A_2, \cdots, A_n$:
# $
# \begin{align}
# \mathrm {P} (A_{n}\cap \ldots \cap A_{1})=\mathrm {P} (A_{n}|A_{n-1}\cap \ldots \cap A_{1})\cdot \mathrm {P} (A_{n-1}\cap \ldots \cap A_{1}).
# \end{align}
# $
# If the two events $A$ and $B$ are independent, knowledge about one event does not affect the probability of the other event. The following conditions are equivalent:
# $
# \begin{align}
# P(A \mid B) &= P(A)\\
# P(A \cap B) &= P(A)P(B).\\
# \end{align}
# $
# In general, if $A_1, A_2, \cdots, A_n$ are independent then
# $
# \begin{align}
# P(A_1 \cap A_2  \ldots \cap A_n) = P(A_1)P(A_2) \cdots P(A_n).
# \end{align}
# $

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

# In[1]:





# 
