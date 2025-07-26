#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np

class Bandit:
    def __init__(self, q, Q):
        self.q = q
        self.Q = Q
        self.n = 0

    def __str__(self):
        return f"q:{self.q} | Q:{self.Q} | n:{self.n}"

    def pull(self):
        return np.random.normal(self.q, 1)

    def update(self, r):
        self.n += 1
        self.Q = self.Q + (1/self.n) * (r - self.Q)
        
def simple_bandit(q):
    return Bandit(q=q,Q=0)

