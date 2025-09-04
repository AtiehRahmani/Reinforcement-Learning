


import numpy as np

class GBandit:
    def __init__(self, q, H):
        self.q = q
        self.H = H
        self.n = 0 #We do not need the variable n in GBandit, but we include it here for consistency with the bandit class.

    def __str__(self):
        return f"q:{self.q} | H:{self.H} | n:{self.n}"

    def pull(self):
        return np.random.normal(self.q, 1)#variance==1 and mean==q  ==> as same as in class Bandit 

    def update(self, new_H):
        self.n += 1
        self.H=new_H

def simple_gbandit(q):
    return GBandit(q=q,H=0)
        