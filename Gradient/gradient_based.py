
import numpy as np
from gbandit import simple_gbandit

def softmax(values):
    exp=np.exp(values)
    sum_exp=np.sum(exp)
    pi=exp/sum_exp
    return pi
def multi_armed_bandit_gradient(num_bandits, num_steps,alpha):
    bandits=[simple_gbandit(np.random.normal(0,1)) for i in range (num_bandits)]
    reward_history=[]    
    R_mean=0 #Rbar
    for i in range(num_steps):
        H_values=[b.H for b in bandits]
        pi=softmax(H_values)
        a=np.random.choice(num_bandits,p=pi)#we use pi as probability distribution
        r=bandits[a].pull()
        reward_history.append(r)

        #update H
        for i, b in enumerate(bandits):
            if i==a: #if i==selected action
                new_H=b.H+alpha*(r-R_mean)*(1-pi[i])
            else:
                new_H=b.H-alpha*(r-R_mean)*pi[i]

            b.update(new_H)
        reward_history.append(r)
        R_mean=np.mean(reward_history)
    return reward_history 
    
#H=[10,10] for testing softmax()
#print(softmax(H))

