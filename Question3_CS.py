#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.stats               


# In[5]:


L = 7.2
n = 100000
x = []
for i in range(n):                                        ### create samples from gamma(5, 7.2)
    u = np.random.uniform(0,1,1000)                       ### create samples from U[0,1]
    s = sum(-1*np.log(1-u[i])/L for i in range(5))        ### inverse transform method
    x.append(s)


# In[7]:


final_x = []
i =0
while(len(final_x)!=100):
    if np.random.uniform(0,1,1) < x[i]**0.4:
        final_x.append(x[i])
        i +=1


# In[8]:


# using chi-square goodness of fit test to check assumption
observed_values=scipy.array(final_x)
expected_values=scipy.array(np.random.gamma(5.4, 7.2, 100))
scipy.stats.chisquare(observed_values, f_exp=expected_values)


# In[ ]:


#as here we found that p-value is very less means null hypothesis is rejected

