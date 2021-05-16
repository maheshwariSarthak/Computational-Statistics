#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Comp Stat - Assignment 2 
# # Solution 1

# In[5]:


import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt  


# In[6]:


df = pd.read_csv("data_treatmet.csv")
df[:5]


# In[ ]:


y1 = df['y1'].to_numpy()
y2 = df['y2'].to_numpy()
y3 = df['y3'].to_numpy()
y4 = df['y4'].to_numpy()
y5 = df['y5'].to_numpy()

y1 = y1[~np.isnan(y1)]
y2 = y2[~np.isnan(y2)]
y3 = y3[~np.isnan(y3)]
y4 = y4[~np.isnan(y4)]
y5 = y5[~np.isnan(y5)]

y1_sum = np.sum((y1-np.mean(y1))**2)
y2_sum = np.sum((y2-np.mean(y2))**2)
y3_sum = np.sum((y3-np.mean(y3))**2)
y4_sum = np.sum((y4-np.mean(y4))**2)
y5_sum = np.sum((y5-np.mean(y5))**2)

sigma1_hat = (y1_sum+y2_sum)/(y1.shape[0]+y2.shape[0]-2)
sigma2_hat = (y2_sum+y3_sum)/(y2.shape[0]+y3.shape[0]-2)
sigma3_hat = (y3_sum+y4_sum)/(y3.shape[0]+y4.shape[0]-2)
sigma4_hat = (y4_sum+y5_sum)/(y4.shape[0]+y5.shape[0]-2)


# In[8]:


q = 1-(.05/2)
CI1_L = np.mean(y1)-np.mean(y2) - scipy.stats.t.ppf(q=q,df=y1.shape[0]+y2.shape[0]-2)*(np.sqrt(sigma1_hat*(1/y1.shape[0] + 1/y2.shape[0])))
CI1_R = np.mean(y1)-np.mean(y2) + scipy.stats.t.ppf(q=q,df=y1.shape[0]+y2.shape[0]-2)*(np.sqrt(sigma1_hat*(1/y1.shape[0] + 1/y2.shape[0])))
CI2_L = np.mean(y2)-np.mean(y3) - scipy.stats.t.ppf(q=q,df=y2.shape[0]+y3.shape[0]-2)*(np.sqrt(sigma2_hat*(1/y2.shape[0] + 1/y3.shape[0])))
CI2_R = np.mean(y2)-np.mean(y3) + scipy.stats.t.ppf(q=q,df=y2.shape[0]+y3.shape[0]-2)*(np.sqrt(sigma2_hat*(1/y2.shape[0] + 1/y3.shape[0])))
CI3_L = np.mean(y3)-np.mean(y4) - scipy.stats.t.ppf(q=q,df=y3.shape[0]+y4.shape[0]-2)*(np.sqrt(sigma3_hat*(1/y3.shape[0] + 1/y4.shape[0])))
CI3_R = np.mean(y3)-np.mean(y4) + scipy.stats.t.ppf(q=q,df=y3.shape[0]+y4.shape[0]-2)*(np.sqrt(sigma3_hat*(1/y3.shape[0] + 1/y4.shape[0])))
CI4_L = np.mean(y4)-np.mean(y5) - scipy.stats.t.ppf(q=q,df=y4.shape[0]+y5.shape[0]-2)*(np.sqrt(sigma4_hat*(1/y4.shape[0] + 1/y5.shape[0])))
CI4_R = np.mean(y4)-np.mean(y5) + scipy.stats.t.ppf(q=q,df=y4.shape[0]+y5.shape[0]-2)*(np.sqrt(sigma4_hat*(1/y4.shape[0] + 1/y5.shape[0])))


# ### Calculating Confidence Intervals

# * An ANOVA test can tell you if your results are significant overall, but it won't tell you exactly where those differences lie.

# In[ ]:


# CI for y1, y2
print("CI for difference of treatment means of y1, y2: [ {0} {1} ]".format(CI1_L,CI1_R))


# In[ ]:


# CI for y2, y3
print("CI for difference of treatment means of y2, y3: [ {0} {1} ]".format(CI2_L,CI2_R))


# In[ ]:


# CI for y3, y4
print("CI for difference of treatment means of y2, y3: [ {0} {1} ]".format(CI3_L,CI3_R))


# In[ ]:


# CI for y4, y5
print("CI for difference of treatment means of y2, y3: [ {0} {1} ]".format(CI4_L,CI4_R))


# # Solution 2

# ## 2 (a)

# * Even  though  the  error  is  independent  but  not  identically  distributed, we can still estimate the parameters of the model $Y=\beta X$ using the leastsquare estimator of $\beta$, i.e., $\hat{\beta}$.  
# * To find the $95\%$ prediction interval for $x= 2.5$, we estimate the value of $Y$ by $\hat{\beta} \tilde{x}$, where $\tilde{x}= [2.5,1]$ and resampling the residuals $(y−y_{pred})$ and adding it to the estimate at $x= 2.5$.  
# * Using this bootstrapped set of values, we find the $95\%$ two-tailed quantile, and we get the $95\%$ confidence intervalas $[12.89948,15.93856]$.

# In[ ]:


import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


# In[2]:


get_ipython().run_line_magic('load_ext', 'rpy2.ipython')


# In[3]:


get_ipython().run_cell_magic('R', '', 'data <- read.csv("data_regression.csv")\nreg <- lm(data$y ~ data$x)\nplot(data$x, data$y)\nsummary(reg)\n\n#Predict for x = 2.5\ny.p <- coef(reg)["(Intercept)"] + coef(reg)["data$x"]*2.5\nprint(y.p)\n\n# Replicate residuals\nres <- numeric(length = length(data$x))\nfor(i in 1:length(res)){\n    temp_pred <- coef(reg)["(Intercept)"] + coef(reg)["data$x"]*data$x[i]\n    res[i] <- data$y[i] - temp_pred\n}\nprint(res)\n\nboot <- 1000\nboot_draws <- sample(res, size = boot, replace = TRUE)\ny.p + quantile(boot_draws, probs = c(0.025, 0.975))')


# ## 2 (b)

# In[ ]:




