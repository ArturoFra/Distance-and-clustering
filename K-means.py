#!/usr/bin/env python
# coding: utf-8

# # Método K-means

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = np.random.random(90).reshape(30,3)
data


# In[3]:


c1 = np.random.choice(range(len(data)))
c2 = np.random.choice(range(len(data)))
clusters_center = np.vstack([data[c1], data[c2]])


# In[4]:


clusters_center


# In[5]:


from scipy.cluster.vq import vq


# In[6]:


vq(data, clusters_center)


# In[7]:


from scipy.cluster.vq import kmeans


# In[8]:


kmeans(data, clusters_center)


# In[9]:


kmeans(data, 2)


# In[ ]:




