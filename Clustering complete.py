#!/usr/bin/env python
# coding: utf-8

# # Clustering con Python

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv("C:/Users/A Emiliano Fragoso/Desktop/MLcourse/python-ml-course-master/datasets/wine/winequality-red.csv", sep=";")


# In[3]:


data.head()


# In[5]:


data.shape


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


plt.hist(data["quality"])


# In[9]:


data.groupby("quality").mean()


# ### Normalizar los datos  

# In[11]:


df_nor=(data-data.min())/(data.max()-data.min())
df_nor.head()


# ### Clustering jerárquico con scikit learn 

# In[12]:


from sklearn.cluster import AgglomerativeClustering


# In[14]:


clus= AgglomerativeClustering(n_clusters=6, linkage="ward").fit(df_nor) 


# In[33]:


md_h=pd.Series(clus.labels_)
md_h


# In[17]:


plt.hist(md)
plt.title("Histograma de los clusters")
plt.xlabel("Cluster")
plt.ylabel("Número de vinos del cluster")


# In[18]:


clus.children_


# In[20]:


from scipy.cluster.hierarchy import dendrogram, linkage


# In[22]:


Z=linkage(df_nor, "ward")


# In[25]:


plt.figure(figsize=(25,10))
plt.title("Dendrograma de los vinos")
plt.xlabel("ID del vino ")
plt.ylabel("Distancia")
dendrogram(Z, leaf_rotation=90., leaf_font_size=12.)
plt.show()


# ## K means 

# In[27]:


from sklearn.cluster import KMeans
from sklearn import datasets


# In[30]:


model= KMeans(n_clusters=6)
model.fit(df_nor)


# In[31]:


model.labels_


# In[36]:


md_k=pd.Series(model.labels_)


# In[ ]:





# In[37]:


df_nor["clust_h"]= md_h
df_nor["clust_k"]=md_k


# In[38]:


df_nor.head()


# In[39]:


plt.hist(md_k)


# In[40]:


model.cluster_centers_


# In[41]:


model.inertia_


# #  Interpretación final

# In[43]:


df_nor.groupby("clust_k").mean()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




