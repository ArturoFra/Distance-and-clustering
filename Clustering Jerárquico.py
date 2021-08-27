#!/usr/bin/env python
# coding: utf-8

# # Clustering Jerárquico y dendrograms

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage


# In[7]:


np.random.seed(4711)
a=np.random.multivariate_normal([10,0],[[3,1],[1,4]], size=[100,])
b=np.random.multivariate_normal([0,20],[[3,1],[1,4]], size=[50,])
X=np.concatenate((a,b))
print(X.shape)
plt.scatter(X[:,0] ,X[:,1])
plt.show


# In[8]:


Z=linkage(X, "ward")
Z


# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


a


# In[13]:


from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist


# In[ ]:





# In[14]:


c, coph_dist = cophenet(Z, pdist(X))
c


# In[15]:


Z[0]


# In[20]:


idx = [33,62,68]
idx2=[15,41,69]
plt.figure(figsize=(10,8))
plt.scatter(X[:,0] ,X[:,1])
plt.scatter(X[idx,0], X[idx,1], c="r")
plt.scatter(X[idx2,0], X[idx2,1], c="y")
plt.show


# # Representación gráfica de un dendograma

# In[24]:


plt.figure(figsize=(25,10))
plt.title("Dendrograma del clustering Jerárquico")
plt.xlabel("Índices de la muestra")
plt.ylabel("Distancias")
dendrogram(Z, leaf_rotation=90., leaf_font_size=8.0, color_threshold=0.1*180)
plt.show()


# # Truncar el dendrograma 
# 

# In[28]:


plt.figure(figsize=(25,10))
plt.title("Dendrograma del clustering Jerárquico")
plt.xlabel("Índices de la muestra")
plt.ylabel("Distancias")
dendrogram(Z, leaf_rotation=90., leaf_font_size=13, color_threshold=0.1*180,
          truncate_mode = "lastp", p=12 , show_leaf_counts=True, show_contracted=True)
plt.show()


# # Dendrograma tuneado

# In[40]:


def dendrogram_tun(*args, **kwargs):
    max_d=kwargs.pop("max_d", None)
    if max_d and "color_treshold" not in kwargs:
        kwargs["color_treshold"]=max_d
    annotate_above = kwargs.pop("annotate_above", 0)
    
    ddata = dendrogram(*args, **kwargs)
    
    if not kwargs.get("no_plot", False):
        plt.title("Clustering Jerárquico con dendrograma truncado")
        plt.xlabel("Índice del dataser (o tamaño del cluster)")
        plt.ylabel("Distancia")
        for i,d,c in zip(ddata['icoord'], ddata["dcoord"], ddata["color_list"]):
            x=0.5 * sum(i[1:3])
            y=d[1]
            if y>annotate_above:
                plt.plot(x,y,"o",c=c)
                plt.annotate("%.3g"%y, (x,y), xytext = (0, -5), textcoords= "offset points", va="top", ha="center")
    if max_d:
        plt.axhline(y=max_d, c="k")
    return ddata

    


# In[ ]:





# In[42]:


dendrogram_tun(Z, truncate_mode = "lastp", p=12, leaf_rotation=90., leaf_font_size=12., show_contracted=True)
plt.show


# # Corte automático del dendrograma 

# In[43]:


from scipy.cluster.hierarchy import inconsistent


# In[44]:


depth = 5


# In[48]:


incons = inconsistent(Z, depth)
incons[-10:]


# In[ ]:





# # Método del codo

# In[56]:


last = Z[-10:,2]
last_rev=last[::-1]
idx=np.arange(1, len(last)+1)
plt.plot(idx, last_rev)

acc=np.diff(last,2)
acc_rev = acc[::-1]
plt.plot(idx[:-2]+1, acc_rev)
plt.show()
k=acc_rev.argmax() + 2
print(k)


# In[57]:


c= np.random.multivariate_normal([40,40],[[20,1], [1,30]], size=[200,] )
d= np.random.multivariate_normal([80,80],[[30,1], [1,30]], size=[200,] )
e= np.random.multivariate_normal([0,100],[[100,1], [1,100]], size=[200,] )
X2=np.concatenate((X,c,d,e),)
plt.scatter(X2[:, 0], X2[:,1])
plt.show()


# In[58]:


Z2= linkage(X2, "ward")


# In[63]:


plt.figure(figsize=(10,10))
dendrogram_tun(Z2, truncate_mode = "lastp", p=30, leaf_rotation=90., leaf_font_size=12.,  
               annotate_above=40)
plt.show()


# In[64]:


last = Z2[-10:,2]
last_rev=last[::-1]
idx=np.arange(1, len(last)+1)
plt.plot(idx, last_rev)

acc=np.diff(last,2)
acc_rev = acc[::-1]
plt.plot(idx[:-2]+1, acc_rev)
plt.show()
k=acc_rev.argmax() + 2
print(k)


# # Recuperando clusters y elementos 

# In[65]:


from scipy.cluster.hierarchy import fcluster


# In[74]:


max_d=25
clusters =fcluster(Z,max_d,criterion="distance")
clusters


# In[75]:


k=2
clusters =fcluster(Z,k,criterion="maxclust")
clusters


# In[69]:


fcluster(Z, 8, depth=10)


# In[76]:


plt.figure(figsize=[10,8])
plt.scatter(X[:,0], X[:,1], c= clusters, cmap="prism")
plt.show()


# In[81]:


max_d=170
clusters =fcluster(Z2,max_d,criterion="distance")
clusters
plt.figure(figsize=[10,8])
plt.scatter(X2[:,0], X2[:,1], c= clusters, cmap="prism")
plt.show()


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




