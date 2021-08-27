#!/usr/bin/env python
# coding: utf-8

# # Distancias

# In[2]:


from scipy.spatial import distance_matrix
import pandas as pd 
import numpy as np


# In[ ]:





# In[3]:


data = pd.read_csv("C:/Users/A Emiliano Fragoso/Desktop/MLcourse/python-ml-course-master/datasets/movies/movies.csv", sep=";")
data.head()


# In[ ]:





# In[ ]:





# In[4]:


movies = data.columns.values.tolist()[1:]


# In[5]:


movies


# In[6]:


dd1=distance_matrix(data[movies], data[movies], p=1)
dd2=distance_matrix(data[movies], data[movies], p=2)
dd10=distance_matrix(data[movies], data[movies], p=10)


# In[7]:


dd1


# In[ ]:





# In[ ]:





# In[8]:


def dm_to_df(dd, col_name):
    import pandas as pd
    return pd.DataFrame(dd, index=col_name, columns = col_name)


# In[ ]:





# In[ ]:





# In[9]:


dm_to_df(dd1, data["user_id"])


# In[ ]:





# In[ ]:





# In[10]:


dm_to_df(dd2, data["user_id"])


# In[ ]:





# In[ ]:





# In[11]:


dm_to_df(dd10, data["user_id"])


# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 


# In[13]:


fig=plt.figure()
ax=fig.add_subplot(111, projection = "3d")
ax.scatter(xs = data["star_wars"], ys= data["lord_of_the_rings"], zs= data["harry_potter"])


# In[ ]:





# In[ ]:





# # Enlaces

# In[14]:


df= dm_to_df(dd1, data["user_id"])
df


# In[ ]:





# In[15]:


Z=[]


# In[17]:


x=1
y=10
df[11]=df[1]+df[10]
df.loc[11]=df.loc[1] + df.loc[10]
Z.append([1,10,0.7,2])  ##id1, id2, d, n elmentos del cluster 

for j in df.columns.values.tolist():
    df.loc[11][j] = min(df.loc[x][j], df.loc[y][j])
    df.loc[j][11] = min(df.loc[j][x], df.loc[j][y])

df=df.drop([1,10])
df=df.drop([1,10], axis=1)
df


# In[18]:


x=2
y=7
n=12
df[n]=df[x]+df[y]
df.loc[n]=df.loc[x] + df.loc[y]
Z.append([x,y,df.loc[x][y],2])  ##id1, id2, d, n elmentos del cluster 

for j in df.columns.values.tolist():
    df.loc[n][j] = min(df.loc[x][j], df.loc[y][j])
    df.loc[j][n] = min(df.loc[j][x], df.loc[j][y])

df=df.drop([x,y])
df=df.drop([x,y], axis=1)
df


# In[19]:


x=5
y=8
n=13
df[n]=df[x]+df[y]
df.loc[n]=df.loc[x] + df.loc[y]
Z.append([x,y,df.loc[x][y],2])  ##id1, id2, d, n elmentos del cluster 

for j in df.columns.values.tolist():
    df.loc[n][j] = min(df.loc[x][j], df.loc[y][j])
    df.loc[j][n] = min(df.loc[j][x], df.loc[j][y])

df=df.drop([x,y])
df=df.drop([x,y], axis=1)
df


# In[20]:


x=11
y=13
n=14
df[n]=df[x]+df[y]
df.loc[n]=df.loc[x] + df.loc[y]
Z.append([x,y,df.loc[x][y],2])  ##id1, id2, d, n elmentos del cluster 

for j in df.columns.values.tolist():
    df.loc[n][j] = min(df.loc[x][j], df.loc[y][j])
    df.loc[j][n] = min(df.loc[j][x], df.loc[j][y])

df=df.drop([x,y])
df=df.drop([x,y], axis=1)
df


# In[21]:


x=9
y=12
z=14
n=15
df[n]=df[x]+df[y]
df.loc[n]=df.loc[x] + df.loc[y]
Z.append([x,y,df.loc[x][y],3])  ##id1, id2, d, n elmentos del cluster 

for j in df.columns.values.tolist():
    df.loc[n][j] = min(df.loc[x][j], df.loc[y][j], df.loc[z][j])
    df.loc[j][n] = min(df.loc[j][x], df.loc[j][y], df.loc[j][z])

df=df.drop([x,y,z])
df=df.drop([x,y,z], axis=1)
df


# In[22]:


x=4
y=6
z=15
n=16
df[n]=df[x]+df[y]
df.loc[n]=df.loc[x] + df.loc[y]
Z.append([x,y,df.loc[x][y],3])  ##id1, id2, d, n elmentos del cluster 

for j in df.columns.values.tolist():
    df.loc[n][j] = min(df.loc[x][j], df.loc[y][j], df.loc[z][j])
    df.loc[j][n] = min(df.loc[j][x], df.loc[j][y], df.loc[j][z])

df=df.drop([x,y,z])
df=df.drop([x,y,z], axis=1)
df


# In[23]:


x=3
y=16
n=17
df[n]=df[x]+df[y]
df.loc[n]=df.loc[x] + df.loc[y]
Z.append([x,y,df.loc[x][y],2])  ##id1, id2, d, n elmentos del cluster 

for j in df.columns.values.tolist():
    df.loc[n][j] = min(df.loc[x][j], df.loc[y][j])
    df.loc[j][n] = min(df.loc[j][x], df.loc[j][y])

df=df.drop([x,y])
df=df.drop([x,y], axis=1)
df


# In[24]:


Z


# # Clustering Jerárquico

# In[26]:


import matplotlib.pyplot as plt 
from scipy.cluster.hierarchy import dendrogram, linkage


# In[27]:


movies


# In[28]:


data[movies]


# In[29]:


Z=linkage(data[movies], "ward")


# In[31]:


Z


# In[36]:


plt.figure(figsize=(25,10))
plt.title("Dendrogram jerárquico para clustering")
plt.xlabel=("ID de los usuarios de Netflix")
plt.ylabel = ("Distancia")
dendrogram(Z, leaf_rotation=90., leaf_font_size=18)
plt.show


# In[ ]:





# In[37]:


Z=linkage(data[movies], "average")


# In[38]:


plt.figure(figsize=(25,10))
plt.title("Dendrogram jerárquico para clustering")
plt.xlabel=("ID de los usuarios de Netflix")
plt.ylabel = ("Distancia")
dendrogram(Z, leaf_rotation=90., leaf_font_size=18)
plt.show


# In[39]:


Z=linkage(data[movies], "complete")
plt.figure(figsize=(25,10))
plt.title("Dendrogram jerárquico para clustering")
plt.xlabel=("ID de los usuarios de Netflix")
plt.ylabel = ("Distancia")
dendrogram(Z, leaf_rotation=90., leaf_font_size=18)
plt.show


# In[40]:


Z=linkage(data[movies], "single")
plt.figure(figsize=(25,10))
plt.title("Dendrogram jerárquico para clustering")
plt.xlabel=("ID de los usuarios de Netflix")
plt.ylabel = ("Distancia")
dendrogram(Z, leaf_rotation=90., leaf_font_size=18)
plt.show


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





# In[17]:





# In[ ]:





# In[ ]:




