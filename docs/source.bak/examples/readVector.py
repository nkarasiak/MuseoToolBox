
# coding: utf-8

# # Read values from vector
# This notebook is an example on how to read fields values from vector (points/lines/polygons).

# In[1]:


from MuseoToolBox.vectorTools import readValuesFromVector


# In[2]:


inVector = 'data/train_withROI.gpkg'


# ## Read only one field
# 

# In[3]:


Y = readValuesFromVector(inVector,'class')


# ## Read as many fields as needed
# Just add as much as needed fields, separated by ','.

# In[4]:


Y,FID,band0 = readValuesFromVector(inVector,'Class','uniqueFID','band_0')


# In[5]:


print(Y[:5])
print(FID[:5])
print(band0[:5])


# # Read multiple fields with same prefix
# 
# To read multiple fields beginning with the same prefix (by using sampleExtraction tools for example),
# simply used **bandPrefix='band_'** to load each field beginning by *band_*.

# In[6]:


Y,X = readValuesFromVector(inVector,'Class',bandPrefix='band_')


# In[7]:


print(Y.shape)
print(X.shape)


# ## Get 10 randoms labels with raster values

# In[8]:


import numpy as np
randomPosition = np.random.randint(0,X.shape[0],10)
print('Values are : ')
print(X[randomPosition,:])


# In[9]:


print('Labels are : ')
print(Y[randomPosition])

