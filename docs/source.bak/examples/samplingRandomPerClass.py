
# coding: utf-8

# # Generate Random Cross-Validation per class

# In[1]:


from MuseoToolBox.vectorTools import crossValidationSelection

inVector = 'data/train.gpkg'
levelField = 'Class'


# ## Select a sampling Method
# In crossValidationSelection, a class samplingMethods contains a function for each method.
# Here we choose the randomCV to generate a Cross-Validation with a percentage per class (or number of train size equal for each class).

# In[2]:


samplingMethod = crossValidationSelection.samplingMethods.randomCV(train_size=0.5,nIter=5,seed=12)
crossValidation = crossValidationSelection.sampleSelection(inVector,levelField,samplingMethod)


# Now the crossValidation is ready to compute. You have two choices : 
# ### Generate the Cross-Validation for Scikit-Learn

# In[3]:


CV = crossValidation.getCrossValidationForScikitLearn()
# print each idx
for tr,vl in CV:
    print(tr)
    print(vl)


# ### Save the Cross-Validation in as many as files as training/validation iteration.
# 
# As Cross-Validation are generated on demand, you have to reinitialize the process and please make sure to have defined a seed to have exactly the same CV.

# In[4]:


CV = crossValidation.saveVectorFiles('data/cv.sqlite')
for tr,vl in CV:
    print(tr,vl)

