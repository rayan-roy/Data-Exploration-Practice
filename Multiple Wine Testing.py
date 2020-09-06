#!/usr/bin/env python
# coding: utf-8

# # We will be using two wine dataset and exploring them
# There is one dataset of whitewine, and another dataset of red wine
# 

# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# To ignore warnings
import warnings
warnings.filterwarnings('ignore')


# Note that the files we will be using has a ";" deliminator for separation

# In[3]:


red_wine_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=";")
white_wine_data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=";")


# ## Data Exploration (EDA)
# 

# In[4]:


red_wine_data.head()


# In[7]:


red_wine_data.info()
# We see that there no null values, so we don't need to drop anything, or do any dataprocessing
# We also see that all the data is numerical


# In[10]:


# Use Describe function to check mean, median, count..
red_wine_data.describe()


# In[4]:


white_wine_data.head()


# In[12]:


white_wine_data.info()
# We learn that there are no missing value; all values are numerical
# We also see that there are more rows/data in the white wine than red wine


# In[13]:


red_wine_data.columns


# In[14]:


white_wine_data.columns


# Line 13 and 14 tells us that all the columns in both red wine and white wine are the same.
# We will add a column where r is red wine and w is for white wine, and then combine the

# In[23]:


# We are creating a column color; you need 'r' instead of r as its a string
red_wine_data['color'] = 'r'
white_wine_data['color'] = 'w'


# In[24]:


red_wine_data.head()


# In[19]:


# To delete a column; we will be removing column colo
red_wine_data.drop(columns = "colo", axis = 1, inplace = True) 


# In[25]:


# To see if the column was removed
red_wine_data.head()


# In[21]:


white_wine_data.head()


# In[28]:


# Now we have the same number of columns, we will be merging them together
# We will be using concat function
combined_wine_data = pd.concat([red_wine_data,white_wine_data])


# In[29]:


# This will help us the top which is r
combined_wine_data.head(20)

# This will help us see the bottom which is w
combined_wine_data.tail(20)


# In[30]:


# To rename the columns, we can try:
# We will be replacing the space between values with dash (-)
combined_wine_data.rename(columns={'fixed acidity': 'fixed-acidity', 'citric acid':'citric-acid', 'volatile acidity':'volatile-acidity',
                          'residual sugar':'residual-sugar', 'free sulfur dioxide':'free-sulfur-dioxide', 'total sulfur dioxide':'total-sulfur-dioxide'},
                 inplace = True)


# In[22]:


# We see the changes have been made
combined_wine_data.head()


# ## Data Visualization

# The first visualization is a histogram
# 

# In[32]:


# We will see that there is a histogram for each column
# Note the bins is like how it works normally in statistics
# For figsize, its (length_size, width_size)
red_wine_data.hist(bins=50, figsize=(16,12))  
plt.show()


# We see that pH and density are normal distributed, while the rest of the plots are skewed to the right.
# We also that volatile acidity is bimodal distributed.
#     From the quality graph, we see that the level 5, level 6 are the highest meaning the medium quality wines are higher than good or bad quality wines.
#     
# Lets test white wines now

# In[33]:


# We keep the histogram size, length and width all same
white_wine_data.hist(bins=50, figsize=(16,12))  
plt.show()


# We see that in white wine pH is only normal distributed, while all others are right skewed. 
# 

# We will be testing the pivot table features:
# pandas.pivot_table(data, values=None, index=None, columns=None, aggfunc='mean', fill_value=None, margins=False, dropna=True, margins_name='All', observed=False)
# This is the format, where data is full data, aggfunc you can decide like np.mean, np.median, np.sum, you also need the
# columns by default

# In[10]:


# We will be seeing some correlation, or if there is some relationship between the factors
white_wine_data.corr()


# We see that there is high positive correlation between fixed_acidity and citric_acid, fixed_acidity and density. Similarly we can observe there is a relatively high negative correlation between fixed_acidity and pH

# In[11]:


#Similarly for red win
red_wine_data.corr()


# We see there is positive correlation between fixe acid and density, fixed acid and citric acid, but negative cor b/w ph and fixed acidity

# In[13]:


# We will be using a heatmap to visualize this correlation matrix this:
# This is for red wines
plt.figure(figsize=(15, 12))
sns.heatmap(red_wine_data.corr(), cmap='bwr', annot=True) 


# In[14]:


# This is for white wines
plt.figure(figsize=(15, 12))
sns.heatmap(white_wine_data.corr(), cmap='bwr', annot=True) 


# In[8]:


white_wine_data.pivot_table(columns, ['quality'], aggfunc=np.median)


# # Visualizing categorical data
# 

# We usually use a barplot or countplot to visualize categorical attributes

# In[40]:


# This is the attribute of quality of the wine
plt.figure(figsize=(12,8))
sns.countplot(combined_wine_data.quality, hue=combined_wine_data.color)#


# We see that the majority quality of both white and red wine is 5-6 which is medium quality which is higher than both the low and high quality of wine

# ## Scatterplot (pairplot in coding):
# We can visualize scatterplot matrix for the better understanding relationship between each pair of variables. It plots every numerical attribute against every other. It usually helps us decipher the heatmap/correlation we found, whether there were any outliers that might have caused issue

# In[41]:


# scatterplot for red wine
sns.pairplot(red_wine_data)


# Notice that at the very bottom of all the graphs is axis. You will also see that there is a histogram in the diagonal, its because both the attributes are same, and it doesn't really give much info.
# 
# We see that there is positive correlation in density vs fixed acidity,and neg cor between ph vs fixed acidity, while other attributes correlation are sparse. You also see lines of dots on the quality attribute because its a categorical data
# 

# In[42]:


sns.pairplot(white_wine_data)


# In[ ]:




