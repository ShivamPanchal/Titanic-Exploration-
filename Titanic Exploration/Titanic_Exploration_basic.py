
# coding: utf-8

# ## Titanic - Exploration 
# In this notebook, we will see the basic data exploration of the Titanic dataset 

# In[33]:

# importing the necessary packages #
import pandas as pd
import numpy as np
from sklearn import linear_model as lm
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

# reading the train file and saving it as a pandas data frame #
train_df = pd.read_csv('../Data/train.csv')
test_df = pd.read_csv('../Data/test.csv')


# In[3]:

# dimensions of the input data (number of rows and columns) #
print "Train dataframe shape is : ", train_df.shape
print "Test dataframe shape is : ", test_df.shape


# In[4]:

# name of the columns #
train_df.columns


# In[5]:

test_df.columns


# In[6]:

# taking a look at the first few rows #
train_df.head(10)


# In[7]:

# getting the summary statistics of the numerical columns #
train_df.describe()


# In[8]:

# getting the datatypes of the individual columns #
train_df.dtypes


# In[9]:

# more information about the dataset #
train_df.info()


# In[10]:

test_df.info()


# So, there are 891 rows in train set and 418 rows in test set. Also as we can see, most of the columns are not null. There are few columns which have null values as well. They are:
#  1. Age
#  2. Cabin
#  3. Embarked
#  4. Cabin
#  
# Out of this, 'Cabin' variable is Null for most part of the rows. So it is better to remove the 'Cabin' variable for model building.

# In[11]:

# dropping the cabin variable #
train_df.drop(['Cabin'], axis=1, inplace=True)
test_df.drop(['Cabin'], axis=1, inplace=True)


# ## Univariate Plots 
# 
# Now we will try to plot the given variables to see how they are distributed. 

# In[12]:

# let us get some plots to see the data #
train_df.Survived.value_counts().plot(kind='bar', alpha=0.6)
plt.title("Distribution of Survival, (1 = Survived)")


# In[13]:

# scatter plot between survived and age #
plt.scatter(range(train_df.shape[0]), np.sort(train_df.Age), alpha=0.2)
plt.title("Age Distribution")


# In[14]:

train_df.Pclass.value_counts().plot(kind="barh", alpha=0.6)
plt.title("Class Distribution")


# ##### TO DO:
# 1. Create a bar graph for the variable 'Embarked'.
# 2. Create a scatter plot for varibale 'fare' and check how it is distributed.

# ## Plots with DV
# Now we will make plots with DV to understand the relationship of the variables with DV

# In[17]:

train_male = train_df.Survived[train_df.Sex == 'male'].value_counts().sort_index()
train_female = train_df.Survived[train_df.Sex == 'female'].value_counts().sort_index()

ind = np.arange(2)
width = 0.3
fig, ax = plt.subplots()
male = ax.bar(ind, np.array(train_male), width, color='r')
female = ax.bar(ind+width, np.array(train_female), width, color='b')
ax.set_ylabel('Count')
ax.set_title('DV count by Gender')
ax.set_xticks(ind + width)
ax.set_xticklabels(('DV=0', 'DV=1'))
ax.legend((male[0], female[0]), ('Male', 'Female'))
plt.show()


# #### TO DO:
# 1. Plot a bar graph between DV and Pclass to see how the DV is distributed between the classes
# 2. Draw and "Box and Whisker plot" between DV and age and see the distribution between age and DV

# ## Supervised Machine Learning
# ### Logistic Regression

# Our competition wants us to predict a binary outcome. That is, it wants to know whether some will die, (represented as a 0), or survive, (represented as 1).
# 
# Logistic Regression is a method to solve these kind of problems. Please read about logistic regression to have a deeper understanding.

# In[38]:

# getting the necessary columns for building the model #
train_X = train_df[["Pclass", "SibSp", "Parch", "Fare"]]
train_y = train_df["Survived"]
test_X = test_df[["Pclass", "SibSp", "Parch", "Fare"]]


# ### Cross Validation
# 
# If we build models on the whole train dataset, how do we know the performance on the model on a new dataset?? 
# 
# So what we can instead do is to build the model on a part fo the dataset and then test it on the other part so that we get an idea of how our model performs on a new data. This process is known as Model Validation in Machine Learning field.
# 
# So now let us split the train data into two parts
# 1. Developement sample
# 2. Validation Sample

# In[40]:

# split the train data into two samples #
dev_X, val_X, dev_y, val_y = train_test_split(train_X, train_y, test_size=0.33, random_state=42)

# Build the machine learning model - in this case, logistic regression #
# Initialize the model #
clf = lm.LogisticRegression()

# Build the model on development sample #
clf.fit(dev_X, dev_y)

# Predict on the validation sample #
val_preds = clf.predict(val_X)
print val_preds[:10]


# So we got the validation sample classes as prediction outputs. Now it is time to check the performance of our model. We have our validation sample predictions and we have the validation sample true labels with us. 
# 
# Let us compute the accuracy then.!

# In[43]:

# import the function that computes the accuracy score #
from sklearn.metrics import accuracy_score

accuracy_score(val_y, val_preds)


# We can also compute other evaluation metrics like precision, recall etc.
# 
# We got the actual classes as outputs from our model. Instead if we need class probabilities, we can do the following

# In[46]:

val_preds = clf.predict_proba(val_X)
val_preds[:10]


# TO DO:
# 1. Add "Age" to the existing variables list and then re-run the logistic regression algorithm. 
# 2. In case of exceptions, how did you overcome the same?
