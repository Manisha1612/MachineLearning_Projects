
# coding: utf-8

# ### Getting Started with Machine Learning using Scikit-learn Python Module.
# 

# Prerequisit : Knoledge of Machine Learning : Supervised Algorithm (Regression , KNN), Python Programming

# # Introducing the Iris Dataset
# Machine Learning on the Iris Dataset
# 1. It is a Supervised Learning Problem. To Predict the species of and Iris using the measurements
# 2. Its a Famous dataset to beginners.
# 3. We will see here Regression and K-Nearest-Neighbour (KNN) Algorithm from sklearn

# In[2]:


# Here we are importing SVM (Support vector Machine , Built in Iris Dataset, Numpy )
from sklearn import svm
# Import dataset to load the data
from sklearn import datasets
import seaborn as sns
import numpy as np
from sklearn.cross_validation import train_test_split


# In[3]:


# load_iris function from datasets module
iris=datasets.load_iris()
iris1=sns.load_dataset("iris")


# In[4]:


print(iris1.head())


# In[5]:


# The type of dataset is saved as "bunch" object , contains Iris dataset and its attributes
print(type(iris))
print(type(iris1))


# In[30]:


iris1.species.value_counts()


# In[6]:


# print the iris dataset
print(iris.data)


# In[7]:


# retuns you total no. of datapoints are there 
iris.data.shape


# In[8]:


# it will give you name of the four features
iris.feature_names


# In[9]:


iris.target


# In[10]:


# to check target values
iris.target.shape


# In[11]:


# Will give you name of flowers which are encoded as 0:Setosa, 1: versicolor, 2 : verginica
iris.target_names


# In[12]:


# print data type of features and response
print(type(iris.data))
print(type(iris.target))


# ### Note :
# #### 1. The value which we are predicting is the response, its a dependent variable.
# #### 2. In Classification(Supervised Learning) , response is categorical
# #### 3. In Regression  (Supervised Learning), response is continuous or ordered.

# # Requirement to work with scikit-learn
# 1. Data and response should be seperated
# 2. All data should be in Numeric Form
# 3. Data should be in structured form (in specific shape)

# In[14]:


x=iris.data
print(x.shape)


# In[15]:


y=iris.target
print(y.shape)


# In[31]:



x_train, x_test, y_train, y_test= train_test_split(x , y , test_size= 0.3, random_state=4)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[32]:


#from sklearn.linear_model import LogisticRegression
#model=LogisticRegression()
model=svm.SVC(kernel='linear')
#model=svm.SVC()


# In[33]:


#xtrain=x_train.reshape(-1,1)
#ytrain=y_train.reshape(-1,1)
#xtest=x_test.reshape(-1,1)
#ytest=y_test.reshape(-1,1)


# In[34]:


model.fit(x_train , y_train)


# In[35]:


accuraccy=model.score(x_test, y_test)


# In[36]:


ypred=model.predict(x_test)
print(ypred, "\nsize :", ypred.shape)
print(iris.target_names)


# In[37]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[38]:


print(accuracy_score(y_test, ypred))


# In[39]:


print(classification_report(y_test, ypred))


# ### ---------- END---------------

# In[ ]:


# KNN
#  Import the class for KNN

