
# coding: utf-8

# ### Getting Started with Machine Learning using Scikit-learn Python Module.
# 

# Prerequisit : Knoledge of Machine Learning : Supervised Algorithm (Regression ), Python Programming

# # Introducing the Iris Dataset
# Machine Learning on the Iris Dataset
# 1. It is a Supervised Learning Problem. To Predict the species of and Iris using the measurements
# 2. Its a Famous dataset to beginners.
# 3. We will see here Regression  Algorithm from sklearn

# In[1]:


# Here we are importing SVM (Support vector Machine , Built in Iris Dataset, Numpy )
from sklearn import svm
# Import dataset to load the data
from sklearn import datasets
import seaborn as sns
import numpy as np
from sklearn.cross_validation import train_test_split


# In[2]:


# load_iris function from datasets module
iris=datasets.load_iris()
iris1=sns.load_dataset("iris")


# In[3]:


print(iris1.tail())


# In[4]:


# The type of dataset is saved as "bunch" object , contains Iris dataset and its attributes
print(type(iris))
print(type(iris1))


# In[5]:


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

# In[13]:


x=iris.data
print(x.shape)


# In[14]:


y=iris.target
print(y.shape)


# In[15]:



x_train, x_test, y_train, y_test= train_test_split(x , y , test_size= 0.3, random_state=4)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[16]:


#from sklearn.linear_model import LogisticRegression
#model=LogisticRegression()
model=svm.SVC(kernel='linear')
#model=svm.SVC()


# In[17]:


#xtrain=x_train.reshape(-1,1)
#ytrain=y_train.reshape(-1,1)
#xtest=x_test.reshape(-1,1)
#ytest=y_test.reshape(-1,1)


# In[18]:


model.fit(x_train , y_train)


# In[19]:


accuraccy=model.score(x_test, y_test)


# In[20]:


ypred=model.predict(x_test)
print(ypred, "\nsize :", ypred.shape)
print(iris.target_names)


# In[21]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[22]:


print(accuracy_score(y_test, ypred))


# In[23]:


print(classification_report(y_test, ypred))


# In[24]:


# Now the model is train and one can test with new value
print(model.predict([[3,5,4,2]]))
print(model.predict([[6.7,3.0,5.2,2.3]]))
x_new=[[3,5,4,2],[5,4,3,2],[6.7,3.0,5.2,2.3]]
print(model.predict(x_new))


# ### ---------- END---------------

# # Using KNN (K Nearest Neighbors) Algorithm

# In[25]:



from sklearn.datasets import load_iris

iris=load_iris()
x=iris.data
y=iris.target
print(x.shape)
print(y.shape)
# KNN
#  Import the class for KNN
from sklearn.neighbors import KNeighborsClassifier


# In[45]:


# create the object of KNeighborsClassifier
# to select nearest neighbors 1
knn=KNeighborsClassifier(n_neighbors=5)    # n=5
knn1=KNeighborsClassifier(n_neighbors=1)# n=1


# In[27]:


print(knn)


# #### Fit the model with data
# #### Here model is learning relationship between Independent (x) variable and dependent variable(y)
# 1. KNN is simple Machine Learning model but it will make Highly Accurate Prediction
# 2. k=1, whill check nearest node and decide the class
# 3. k=5, will check , out of 5 how many nearest neighbors are close to the test node. 
#    Then accordingly decide the test class.

# In[37]:


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.3, random_state=4)


# In[38]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)

print(y_test.shape)


# In[46]:


knn.fit(x_train, y_train)   # knn, n=5
knn1.fit(x_train, y_train)   # knn, n=1


# In[47]:


#ypred=knn.predict(x)
#print(ypred)
ytestpred=knn.predict(x_test) # n=5
print("ytestpred = ", ytestpred)
ytestpred1=knn1.predict(x_test)  #n=1


# In[50]:


from sklearn import metrics
#print(metrics.accuracy_score(y,ypred))
print("accuracy N=5", metrics.accuracy_score(y_test,ytestpred))     #knn, n=5
print("accuracy N=1",metrics.accuracy_score(y_test, ytestpred1))  # knn, n=1


# In[53]:


# Test prediction with new value  ( This code is executed with splitting data using train test split)
print(knn.predict([[3,5,4,2]]))
x_new=[[3,5,4,2],[5,4,3,2],[6.7,3.0,5.2,2.3]]
print(knn.predict(x_new))


# In[52]:


# Now the model is train and one can test with new value
print(knn.predict([[3,5,4,2]]))
print(knn.predict([[6.7,3.0,5.2,2.3]]))
x_new=[[3,5,4,2],[5,4,3,2],[6.7,3.0,5.2,2.3]]
print(knn.predict(x_new))


# # Using Logistic Regression Model

# In[54]:


# using Logistic Regression Model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
lgr=LogisticRegression()
lgr.fit(x,y)
lgr.predict(x)


# In[55]:


ypred=lgr.predict(x)


# In[56]:


len(ypred)


# In[61]:


print("Accuracy without data split : ", metrics.accuracy_score(y,ypred))


# In[62]:


# Now the model is train and one can test with new value   (Without data split)
print(lgr.predict([[3,5,4,2]]))
print(lgr.predict([[6.7,3.0,5.2,2.3]]))
x_new=[[3,5,4,2],[5,4,3,2],[6.7,3.0,5.2,2.3]]
print(lgr.predict(x_new))


# In[63]:


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.3, random_state=4)


# In[66]:


# prediction after data split ,ypredtest on test data.
ytestpred=lgr.predict(x_test)
print("Accuracy after data split:", metrics.accuracy_score(y_test , ytestpred))


# In[65]:


# Now the model is train and one can test with new value   (With data split)
print(lgr.predict([[3,5,4,2]]))
print(lgr.predict([[6.7,3.0,5.2,2.3]]))
x_new=[[3,5,4,2],[5,4,3,2],[6.7,3.0,5.2,2.3]]
print(lgr.predict(x_new))


# ### Conclusion : We have tested three Model:
#                                                 Linear regression using SVM : 98%,
#                                                 KNN algorithm : 98% (N=5) and
#                                                 Logistic regression : 96%
