# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 23:33:28 2019

@author: user
"""

import pandas as pd
import numpy as np
import scipy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling as profile   # To check data distributions and correlations
import warnings     # for supressing a warning when importing large files
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from pylab import rcParams
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers


data=pd.read_csv("/home/atul/Documents/ResearchPapers/journals/AutomobileFraud/DataSets/MAM/my_carclaims.csv")

data=data.replace({'Sex':2,'MaritalStatus':2,'Fault':2,'AccidentArea':2,'PoliceReportFiled:':2,'WitnessPresent':2,'AgentType':2},0)
data.head(1)
VehicleCategory= pd.get_dummies(data['VehicleCategory'],drop_first=True)

#DriverRatings= pd.get_dummies(data['DriverRating'],drop_first=True)
BasePolicys= pd.get_dummies(data['BasePolicy'],drop_first=True)

data.drop(['DriverRating','BasePolicy'],axis=1,inplace=True)

FraudFound=data['FraudFound']
data.drop(['FraudFound'],axis=1,inplace=True)
data=pd.concat([data,VehicleCategory,BasePolicys],axis=1)

train=data
test=FraudFound

X_train,X_test,y_train,y_test=train_test_split(train,test,test_size=0.3)

from sklearn.svm import SVC
clf=SVC()

# Training the dataset using Suport Vector Machines Model
clf=clf.fit(X_train,y_train)
# Prediction 

prediction=clf.predict(X_test)
print("The prediction by the machine learning model  is\n",prediction)
from sklearn.metrics import accuracy_score
# Finding the accuracy of the model
a=accuracy_score(y_test,prediction)
print("The accuracy of this model is: ", a*100)


