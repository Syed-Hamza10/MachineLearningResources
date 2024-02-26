# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#1. Get/Download Dataset
#2. Import Libraries
#3. Import Data Set into memory objects
#4 .Fill/Handle Missing Data (if there is missing data)
#5. Make the data Categorical --> converting text values to numeric 
    #-- categorical 1,2,3 -OR- one hot encoding
#6. Split data into Training and Test sets --> X_train, X_test, y_train, y_test
#7. Feature Scaling --> bringing all features values in comparable ranges say [-1,1]
#8. Data Processing --> Good to go for data processing

###IMPORT LIBRARIES####
import numpy as np
# numpy supports large arrays and matrices and other 
#mathematical functions
import matplotlib.pyplot as plt
# for visualization and graphs
import pandas as pd
# libraries for data loading and manipulation

####IMPORT DATA SET#####
dataset = pd.read_csv('data.csv')
#independent variable X
X = dataset.iloc[:, :-1].values
#dependent variable y
y = dataset.iloc[:,3].values
#CHANGE THE INDEX FOR y


####DEALING WITH MISSING VALUES######

##import SimpleImputer class and apply strategy mean, 
#most_frequent, median

from sklearn.impute import SimpleImputer

#CHECK FOR THE STRATEGY
imputer = SimpleImputer(strategy = 'mean')

#CHANGE THE INDEX FOR APPROPRIATE COLUMNS OF X
imputer = imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:, 1:3])


####ENCODING CATEGORICAL DATA########
from sklearn.preprocessing import LabelEncoder
labelEncoder_X = LabelEncoder()
#CHANGE THE INDEX FOR APPROPRIATE COLUMNS OF X
X[:,0] = labelEncoder_X.fit_transform(X[:,0])

labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)

####ENCODING CATEGORICAL DATA OneHotEncoder########
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#CHANGE THE INDEX FOR APPROPRIATE COLUMNS OF X
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)

####SPLITTING DATA INTO TRAINING AND TEST SETS
from sklearn.model_selection import train_test_split
#CHECK FOR TRAIN TEST SPLIT RATIO
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
#X for independent variables and y for dependent variables
#random_state for random sampling
#we can also use stratified sampling (almost equal number of representation for each class)

####FEATURE SCALING

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
print(X_train)
X_test = sc_X.transform(X_test)
print(X_test)
#for classification problems we don't need to apply feature sacling on test variable, 
#however for regression we should apply it
