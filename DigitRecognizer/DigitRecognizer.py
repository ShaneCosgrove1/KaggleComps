#Digit Recognizer model by Shane Cosgrove
#Predict who survived from the test data
#Imports
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import metrics
#Read in data
test_data= pd.read_csv("/Users/Owner/KaggleComps/DigitRecognizer/test.csv")
train_data = pd.read_csv("/Users/Owner/KaggleComps/DigitRecognizer/train.csv")
#Print top rows of data
#print(train_data.head())
#print(test_data.head())

y_train = train_data["label"]

#print(y_train)

x_train = train_data.drop(labels = ["label"],axis = 1) 

#print(x_train.isnull().any().describe()) #check what columns contains nulls
#print(test_data.isnull().any().describe())


x_train, x_test, y_train,y_test = train_test_split(x_train,y_train,test_size=0.2,random_state=32)

#Create SVM Classifier
clf = svm.SVC(kernel='linear')

clf.fit(x_train, y_train)

y_pred = clf.predict(test_data)

print("Runnin File")

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


#Create a dataframe of predictions and output to csv file
output = pd.DataFrame({'Label':y_pred})
output.to_csv('submission.csv',index=True)
print("Your submission was successfully saved")

