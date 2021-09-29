#Titanic model by Shane Cosgrove
#Predict who survived from the test data
#Imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Read in data
test_data= pd.read_csv("/Users/Owner/KaggleComps/TitanicComp/test.csv")
train_data = pd.read_csv("/Users/Owner/KaggleComps/TitanicComp/train.csv")
#Print top rows of data

#print(train_data.head())
#print(test_data.head())

#Get survival rate of men and women from training data
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

print("% of women who survived:", rate_women)

y = train_data["Survived"]

features = ["Pclass","Sex","SibSp","Parch"]

X_train= pd.get_dummies(train_data[features]) #Dummy encoded values in features
X_test = pd.get_dummies(test_data[features])

#Random Forest Model
model = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=32)
model.fit(X_train,y)
print(model.score(X_train,y))
predictions = model.predict(X_test)

#Create a dataframe of predictions and output to csv file
output = pd.DataFrame({'PassengerID': test_data.PassengerId,'Survived':predictions})
output.to_csv('submission.csv',index=False)
print("Your submission was successfully saved")

