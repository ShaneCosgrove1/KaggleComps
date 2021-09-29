import pandas as pd

trainData = pd.read_csv(r"train.csv");
print(trainData)

y = trainData["Survived"]
print(y)

#