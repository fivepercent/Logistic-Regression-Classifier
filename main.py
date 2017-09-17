# Imports
import pandas as pd # load csv's (pd.read_csv)
import numpy as np # math (lin. algebra)
import math

data = pd.read_csv('西瓜数据集3_0a.csv',sep=',')
m=data.shape[0]
n=data.shape[1]-1
print ("Num of rows: "+ str(data.shape[0]))
print ("Num of columns: "+ str(data.shape[1]))

LR = LogisticRegression(None, None, None, None)
test_sample=np.array([[0.593,0.042]])
x,y=LR.loadData(data)
LR.fit(x,y)
LR.predict(test_sample)