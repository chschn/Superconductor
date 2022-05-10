import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

autompg = pd.read_csv('D:\\Digethic\\Software_Engineering_I\\Project\\ml-tutorial\\data\\auto-mpg.csv', sep=';', header=0)
#autompg = pd.read_csv('..\\data\\auto_mpg.csv', sep=';', header=0)
print(autompg)

y = autompg["mpg"]
autompg.pop("mpg")
X_train, X_test, y_train, y_test = train_test_split(autompg, y, test_size=0.2)

print(X_train)
print(y_train)
print(X_test)
print(y_test)

#reg = LinearRegression().fit(X_train,y_train)
#print(reg.coef_)
#print(reg.score(X_train,y_train))

filename = 'D:\\Digethic\\Software_Engineering_I\\Project\\ml-tutorial\\data\\models\\auto-mpg.sav'
reg = pickle.load(open(filename, 'rb'))

y_predict = reg.predict(X_test)
print(y_predict, np.array(y_test))
