import pandas as pd
import matplotlib as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv('automobile.csv')
print(data.describe())
print(data.head())

X = data.iloc[:,:-1]
y = data["price"]

model = DecisionTreeRegressor()

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model.fit(X_train, y_train)

