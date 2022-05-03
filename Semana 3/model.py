import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
dataset = pd.read_csv("50_Startups.csv")
print(dataset.columns)
x=dataset.iloc[:,:4]
y=dataset.iloc[:,4:5]
from sklearn.preprocessing import OneHotEncoder
# # use when different features need different preprocessing
from sklearn.compose import make_column_transformer
column_trans = make_column_transformer((OneHotEncoder(),['State']),remainder='passthrough')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(column_trans,LinearRegression())
pipe.fit(x_train,y_train)
pred=pipe.predict(x_test)
print(pipe.score(x_test,y_test))
import pickle
pickle.dump(pipe, open('model.pkl', 'wb'))