import pandas as pd 
import numpy as np 


data = pd.read_csv("Data.csv")

#header=None
#data = pd.read_csv("Data_nohead.csv", header=None)

#read csv
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values	# .values  --> np.array
print(X)
print(y)
