import pandas as pd
import os

'''Load the data'''
train_data = os.getcwd() + "/disease_train_data.csv"
test_data = os.getcwd() + "/disease_test_data.csv"

train_data = pd.read_csv(train_data)
test_data = pd.read_csv(test_data)

'''Searching for NA values and modifying or dropping the values'''
print(train_data.columns[train_data.isnull().any()])
print(test_data.columns[test_data.isnull().any()])

#A training column is NA, let us check the values
print(train_data.loc[0:20, 'Unnamed: 133'])

#The column has only NA values, so it must be removed
train_data.drop(train_data.columns[133], axis=1, inplace=True)

#print(test_data.isnull())
#print(test_data.keys())
#print(train_data.keys())
#print(len(train_data.columns))
#print(len(test_data.columns))
#train_data.fillna(//insert method here for replacing na values)

'''Inspecting feature types'''
print(train_data.dtypes.value_counts())
"""
All columns have integer data type, except for the 'prognosis' columns, which
is 'object' type or strings in this dataset
"""

'''Inspecting feature scales/values?'''
x = 0
y = 0
scale_set = []
while (x < len(train_data.columns)-1):
    scale = (train_data.iloc[:, x].unique())
    scale = list(scale)
    if scale not in scale_set:
        scale_set.append(scale)
    x = x+1
    
print(scale_set)
    
#Looks like the scale is 0-1, with some features having only 0 values

'''Checking features with only 0 values'''
