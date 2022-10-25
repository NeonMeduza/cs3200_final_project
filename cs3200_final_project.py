import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

'''Set seed'''
np.random.seed(1)

'''Load the data'''
train_data = os.getcwd() + "/disease_train_data.csv"
test_data = os.getcwd() + "/disease_test_data.csv"

train_data = pd.read_csv(train_data)
test_data = pd.read_csv(test_data)

'''Searching for NA values and modifying or dropping the values'''
#print(train_data.columns[train_data.isnull().any()])
#print(test_data.columns[test_data.isnull().any()])

#A training column is NA, let us check the values
#print(train_data.loc[0:20, 'Unnamed: 133'])

#The column has only NA values, so it must be removed
train_data.drop(train_data.columns[133], axis=1, inplace=True)
print(len(train_data))
print(len(test_data))

#print(test_data.isnull())
#print(test_data.keys())
#print(train_data.keys())
#print(len(train_data.columns))
#print(len(test_data.columns))
#train_data.fillna(//insert method here for replacing na values)

'''Inspecting feature types'''
#print(train_data.dtypes.value_counts())
"""
All columns have integer data type, except for the 'prognosis' columns, which
is 'object' type or strings in this dataset
"""

'''Inspecting feature scales/values'''
x = 0
scale_dict = {}
while (x < len(train_data.columns)-1):
    scale = (train_data.iloc[:, x].unique())
    scale = list(scale)
    keys = list(scale_dict.keys())
    if str(scale) not in keys:
        scale_dict[str(scale)] = [train_data.columns[x]]
    else:
        scale_dict[str(scale)] += [train_data.columns[x]]
    x = x+1
    
#print(scale_dict)
    
#Looks like the scale is 0-1, with some features having only 0 values

'''Checking features with only 0 values'''
#print("\n")
#print(scale_dict['[0]'])

'''
It looks like none of the diseases correspond to the 'fluid_overload'
symptom; what can we do with it?
'''


'''Meanwhile, we can encode the targets into more discrete values'''
<<<<<<< HEAD
#print("orig unique targets: ", np.unique(train_data['prognosis']))
#print("orig unique targets count: ", len((np.unique(train_data['prognosis']))))
encoder = LabelEncoder()
train_data['prognosis'] = encoder.fit_transform(train_data['prognosis'])
test_data['prognosis'] = encoder.fit_transform(test_data['prognosis'])
#print("unique targets: ", np.unique(train_data['prognosis']))
=======
print("orig unique targets: ", np.unique(train_data['prognosis']))
print("orig unique targets count: ", len((np.unique(train_data['prognosis']))))
encoder = LabelEncoder()
train_data['prognosis'] = encoder.fit_transform(train_data['prognosis'])
test_data['prognosis'] = encoder.fit_transform(test_data['prognosis'])
print("unique targets: ", np.unique(train_data['prognosis']))
encoder = LabelEncoder()
train_data['prognosis'] = encoder.fit_transform(train_data['prognosis'])
test_data['prognosis'] = encoder.fit_transform(test_data['prognosis'])
>>>>>>> e3a0970da7f6b704aeaa9c60a4c4c83689da5f7f


'''Let us now try some testing on KNN model'''
#Split data
x = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,
                                                    random_state=1)

<<<<<<< HEAD

=======
>>>>>>> e3a0970da7f6b704aeaa9c60a4c4c83689da5f7f
#Create model and train iteratively
neighbor_cnt = 1
neighbor_dict = {}
while (neighbor_cnt < 300):
    model = KNeighborsClassifier(n_neighbors=neighbor_cnt)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    neighbor_dict[neighbor_cnt] = (accuracy_score(preds, y_test) * 100)
    neighbor_cnt += 1
#print(neighbor_dict.keys())
#print(neighbor_dict.values())
    
#FIXME number of classes between preds and ground truth not equal
#print("Error score: ", (1 - accuracy_score(preds, y_test)) * 100)

#Plot for elbow method
plt.title("Accuracy with varying neighbors")
#plt.invert_xaxis()
plt.plot(neighbor_dict.keys(), neighbor_dict.values())

#Variance might be high here; decrease neighbors until accuracy is <95%
def filtering(x):
    i = 1
    while(i <= len(x)):
        if (x[i] < 95):
            optimal_k = i
            break
        i = i+1
    return optimal_k

opt_k = filtering(neighbor_dict)
print(opt_k)
print(neighbor_dict[opt_k])

#Seems 213 might be a good value for number of neighbors; let us test on
#other datasets to see if it generalizes well
splits = 4
k = 1

#copy training data
x_train_copy = x_train.copy()
y_train_copy = y_train.copy()

#store validation block
x_val = x_train_copy[int(((k-1)/splits) * len(x_train_copy)):int((k/splits) * len(x_train_copy))]
y_val = y_train_copy[int(((k-1)/splits) * len(y_train_copy)):int((k/splits) * len(y_train_copy))]

'''#FIXME remove valid block from training data
start_i = int(((k-1)/splits) * len(x_train))
end_i = int((k/splits) * len(x_train))
while start_i < end_i:
    x_train_copy.drop([start_i], inplace=True)
    y_train_copy.drop([start_i], inplace=True)
        
<<<<<<< HEAD
    start_i = start_i+1'''
=======
    total_scores[K] = avg_scores
    total_weights[K] = min_coef
            
    #Print the optimal validation block number and the average error score
    #print("Optimal validation block: ", min_key)
    #print("Average error when K=", K, ": ", avg_scores)

    K = K+1'''
>>>>>>> e3a0970da7f6b704aeaa9c60a4c4c83689da5f7f
