import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold

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
#print("orig unique targets: ", np.unique(train_data['prognosis']))
#print("orig unique targets count: ", len((np.unique(train_data['prognosis']))))
encoder = LabelEncoder()
train_data['prognosis'] = encoder.fit_transform(train_data['prognosis'])
test_data['prognosis'] = encoder.fit_transform(test_data['prognosis'])
#print("unique targets: ", np.unique(train_data['prognosis']))


'''Let us now try some testing on KNN model'''
#Split data
x = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1,
                                                    random_state=1)

#Detect important features
var_thres = VarianceThreshold(threshold=0.1)
var_thres.fit(x_train)
arr = var_thres.get_support()
opt_feat = list(encoder.fit_transform(arr))
print(opt_feat)

#filter out unwanted features by adding good features to new dataframe
print(len(x_train.columns))
i = 0
j = 0
x_train_new = pd.DataFrame()
x_test_new = pd.DataFrame()
while (i < len(opt_feat)):
    if(opt_feat[i] == 1):
        #pd.concat([x_train_new, x_train], axis=1)
        #x_train.drop(x_train.columns[i], axis=1, inplace=True)
        #print(len(x_train.columns))
        x_train_new.insert(j, x_train.columns[i], x_train.loc[:, x_train.columns[i]], False)
        x_test_new.insert(j, x_test.columns[i], x_test.loc[:, x_test.columns[i]], False)
        j = j+1
    i = i+1
print("num_features:", len(x_train_new.columns))
'''
#Create model and train iteratively
neighbor_cnt = 1
neighbor_dict = {}
while (neighbor_cnt < 300):
    model = KNeighborsClassifier(n_neighbors=neighbor_cnt)
    model.fit(x_train_new, y_train)
    preds = model.predict(x_test_new)
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
    optimal_k = 0
    maxK = 0
    while(i <= len(x)):
        if (x[i] > maxK):
            optimal_k = i
            maxK = x[i]
        i = i+1
    return optimal_k

opt_k = filtering(neighbor_dict)
print("num_neighbors:", opt_k)
print("accuracy:", neighbor_dict[opt_k])
'''

'''
The model is tested on other datasets to determine how well it generalizes,
based on its current accuracy to see what neighbor and features counts are best
'''
blocks = 2
test_scores = {}
neighbor_scores = {}
block_scores = {}
while blocks < 5:
    tests = 1
    while tests <= blocks:
        x_train_copy = x_train.copy()
        y_train_copy = y_train.copy()
        x_test_copy = x_test.copy()
        y_test_copy = y_test.copy()
        
        #Create train sets
        x_train_slice = x_train[int(((tests-1)/blocks) * len(x_train)):int((tests/blocks) * len(x_train))]
        y_train_slice = y_train[int(((tests-1)/blocks) * len(y_train)):int((tests/blocks) * len(y_train))]
        x_test_slice = x_test[int(((tests-1)/blocks) * len(x_test)):int((tests/blocks) * len(x_test))]
        y_test_slice = y_test[int(((tests-1)/blocks) * len(y_test)):int((tests/blocks) * len(y_test))]
            
        #Remove train sets from other train sets
        start_i = int(((tests-1)/blocks) * len(x_train))
        end_i = int((tests/blocks) * len(x_train))
        while start_i < end_i:
            x_train_copy.drop([start_i], inplace=True)
            y_train_copy.drop([start_i], inplace=True)
            x_test_copy.drop([start_i], inplace=True)
            y_test_copy.drop([start_i], inplace=True)
                
            start_i = start_i+1
        
        #Train on current set
        neighbor_cnt = 1
        while neighbor_cnt <= 300:
            model = KNeighborsClassifier(n_neighbors=neighbor_cnt)
            model.fit(x_train_slice, y_train_slice)
            preds = model.predict(x_test_slice)
            neighbor_scores[i] = accuracy_score(y_test_slice, preds)
            
            neighbor_cnt = neighbor_cnt+1
            
        tests = tests+1
        
    blocks = blocks+1

'''
splits = 15
K = 2
scores = {}
sub_scores = {}
total_scores = {}
#neighbors = {}
total_neighbors = {}
    
#Iterate through different K splits
while K <= splits:
    #Iterate through different k-th blocks
    k = 1
    while k <= K:
        #Create copies of data to reload at beginning of each iteration
        x_train_copy = x_train.copy()
        y_train_copy = y_train.copy()

        #Create validation sets
        x_val = x_train[int(((k-1)/K) * len(x_train)):int((k/K) * len(x_train))]
        y_val = y_train[int(((k-1)/K) * len(y_train)):int((k/K) * len(y_train))]
            
        #Remove the validation block from the training sets
        start_i = int(((k-1)/K) * len(x_train))
        end_i = int((k/K) * len(x_train))
        while start_i < end_i:
            x_train_copy.drop([start_i], inplace=True)
            y_train_copy.drop([start_i], inplace=True)
                
            start_i = start_i+1
                
        #Begin training
        while (i < 300):
            model = KNeighborsClassifier(n_neighbors=i)
            model.fit(x_train_copy, y_train_copy)
            preds = model.predict(x_val)
            score = (accuracy_score(preds, y_test) * 100)
            sub_scores[i] = score
            i = i+1
        #print("Error score when validation block is", k, ": ", score)
            
        #Store score and current validation block number in dictionary
        scores[k] = sub_scores
        #weights[k] = model.coef_
            
        k = k+1
        
        #Error scores are plotted to visually detect which value of k is optimal
        plt.plot(scores.keys(), scores.values())
        plt.title("Error scores for k-th validation blocks", fontweight="bold")
        plt.xlabel("k-th validation block")
        plt.ylabel("Error score")
        plt.xticks(range(1,(len(scores)+1)))
        plt.gca().invert_xaxis()
        plt.show()
        
        
        Search through the dictionary to find the highest score and its block number,
        along with computing the sum of the scores to help determine the average
        
        
    i = 1
    max_score = scores[1]
    #min_key = 1
    max_neighbors = 1
    avg_scores = 0
    while i < len(scores):
        if (max_score < scores[i]):
            max_score = scores[i]
            #min_key = i
            max_neighbors = i
        avg_scores = avg_scores+scores[i]
        i = i+1
            
    avg_scores = avg_scores/K
        
    total_scores[K] = avg_scores
    total_neighbors[K] = max_neighbors
            
    #Print the optimal validation block number and the average error score
    #print("Optimal validation block: ", min_key)
    #print("Average error when K=", K, ": ", avg_scores)

    K = K+1
'''
'''#Seems 213 might be a good value for number of neighbors; let us test on
#other datasets to see if it generalizes well
splits = 4
k = 1

#copy training data
x_train_copy = x_train.copy()
y_train_copy = y_train.copy()

#store validation block
x_val = x_train_copy[int(((k-1)/splits) * len(x_train_copy)):int((k/splits) * len(x_train_copy))]
y_val = y_train_copy[int(((k-1)/splits) * len(y_train_copy)):int((k/splits) * len(y_train_copy))]'''

'''#FIXME remove valid block from training data
start_i = int(((k-1)/splits) * len(x_train))
end_i = int((k/splits) * len(x_train))
while start_i < end_i:
    x_train_copy.drop([start_i], inplace=True)
    y_train_copy.drop([start_i], inplace=True)
        
    start_i = start_i+1'''