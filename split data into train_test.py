import subprocess as sp
import pandas as pd

sp.call('clear', shell = True)

data_train = pd.read_csv('training_data.csv')
del data_train['Unnamed: 0']
keywords = ['tech','computer','science','program','software',
           'internet','algorithm','data','computation','artificial','processing',
           'cyber','machine','digital']

data_train = data_train.sample(frac = 1).reset_index(drop = True)    
data_train['Title'] = data_train.Title.astype(str)

###Case 1###
for keyword in keywords:
    for title in data_train.index:
        my_string = data_train.loc[title,'Title'].lower()
        if my_string.find(keyword) != -1:
            data_train.loc[title,keyword] += 1

data_test = data_train.loc[0:4,:]
data_train = data_train.loc[5:,:].reset_index(drop = True)

data_train.to_csv('new_training_data.csv')
data_test.to_csv('test_data.csv')

###Case 2###
for keyword in keywords:
    for title in data_train.index:
        my_string = data_train.loc[title,'Title'].lower()
        data_train.loc[title,keyword] = my_string.count(keyword)

data_test = data_train.loc[0:4,:]
data_train = data_train.loc[5:,:].reset_index(drop = True)

data_train.to_csv('new_training_data.csv')
data_test.to_csv('test_data.csv')



































