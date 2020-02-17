import subprocess as sp
import pandas as pd
import numpy as np

sp.call('clear', shell = True)

#Column and Row labels for the data frames that will hold the conditional #
#probabilities#
keywords = ['tech','computer','science','program','software',
           'internet','algorithm','data','computation','artificial','processing',
           'cyber','machine','digital']
index = ['X=1|Y=1','X=0|Y=1','X=1|Y=0','X=0|Y=0']

#Import training set and initialize its dataframe of conditional probabilities#
data_train = pd.read_csv('new_training_data.csv')
del data_train['Unnamed: 0']

probabilities = np.zeros(shape = (4,len(data_train.index)-1))
probabilities = pd.DataFrame(probabilities,columns = keywords,
                                   index = index)

#Calculate the frequencies of each OUTCOME#
y_yes = int(data_train['OUTCOME'].agg(['sum']))
y_no = len(data_train['OUTCOME']) - y_yes
prob_y_yes = (y_yes/len(data_train.index))
prob_y_no = (y_no/len(data_train.index))


#Calculate conditional probabilities given OUTCOME is 1#
for column in range(len(probabilities.columns)):
    x_yes = x_no = 0
    for row in data_train.index:
        if data_train.iloc[row,-1] == 1:
            if data_train.iloc[row,column+1] == 1:
                x_yes += 1
            else:
                x_no += 1
    probabilities.iloc[0,column] = (x_yes/y_yes)
    probabilities.iloc[1,column] = (x_no/y_yes)
    
#Calculate conditional probabilities given OUTCOME is 0#
for column in range(len(probabilities.columns)):
    x_yes = x_no = 0
    for row in data_train.index:
        if data_train.iloc[row,-1] == 0:
            if data_train.iloc[row,column+1] == 1:
                x_yes += 1
            else:
                x_no += 1
        probabilities.iloc[2,column] = x_yes/y_no
        probabilities.iloc[3,column] = x_no/y_no
del(column,row,x_yes,x_no,index)

#Import test set and initialize its data frame of conditional probabilities#
data_test = pd.read_csv('test_data.csv')
del data_test['Unnamed: 0']

keywords = ['Probability Yes','Probability No', 'Final Decision']

probabilities_test = np.zeros(shape = (len(data_test.index),3))
probabilities_test = pd.DataFrame(probabilities_test,columns = keywords)

#Calculate probabilities of OUTCOME given the x's and compare to the true#
#probabilities#
#Probability that OUTCOME is 1#
for row in probabilities_test.index:
    conditional_prob = 1
    for column in range(1,14):
        if data_test.iloc[row,column] == 1:
            conditional_prob *= probabilities.iloc[0,column]
        else:
            conditional_prob *= probabilities.iloc[1,column]
    probabilities_test.iloc[row,0] = conditional_prob * prob_y_yes

#Probability that OUTCOME is 0#
for row in probabilities_test.index:
    conditional_prob = 1
    for column in range(1,14):
        if data_test.iloc[row,column] == 1:
            conditional_prob *= probabilities.iloc[2,column]
        else:
            conditional_prob *= probabilities.iloc[3,column]
    probabilities_test.iloc[row,1] = conditional_prob * prob_y_no

#Decide which OUTCOME to choose and calculate the model's accuracy#
percent_accuracy = 0
for row in probabilities_test.index:
    if probabilities_test.iloc[row,0] >= probabilities_test.iloc[row,1]:
        probabilities_test.iloc[row,-1] = 1
    else:
        probabilities_test.iloc[row,-1] = 0
    
    if probabilities_test.iloc[row,-1] == data_test.iloc[row,-1]:
        percent_accuracy += (1/len(probabilities_test.index))

print("Percent Accuracy: ", percent_accuracy)

predicted_outcomes = pd.Series(probabilities_test['Final Decision'])
data_test['Predicted Outcomes'] = predicted_outcomes

