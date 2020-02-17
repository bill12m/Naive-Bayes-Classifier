importm subprocess as sp
import pandas as pd
import numpy as np
sp.call('clear', shell = True)

#Import data#
raw_data = pd.read_csv('raw_data.csv', header = None)
del raw_data[0]
title = raw_data.values

#Create array with the keywords we'll be searching for.#
columns = ['Title','tech','computer','science','program','software',
           'internet','algorithm','data','computation','artificial','processing',
           'cyber','machine','digital','OUTCOME']
#Merge the titles with the empty columns into one data frame#
input_variables = np.zeros(shape = (len(raw_data.index),len(columns) - 1))
input_variables = np.concatenate([title,input_variables],axis = 1)
input_variables = pd.DataFrame(input_variables)
input_variables.columns = columns

input_variables.to_csv('training_data.csv')































