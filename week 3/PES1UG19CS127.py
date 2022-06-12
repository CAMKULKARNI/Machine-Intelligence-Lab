'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import random


'''Calculate the entropy of the enitre dataset'''
# input:pandas_dataframe
# output:int/float


def get_entropy_of_dataset(df):
    entropy = 0
    results = df.iloc[:, -1]
    n = 0
    counts = {}
    for i in results:
        if i in counts:
            counts[i] += 1
        else:
            counts[i] = 1
        n += 1

    for i in counts:
        entropy -= (counts[i] / n) * (np.log(counts[i] / n) / np.log(2))

    return entropy


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float


def get_avg_info_of_attribute(df, attribute):
    groups = df.groupby(by=attribute)
    n = len(df)

    values = {}
    for key, value in groups:
        values[key] = np.array((value.iloc[:, -1].value_counts()))

    avg_info = 0
    for i in values:
        x = sum(values[i])
        avg_info -= (x / n) * sum((values[i] / x)
                                  * ((np.log(values[i] / x)) / np.log(2)))
    return avg_info


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float


def get_information_gain(df, attribute):
    information_gain = get_entropy_of_dataset(
        df) - get_avg_info_of_attribute(df, attribute)
    return information_gain


#input: pandas_dataframe
#output: ({dict},'str')

def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    values = {}
    for i in df.columns[:-1]:
        values[i] = get_information_gain(df, i)

    return (values, max(values, key=lambda x: values[x]))
