#!/usr/bin/python
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import glob
import sys
import imp
train_dev_without = pd.DataFrame() 
train_dev_with = pd.DataFrame() 
test_without = pd.DataFrame() 
test_with = pd.DataFrame()

def separate_subjects(test_subject_1, test_subject_2):
    directory = '/raid/togzhan_syrymova/lump_project/data/'
    os.chdir(directory)
    listdir_all = glob.glob('all*raw*')
    listdir_test = glob.glob('test*raw*')
    print(listdir_all)

    all_data_with_raw = pd.read_csv("all_subject_pressure_data_without_raw.csv", sep = ',', header = None)
    all_data_without_raw = pd.read_csv("all_subject_pressure_data_with_raw.csv", sep = ',', header = None)
    data = 32*9
    train_dev_without = pd.concat([ \
             pd.DataFrame(all_data_without_raw.values[:, 0:test_subject_1*data]), \
             pd.DataFrame(all_data_without_raw.values[:, test_subject_1*data:(test_subject_2 - 1)*data]),
             pd.DataFrame(all_data_without_raw.values[:, (test_subject_2+1)*data:])
                                  ], axis = 1)
    train_dev_with = pd.concat([ \
             pd.DataFrame(all_data_with_raw.values[:, 0:test_subject_1*data]), \
             pd.DataFrame(all_data_with_raw.values[:, test_subject_1*data:(test_subject_2 - 1)*data]),
             pd.DataFrame(all_data_with_raw.values[:, (test_subject_2+1)*data:])
                               ], axis = 1)
    # test sets
    test_without = pd.concat([ \
             pd.DataFrame(all_data_without_raw.values[:, test_subject_1*data:(test_subject_1+1)*data]), \
             pd.DataFrame(all_data_without_raw.values[:, test_subject_2*data:(test_subject_2+1)*data]),
                               ], axis = 1)

    test_with = pd.concat([ \
             pd.DataFrame(all_data_with_raw.values[:, test_subject_1*data:(test_subject_1+1)*data]), \
             pd.DataFrame(all_data_with_raw.values[:, test_subject_2*data:(test_subject_2+1)*data]),
                               ], axis = 1)
    print(np.shape(train_dev_without))
    return train_dev_without, train_dev_with, test_without, test_with
    
################################################################    
def split_dev_train(train_dev_without, train_dev_with, test_subject_1, test_subject_2):
    train_dev_without, train_dev_with, test_without, test_with = separate_subjects(test_subject_1, test_subject_2)
    var = '_s_' + str(test_subject_1) + '_s_' + str(test_subject_2)
    train_dev_without.columns = [''] * len(train_dev_without.columns)
    train_dev_with.columns = [''] * len(train_dev_with.columns)
    ##### LABELS
    label_with = []
    label_without = []
    labels_with_ = pd.DataFrame()
    labels_without_ = pd.DataFrame()
    a = pd.DataFrame([1])
    b = pd.DataFrame([0])
    l = np.shape(train_dev_without)[1]
    label_with = a.values[:].repeat(l)
    label_without = b.values[:].repeat(l)
    print(np.shape(label_with))
    for i in range(9):
        labels_with_ = pd.concat([labels_with_,  pd.DataFrame([np.zeros(32) + i])]) # 9 prot from 0 to 32
        labels_without_ = pd.concat([labels_without_,  pd.DataFrame([np.zeros(32) + i])])
        
    labels_with_ = pd.DataFrame(labels_with_.values[:].repeat(8))
    label_with = pd.DataFrame(label_with)
    label_without = pd.DataFrame(label_without)
    labels_with__ = (label_with.values[:])*10 + (labels_with_.values[:])

    # create labels without like 01
    labels_without_ = pd.DataFrame(labels_without_.values[:].repeat(8))
    labels_without__ = (label_without.values[:])*10 + (labels_without_.values[:]) 
    # concat with and without 
    bin_labels = pd.concat([pd.DataFrame(labels_without__), pd.DataFrame(labels_with__)], axis = 0)
    bin_labels = bin_labels.reset_index(drop=True)

    ##### 
    df_d = pd.concat([pd.DataFrame(train_dev_without), pd.DataFrame(train_dev_with)], axis = 1)
    df_data_t = df_d.transpose()
    df_data_t = df_data_t.reset_index(drop=True)
    df_data = pd.concat([df_data_t, bin_labels], axis = 1)
    
    df_train, df_eval = train_test_split(df_data, test_size = 1.0/3, random_state=777, stratify = bin_labels)
    print("Saving \n")
    df_train.to_csv(directory + "df_train" + var + ".csv", index=False, header = False) # , header = False
    print("train saved \n")
    df_eval.to_csv(directory + "df_dev" + var + ".csv", index=False, header = False)
    print("deV saved \n")
    test_with.to_csv(directory + "test_pressure_data_with" + var + ".csv", index=False, header = False)
    print("test with saved \n")
    test_without.to_csv(directory + "test_pressure_data_without" + var + ".csv", index=False, header = False)
    print("test without saved \n")
    return df_train, df_eval, test_with, test_without
################################################################

if __name__ == '__main__':
    directory = '/raid/togzhan_syrymova/lump_project/data/'
    os.chdir(directory)
    while(True):
        try:
            print("Please enter the INDEX of the subjects that you want to put into test set (INDEX 0 - 9):  \n")
            print("Please enter -1 to exit \n")
            test_subject_1 = int(input( "INDEX of the 1st subject(index should be less than 2nd subject's):   "))
            if ((test_subject_1 == -1) or (test_subject_1 > 9)):
                print("Please try again \n")
                break
            test_subject_2 = int(input( "INDEX of the 2nd subject:  "))
            if ((test_subject_2 == -1) or (test_subject_2 > 9) or (test_subject_1 > test_subject_2)):
                print("Please try again \n")
                break
                
        except ValueError:
            print("Try again!")
            break
        train_dev_without, train_dev_with, test_without, test_with = split_dev_train(train_dev_without, train_dev_with, test_subject_1, test_subject_2)
        break
   