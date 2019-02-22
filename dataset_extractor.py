#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 18:23:02 2018

@author: shanmukha
"""

import pickle
import numpy as np

def dataset_maker():
        
    ds = pickle.load(open('dataset.p','rb'),encoding='latin1')
    print('Dividing datasets .......')
    print("length of dataset is ",len(ds))
    np.random.shuffle(ds)
    modified_set_len = int(len(ds)*0.85)
    modified_set = ds[:modified_set_len]
    test_set = ds[modified_set_len:]
    test_set_len = len(test_set)
    train_set_len = int(len(modified_set))
    train_set = modified_set[:train_set_len]
    validation_set = modified_set[train_set_len:]
    validation_set_len = len(validation_set)
    print('length of train set:',train_set_len)
    print('length of validation set:',validation_set_len)
    print('length of test set:',test_set_len)
    np.random.shuffle(ds)
    return ds,train_set,validation_set,test_set 

