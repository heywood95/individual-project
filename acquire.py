#!/usr/bin/env python
# coding: utf-8

# In[1]:


### ACQUIRE ###

import pandas as pd
import numpy as np
import os
from env import host, username, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def get_world_govt_data():
    '''
    This function reads the world government data from the databank.worldbank.org
    site into a df.
    '''
    
    # Read in DataFrame from csv.
    df = pd.read_csv('world_govt.csv')
    
    return df

def acquire_world_govt():
    '''
    This function reads in the world government data from the databank.worldbank.org
    site, writes data to a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('world_govt.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('world_govt.csv', index_col=0)
        
    else:

        #creates new csv if one does not already exist
        df = get_world_govt_data()
        df.to_csv('world_govt.csv')

    return df


# In[ ]:




