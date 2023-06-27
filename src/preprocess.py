# 1. Data Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import params

def get_data():
    # Get the data for modeling.
    df = pd.read_csv(params.filename_ip_data)    
    X = np.asarray(df[params.X_columns])    
    y = np.asarray(df[params.y_column].astype('int'))
    return [X,y]

def normalize_data(X_fit,X_transform):
    X_transform = StandardScaler().fit(X_fit).transform(X_transform)
    return(X_transform)

def split_data(X, y, test_size = 0.2):
    return train_test_split( X, y, test_size=test_size)
    


