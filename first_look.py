import pandas as pd
import numpy as np
import pickle
import time

from sklearn import metrics, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedKFold

import matplotlib.pyplot as plt

def z_score_dataframe(data_frame):
    z_scored_data = (data_frame - data_frame.mean())/data_frame.std()
    return(z_scored_data)





data = pd.read_csv('./abalone-data/abalone.data', header=None, names=['sex', 'length', 'diameter', 'height', 'whole weight', 'shucked weight', 'viscera weight', 'shell weight', 'rings'])

meta_dataset = data['sex']

zscore = z_score_dataframe(data.drop(['sex'], axis=1))







