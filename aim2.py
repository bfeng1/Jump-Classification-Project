# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 13:44:46 2021

@author: bfeng1
"""

import json
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.stats import mode
from scipy.spatial.distance import squareform
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#%%
class jump:
    n = 135
    r = 5
    def __init__(self, name, jump_cycle):
        self.name = name
        self.jump_cycle = jump_cycle
    def csv2df(self):
        csv_file = self.name + '.csv'
        df = pd.read_csv(csv_file)
        # cleaning the dataset(drop the rows with ratio is higher than 2.25)
        df['left angle ratio'] = df['Angle1']/df['Angle3']
        df.drop(df[df['left angle ratio']>2.25].index, inplace = True)
        df.drop(df[df['left angle ratio']<0.75].index, inplace = True)
        df['smoothed'] = savgol_filter(df['left angle ratio'], 25, 2)
        return df
        
    def finetune(self):
        df = jump.csv2df(self)
        jump_cycle = self.jump_cycle
        new_results = []
        for domain in jump_cycle:
            current_list = []
            for inx in domain:
                start = inx - jump.r
                end = inx + jump.r
                temp = df[start:end]
                max_val = temp['left angle ratio'].max()
                ind = temp[temp['left angle ratio'] == max_val].index.values.astype(int)
                try:
                    ind = ind[0]
                except:
                    ind = 0
                current_list.append(ind)
            new_results.append(current_list)
        check = (jump_cycle == new_results)
        if check is False:
            print('old cycle {}: {}'.format(self.name, jump_cycle))
            print('new cycle {}: {}'.format(self.name, new_results))
        elif check is True:
            print('The jump cycle has been finetuned')
        return new_results    
   
    def resample_df(self):
        df_list = []
        jump_cycle = self.jump_cycle
        df = jump.csv2df(self)
        for i in range(len(jump_cycle)):
            temp = df[jump_cycle[i][0]:jump_cycle[i][1]]
            resample_data = resample(temp, n_samples = jump.n, replace = False, random_state = 0).sort_index()
            # resample_data: resampled dataframe
            resample_data = resample_data.reset_index()
            df_list.append(resample_data)
            # create  plots with resampled data
        return df_list
    def vis(self):
        df_list = jump.resample_df(self)
        a = (len(df_list)+1)//2
        b = 2
        plt.figure(figsize = (14,22))
        for i in range(len(df_list)):
            plt.subplot(a,b,i+1)
            plt.title('subplots {}{}{} : cycle {}'.format(a,b,i+1,i+1))
            plt.xlabel('frame number')
            plt.ylabel('Left angle ratio')
            sns.scatterplot(data = df_list[i], x = df_list[i].index, y = 'left angle ratio')
            sns.lineplot(data = df_list[i], x = df_list[i].index, y = 'smoothed')
        print('the process is done for the jump {}'.format(self.name))

          
# #%%
# # create lists to store the names of csv files
# # create jump cycle(manually select range, then autocorrect by algorithm)
# good_jump_cycle = [[154,309],[398,539],[651,786],[825,980],[1018,1158],[1188,1337],[1374,1524],[1555,1698],[1737,1881],[1895,2054]]
# # cycle1: [010262,010456], [010469, 010638], [010655,010821],[010829,010998],[010998,011163],[011168,011331], [011331, 011497],[011497,011659],[011670,011849],[011849,012015]
# inner_jump_cycle=[ [397,562],[562,742],[742,902],[902,1060],[1060,1232],[1232,1398],[1398,1583],[1583,1760]]
# # cycle1: [001550,001700], [001716, 001902], [001930,002095],[002128,002300],[002330,002520],[002540,002709], [002729, 002900],[002916,003078],[003085,03249]
# outer_jump_cycle = [[379,552],[579,767],[767,973],[991,1171],[1171,1351],[1364,1527],[1543,1697]]




#%%



class KnnDtw(object):
    """K-nearest neighbor classifier using dynamic time warping
    as the distance measure between pairs of time series arrays
    
    Arguments
    ---------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for KNN
        
    max_warping_window : int, optional (default = infinity)
        Maximum warping window allowed by the DTW dynamic
        programming function
            
    subsample_step : int, optional (default = 1)
        Step size for the timeseries array. By setting subsample_step = 2,
        the timeseries length will be reduced by 50% because every second
        item is skipped. Implemented by x[:, ::subsample_step]
    """
    
    def __init__(self, n_neighbors=5, max_warping_window=10000, subsample_step=1):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step
    
    def fit(self, x, l):
        """Fit the model using x as training data and l as class labels
        
        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
            Training data set for input into KNN classifer
            
        l : array of shape [n_samples]
            Training labels for input into KNN classifier
        """
        
        self.x = x
        self.l = l
        
    def _dtw_distance(self, ts_a, ts_b, d = lambda x,y: abs(x-y)):
        """Returns the DTW similarity distance between two 2-D
        timeseries numpy arrays.

        Arguments
        ---------
        ts_a, ts_b : array of shape [n_samples, n_timepoints]
            Two arrays containing n_samples of timeseries data
            whose DTW distance between each sample of A and B
            will be compared
        
        d : DistanceMetric object (default = abs(x-y))
            the distance measure used for A_i - B_j in the
            DTW dynamic programming function
        
        Returns
        -------
        DTW distance between A and B
        """

        # Create cost matrix via broadcasting with large int
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = sys.maxsize * np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in range(1, M):
            cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

        for j in range(1, N):
            cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

        # Populate rest of cost matrix within window
        for i in range(1, M):
            for j in range(max(1, i - self.max_warping_window),
                            min(N, i + self.max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

        # Return DTW distance given window 
        return cost[-1, -1]
    
    def _dist_matrix(self, x, y):
        """Computes the M x N distance matrix between the training
        dataset and testing dataset (y) using the DTW distance measure
        
        Arguments
        ---------
        x : array of shape [n_samples, n_timepoints]
        
        y : array of shape [n_samples, n_timepoints]
        
        Returns
        -------
        Distance matrix between each item of x and y with
            shape [training_n_samples, testing_n_samples]
        """
        
        # Compute the distance matrix        
        dm_count = 0
        
        # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
        # when x and y are the same array
        if(np.array_equal(x, y)):
            x_s = np.shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)
        
            
            for i in range(0, x_s[0] - 1):
                for j in range(i + 1, x_s[0]):
                    dm[dm_count] = self._dtw_distance(x[i, ::self.subsample_step],
                                                      y[j, ::self.subsample_step])
                    
                    dm_count += 1
            
            # Convert to squareform
            dm = squareform(dm)
            return dm
        
        # Compute full distance matrix of dtw distnces between x and y
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            dm = np.zeros((x_s[0], y_s[0])) 
            
            for i in range(0, x_s[0]):
                for j in range(0, y_s[0]):
                    dm[i, j] = self._dtw_distance(x[i, ::self.subsample_step],
                                                  y[j, ::self.subsample_step])
                    # Update progress bar
                    dm_count += 1
        
            return dm
        
    def predict(self, x):
        """Predict the class labels or probability estimates for 
        the provided data

        Arguments
        ---------
          x : array of shape [n_samples, n_timepoints]
              Array containing the testing data set to be classified
          
        Returns
        -------
          2 arrays representing:
              (1) the predicted class labels 
              (2) the knn label count probability
        """
        
        dm = self._dist_matrix(x, self.x)
        # Identify the k nearest neighbors
        knn_idx = dm.argsort()[:, :self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.l[knn_idx]
        
        # Model Label
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1]/self.n_neighbors

        return mode_label.ravel(), mode_proba.ravel()
    
#%%
good = ['good1', 'good2','good4','good6']
inner = ['inner1', 'inner2', 'inner3']
outer= ['outer1', 'outer2']
with open('list_info.txt','r') as file:
    input_lines = [line.strip() for line in file]
all_csv = good+inner+outer
info = {}
info['name'] = all_csv
info['cycle'] = input_lines
#%%
# structure dataset for algorithm training
good_dataset = []
inner_dataset = []
outer_dataset = []
n = 135
for i in range(len(all_csv)):
    temp = jump(info['name'][i], json.loads(info['cycle'][i]))
    # temp.finetune()
    # temp.vis(n)
    if i < len(good):
        good_dataset += temp.resample_df()
    elif i < len(good+inner):
        inner_dataset += temp.resample_df()
    else:
        outer_dataset += temp.resample_df()
total_x = good_dataset+inner_dataset+outer_dataset
for i in range(len(total_x)):
    total_x[i]['series_id'] = i
X = pd.concat(total_x)
#%%
# compare time series signal for good jump and bad (inner+outer) jump
# load the label file

y = pd.read_csv('lable.csv')
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y.jump)
y['label'] = encoded_labels
#%%

# create feature column
feature_columns = X.columns.tolist()[2:]
# construct sequence
sequences = []

for series_id, group in X.groupby('series_id'):
    sequence_features = group[feature_columns]
    label = y[y.series_id == series_id].iloc[0].label
    
    sequences.append((sequence_features, label))

def create_data(sequences, test_size = 0.2):
    train_sequences, test_sequences = train_test_split(sequences, test_size = 0.2)        
    train_X = np.empty(shape = (len(train_sequences),135), dtype = 'object')
    train_y = []
    test_X = np.empty(shape = (len(test_sequences),135), dtype = 'object')
    test_y = []
    for i in range(len(train_sequences)):
        temp_x = train_sequences[i][0]['left angle ratio'].to_list()
        train_X[i][:] = temp_x
        train_y.append(train_sequences[i][1])
    for i in range(len(test_sequences)):
        temp_x = test_sequences[i][0]['left angle ratio'].to_list()
        test_X[i][:] = temp_x
        test_y.append(test_sequences[i][1])
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    return train_X, test_X, train_y, test_y

#%%

iten = 20
score = 0
score_list = []
false_negative_rate = []
false_positive_rate = []
for i in range(iten):
    m = KnnDtw(n_neighbors=1, max_warping_window=15)
    train_X, test_X, train_y, test_y = create_data(sequences)
    m.fit(train_X, train_y)
    label, proba = m.predict(test_X)
    temp_score = accuracy_score(label,test_y)
    tn, fp, fn, tp = confusion_matrix(test_y, label).ravel()
    false_positive_rate.append(fp/(fp+tn))
    false_negative_rate.append(fn/(fn + tp))
    score_list.append(temp_score)
    score += temp_score
print('the accuracy of the classifier: {}%'.format(score/iten*100))
print('false positive rate: {}'.format(np.mean(false_positive_rate)))
print('false negative rate: {}'.format(np.mean(false_negative_rate)))
