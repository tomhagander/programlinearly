import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection

def prepare_data_gaussian():
    num_dtp=1000
    dims=500
    spars_param=0.8

    data = dict()
    feats = np.zeros((num_dtp, dims))
    for i in range(dims):
        dev = np.random.rand()
        x = np.random.uniform(-1,1,size=(num_dtp,)) + np.random.normal(0,dev,(num_dtp,))
        np.random.shuffle(x)
        feats[:,i] = x
    
    theta = 5 * np.random.rand(dims)
    ind = np.random.choice(np.arange(dims), round(dims*spars_param), replace=False)
    theta[ind] = np.random.random() * 0.0001

    bias = 0.3
    y = (feats @ theta).reshape(-1,1) + bias + np.random.normal(0,0.5,(num_dtp,1))

    # randomly split data into 60% train and 40% test set
    trainX, testX, trainYo, testYo = \
      model_selection.train_test_split(feats, y, 
      train_size=0.60, test_size=0.40, random_state=13)

    # normalize feature values
    scaler = preprocessing.StandardScaler()  
    data['trainX'] = scaler.fit_transform(trainX)  
    data['testX']  = scaler.transform(testX)
    
    # map targets to log-space
    data['trainY'] = trainYo.reshape(-1,)
    data['testY']  = testYo.reshape(-1,)

    return data

def prepare_data_news():
    # https://archive.ics.uci.edu/ml/datasets/online+news+popularity
    data = dict()
    
    filename = 'OnlineNewsPopularity/OnlineNewsPopularity.csv'
    
    # read the data
    allfeatnames = []
    textdata      = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(allfeatnames)==0:
                allfeatnames = row
            else:
                textdata.append(row)

    # put the data into a np array
    dataX = np.empty((len(textdata), len(allfeatnames)-3))
    dataY = np.empty(len(textdata))
    for i,row in enumerate(textdata):
        # extract features (remove the first 2 features and the last feature)
        dataX[i,:] = np.array([float(x) for x in row[2:-1]])
        # extract target (last entry)
        dataY[i] = float(row[-1])

    # extract feature names
    data['featnames'] = [x.strip() for x in allfeatnames[2:-1]]

    # extract a subset of data
    dataX = dataX[::6]
    dataY = dataY[::6]
    
    # randomly split data into 60% train and 40% test set
    trainX, testX, trainYo, testYo = \
      model_selection.train_test_split(dataX, dataY, 
      train_size=0.60, test_size=0.40, random_state=4487) # 4487

    # normalize feature values
    scaler = preprocessing.StandardScaler()  
    data['trainX'] = scaler.fit_transform(trainX)  
    data['testX']  = scaler.transform(testX)
    
    # map targets to log-space
    data['trainY'] = np.log10(trainYo)
    data['testY']  = np.log10(testYo)

    return data

def plot_result(result):
    ''' Input Format:
        task 1-2: result = {'taskID':'1-2', 'alpha':[], 'train_err':[], 'test_err':[]}
        task 1-3: result = {'taskID':'1-3', 'feat_num':[], 'train_err':[], 'test_err':[]}
        task 1-4: result = {'taskID':'1-4', 'sample_num':[], 'train_err':[], 'test_err':[]}
        task 1-5: result = {'taskID':'1-5', 'cost':[], 'train_err':[], 'test_err':[]}
        task 2: result = {'taskID':'2', 'cost':[], 'train_err':[], 'test_err':[]}

    '''
    if result['taskID'] == '1-2':
        x_value = result['alpha']
        x_label = 'penalty for sparsity'
        x_range = None
        x_scale = "log"
    
    elif result['taskID'] == '1-3':
        x_value = result['feat_num']
        x_label = 'number of features'
        x_range = (0,1)
        x_scale = "linear"
    
    elif result['taskID'] == '1-4':
        x_value = result['sample_num']
        x_label = 'number of samples'
        x_range = (0,1)
        x_scale = "linear"

    else:  # result['taskID'] == '1-5' or '2'
        x_value = result['cost']
        x_label = 'communication cost during training'
        x_range = (0,1)
        x_scale = "linear"
        
    plt.plot(x_value, result['train_err'], label = 'train_error', marker='x', markersize=8)
    plt.plot(x_value, result['test_err'], label = 'test_error', marker='o', markersize=8)

    plt.xlabel(x_label, fontsize=12) 
    plt.ylabel('MAE', fontsize=12)
    plt.title("Result of Task " + result['taskID'], fontsize=14)
    plt.legend()
    plt.xscale(x_scale)
    plt.xlim(x_range)
    plt.show()