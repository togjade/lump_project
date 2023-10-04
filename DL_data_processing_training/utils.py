#!/usr/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import os
import glob
import pdb
import scipy.stats as stats
from scipy.signal import spectrogram
import sys
import yaml
import imp
from torch.utils.data import DataLoader
from torchvision import transforms
# from data_loader import lumpDataset, ToTensor
# from utils import *

from models import *

os.chdir('/raid/togzhan_syrymova/lump_project/scripts/spectrogram/')
################################################################
def reshape_data(x):
    x = x.values[:]
    print(np.shape(x))
    x = x.reshape(x.shape[0], 15, x.shape[1]//15)# 15, x.shape[1]//15
    x = np.swapaxes(x, 1, 2)
    return x
################################################################    
def get_labels(x, count):
#     labels = x
#     if count > 0:
    labels = pd.DataFrame()
    for i in range(count):
        labels = pd.concat([labels, x], axis =0)#x.loc[x.index.repeat(count)].reset_index(drop=True)
    return labels.values[:]
################################################################
def extract_slices(x, window_len, stride_len):
    width = x.shape[1]
    x_sliced = x[:, 0:window_len, :]
    count = 1
    index_end = 0
    while  index_end+stride_len <= width:
        # extract input sequences
        index_start = count*stride_len
        index_end = count*stride_len + window_len
        x_temp = x[:, index_start:index_end, :]
        x_sliced = np.vstack((x_sliced, x_temp))
        count += 1   
    print('>> stride count: ', count)
    return x_sliced, count
################################################################
def to_tensor(x, y):
#     x, y = shuffle(x,y)
    if torch.is_tensor(x) == False:
        x = torch.FloatTensor(x.astype('float64'))
    if torch.is_tensor(y) == False:
        y = torch.FloatTensor(y.astype('float64'))
    return x, y
################################################################
def compute_spectogram(x):
    fs = 160
    f, t, data = spectrogram(x[:, :, :], fs, axis = 1)
    return f, t, np.swapaxes(data, 2, 3)
################################################################
def read_data(window_len, stride_len, path, idx, var):
    # go to dir and read data 
    os.chdir(path)
    csv_files_train = glob.glob('df_train' + var + '.csv') # *raw*8
    csv_files_test = glob.glob('test*' + var + '.csv')
    csv_files_dev = glob.glob('df_dev' + var + '.csv')
    print(csv_files_train)
    print(csv_files_test)
    print(csv_files_dev)
    # train data
    df_train_ = pd.read_csv(csv_files_train[0], sep = ',', header = None)
#     df_train_ = pd.DataFrame(df_train_.values[1:,:])
    # test data
    df_test1 = pd.read_csv(csv_files_test[0], sep = ',', header = None)
    df_test2 = pd.read_csv(csv_files_test[1], sep = ',', header = None)
    df_test_ = pd.concat([df_test1, df_test2], axis = 1)
    df_test_ = df_test_.transpose()
    # dev data
    df_dev_ = pd.read_csv(csv_files_dev[0], sep = ',', header = None)
#     df_dev_ = pd.DataFrame(df_dev_.values[1:,:])
    ################################################################
    # Get labels 
#     if var =='*raw*':
    if df_train_.values[1,-1].astype('float64') > 1:
        train_y = np.floor(pd.DataFrame((df_train_.values[:,-1].astype('float64')))/10)
        dev_y = np.floor(pd.DataFrame((df_dev_.values[:,-1].astype('float64')))/10)
    else:
        train_y = pd.DataFrame((df_train_.values[:,-1].astype('float64')))
        dev_y = pd.DataFrame((df_dev_.values[:,-1].astype('float64')))
        
    # Create test labels
    test_y = pd.concat([ pd.DataFrame([0 for i in range(np.shape(df_test1)[1])]), pd.DataFrame([1 for i in range(np.shape(df_test2)[1])])], axis= 0)
    test_y = test_y.reset_index(drop=True)
    ################################################################
    # get data
    df_train = pd.DataFrame(df_train_.values[:,:-1].astype('float64'))
    df_dev = pd.DataFrame(df_dev_.values[:,:-1].astype('float64'))
    df_test = df_test_.astype('float64')
    ################################################################
    # reshape to 3d 
    df_train = reshape_data(df_train)
    df_dev = reshape_data(df_dev)
    df_test = reshape_data(df_test)
    ################################################################
#     # slice the data
    if idx == 1:
        df_train, count = extract_slices(df_train,  window_len, stride_len)
        df_dev, count = extract_slices(df_dev,  window_len, stride_len)
        df_test, count = extract_slices(df_test,  window_len, stride_len)
    else:
        count = 0
    ################################################################
    # compute z-score
    df_train = stats.zscore(df_train, axis = 1)
    df_dev = stats.zscore(df_dev, axis = 1)
    df_test = stats.zscore(df_test, axis = 1)
    plt.plot(df_train[1, :, 3])
    plt.show()
    ################################################################
    # adapt labeling to liced data 
    print(count)
    train_y = get_labels(train_y,  count)
    dev_y = get_labels(dev_y,  count)
    test_y = get_labels(test_y,  count)
    return  df_train, train_y, df_dev, dev_y, df_test, test_y
#     return  df_train_sliced, train_y_sliced, df_dev_sliced, dev_y_sliced, df_test_sliced, test_y_sliced
################################################################
def train_model(train_x, train_y, dev_x, dev_y, test_x, test_y, model, model_save_path, criterion, optimizer, device):
    best_epoch = 0
    dev_acc_max=0
    epoch_max=0
    writer = SummaryWriter(comment='__' + 'Overtesting')
    os.chdir('/raid/togzhan_syrymova/lump_project/scripts/spectrogram/')
    config = yaml.safe_load(open("config.yaml"))
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    for e in range(num_epochs):
        total_loss = 0
        for i in range(0, train_x.shape[0], batch_size):
            if i+batch_size >= train_x.shape[0]:
                x = train_x[i:]
                y = train_y[i:]
            else:
                x = train_x[i:i+batch_size]
                y = train_y[i:i+batch_size]
            y_pred = model(x.to(device))
            loss = criterion(y_pred, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        ################    
        print("Epoch {}/{}, Train Loss: {:.3f}".format(e+1, num_epochs, total_loss))
        writer.add_scalar('total_loss: ', total_loss)
        with open(model_save_path + '_total_loss.txt', "a") as myfile:
            myfile.write(str(total_loss))
            myfile.write("\n")
        ################
        with torch.no_grad():
            model.eval()
            output = model(dev_x.to(device))
            output = (output>0.5).float()
            acc    = accuracy_score(dev_y, output.cpu())
            if acc > dev_acc_max:
                dev_acc_max=acc
                epoch_max=e+1
                torch.save(model, model_save_path + '.pt')
        print("Dev Accuracy: {:.3f}".format(acc))
        writer.add_scalar('Dev Accuracy: ', acc)
        model.train()
    print("##############################################")
    print("Best dev accuracy is {:.3f} at epoch {}".format(dev_acc_max, epoch_max))
    print("Number of class 1 samples: ", (dev_y>0.5).float().sum().item())
    print("##############################################")    
    test_model(model_save_path, device, test_x, test_y, dev_acc_max)
################################################################
def check_nan(x):
    if(np.isnan(x).any()):
        print("contain NaN values")
    else:
        print("does not contain NaN values")
################################################################
def test_model(model_path, device, test_x, test_y, dev_acc_max):
    if dev_acc_max > 0.5:
        with torch.no_grad():
        #             pdb.set_trace()
            model = torch.load(model_path+'.pt')
            model.eval().to(device)
            output = model(test_x.to(device))
            output = (output>0.5).float()
            acc    = accuracy_score(test_y, output.cpu())
            print("Test Accuracy: {:.3f}".format(acc))
            print(model)
            
################ 
####        ####
################
# def main(model_type):
#     var = '_L_h1'#'_L_h1'
#     print(var)
#     os.chdir('/raid/togzhan_syrymova/lump_project/scripts/spectrogram/')
#     config = yaml.safe_load(open("config.yaml"))
#     print(config)
#     num_epochs = config['num_epochs']
#     batch_size = config['batch_size']
#     learning_rate = config['learning_rate']
#     channels_in = config['channels_in']
#     D_out = config['D_out']
#     H = config['hidden_size']
#     kernel_size = config['kernel_size']
#     stride_len = config['stride']
#     dropout = config['dropout'] 
#     ##
#     window_len = config['window_len']
#     stride_len = config['stride_len']
#     path = config['path']
#     ################################################################
#     # GET DATA #
#     idx = 1
#     train_x_, train_y_, dev_x_, dev_y_, test_x_, test_y_ = read_data(window_len, stride_len, path, idx, var)
#     print('Dataset:', train_x_.shape, train_y_.shape, dev_x_.shape, dev_y_.shape, test_x_.shape, test_y_.shape)
#     ################################################################        
#     print('train_x ')        
#     check_nan(train_x_)  
#     print('dev_x ')        
#     check_nan(dev_x_)  
#     print('test_x ')        
#     check_nan(test_x_)   
#     ################################################################
# #     compute spectograms #
# #     f, t, train_x1 = compute_spectogram(train_x_)
# #     f, t, dev_x1 = compute_spectogram(dev_x_)
# #     f, t, test_x1 = compute_spectogram(test_x_)
#     idx = 1
#     train_x, train_y  = to_tensor(train_x_, train_y_, idx)
# #     l = np.shape(train_x)
# #     train_x = np.reshape(train_x, [l[0], l[1]*l[2], l[3]])
#     dev_x, dev_y = to_tensor(dev_x_, dev_y_, idx)
# #     l = np.shape(dev_x)
# #     dev_x = np.reshape(dev_x, [l[0], l[1]*l[2], l[3]])
#     test_x, test_y = to_tensor(test_x_, test_y_, idx)
    
# #     l = np.shape(test_x)
# #     test_x = np.reshape(test_x, [l[0], l[1]*l[2], l[3]])

#     train_x = np.swapaxes(train_x, 1, 2)
#     dev_x = np.swapaxes(dev_x, 1, 2)
#     test_x = np.swapaxes(test_x, 1, 2)
#     print(np.shape(train_x))
    
#     ###############################################################
# #     train_x, train_y  = to_tensor(train_x, train_y_, idx)
# #     dev_x, dev_y  = to_tensor(dev_x, dev_y_, idx)
# #     test_x, test_y  = to_tensor(test_x, test_y_, idx)
    
#     D_in = np.shape(train_x)[2]
#     print(D_in)
#     print('Started training')
#     seed = 777
#     torch.manual_seed(seed)
#     gpu_id = 1
#     device = torch.device("cuda:" + str(gpu_id))
#     ################################################################
#     # ADAPT DIRECTORY AND MODEL NAMING
#     # ADD MODEL TO README OF THE FOLDER
#     directory = '/raid/togzhan_syrymova/lump_project/models/'
#     os.chdir(directory)
#     # DEFIEN THE MODEL
# #     model = ConvLayerNet(batch_size, device, channels_in, D_out, H).to(device)
#     model = ConvLayerNet3(batch_size, device, channels_in, D_in, H).to(device)
#     model.to(device)
#     model_name = type(model).__name__
#     ################################################################
#     pytorch_total_params = sum(p.numel() for p in model.parameters())
#     print('# of params: ', pytorch_total_params)
#     ################################################################  
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = torch.nn.BCEWithLogitsLoss()
#     ################################################################  
#     # CREATE DIR TO SAVE MODEL AND OTHEHR STUFF
#     if os.path.isdir(model_name)==False:
#         os.mkdir(model_name)
#     opt = str(optimizer).split("(")[0]
#     model_name2 = model_name + '_epochs_' + str(num_epochs) + '_lr_' + str(learning_rate) + "_bs_" + str(batch_size) \
#     + '_window_len_' + str(window_len) + '_stride_len_' + str(stride_len) + '_'+ str(opt).split(" ")[0] + '_' + str(criterion).split("(")[0]
    
#     model_save_path = directory + model_name + '/' + model_name2
#     ################################################################  
#     # SAVE THE MODEL AND README
#     with open(directory + model_name + '/' + 'README_' + model_name2 + '.txt', "a") as myfile:
#             myfile.write(str(model))
#             myfile.write("\n OPTIMIZE: ")
#             myfile.write(str(optimizer))
#             myfile.write("\n CRITERION: ")
#             myfile.write(str(criterion))
#             myfile.write("\n")
#             [myfile.write(str(items) + '\n') for items in str(config).split(", ")]
#             myfile.write("\n")
            
#     print(model_name)
#     print(model_save_path)
#     print(model)
#     train_model(train_x, train_y, dev_x, dev_y, model, model_save_path, criterion, optimizer, device)
#     print('Start testing')
#     test_model(model_save_path, device, test_x, test_y)
    
#     exit()
################
####        ####
################

# if __name__ == '__main__':
#     model_type = 'Conv'
#     main(model_type)
    