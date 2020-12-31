
import pandas as pd

import numpy as np
from scipy import signal


# get start and end time points for different tasks
def time_point(time_file):
    stim = pd.read_csv(time_file)
    task_time = stim.iloc[:, 0]
    time_point = []
    for i in range(4):
        time_point.append(task_time[i * 100])
        if i > 0:
            time_point.append(task_time[i * 100 -1])

    time_point.append(task_time[len(task_time)-1])
    time_point.sort() # time_point contains time points when experiments start and end
    return time_point

# remove data before and after tasks 
def remove_pre_post(full_data, time_point):
    # full_data: raw experiment data, should be pandas format
    # return pandas format data, remove data before 0 back and after 3 back
    begin = time_point[0]
    end = time_point[-1]
    time_stamp = full_data.iloc[:, 0]
    cut_idx = []
    for idx, val in enumerate(time_stamp):
        if val >= begin and val <= end:
            cut_idx.append(idx)
    print(cut_idx.shape)
    #exp_data = full_data.iloc[cut_idx[0]:cut_idx[-1]]
    #print(exp_data)
    return cut_idx



# low pass filter
def low_pass_filter(fs, fc_low, data):
    w_low = fc_low / (fs / 2) 
    d, c = signal.butter(5, w_low, 'low')
    #filtered_data = signal.filtfilt(d, c, data.as_matrix().T)
    filtered_data = signal.filtfilt(d, c, data)
    return filtered_data


# read data from .csv file and process data
def read_one_piece(data_file, time_file, low_pass=False):
    NIRS_data = pd.read_csv(data_file)
    cri_point = time_point(time_file)
    if low_pass:
        clean_data = low_pass_filter(fs=12, fc_low=0.3, data=NIRS_data.iloc[:,1:])
    else:
        clean_data = NIRS_data.iloc[:,1:].as_matrix().T
    print('data shape ',clean_data.shape)
    
    return clean_data, cri_point, NIRS_data

# extract task period
def extract_task_idx(full_data, time_point, task_name = '0'):
    # full_data: raw experiment data, should be pandas format
    # return pandas format data, extract target task data
    assert int(task_name) < 4
    piece_num = int(task_name)
    begin = time_point[piece_num * 2]
    end = time_point[piece_num * 2 + 1]
    #print(begin, end)
    time_stamp = full_data.iloc[:, 0]
    cut_idx = []
    for idx, val in enumerate(time_stamp):
        if val >= begin and val <= end:
            cut_idx.append(idx)
    return np.asarray(cut_idx)

# extract resting period
def extract_rest_idx(full_data, time_point, rest_name = '0'):
    assert int(rest_name) < 5
    piece_num = int(rest_name)
    time_stamp = full_data.iloc[:, 0]
    if piece_num == 0:
        begin = 0
        end = time_point[0]
    elif piece_num == 4:
        begin = time_point[-1]
        end = 1000000
    else:
        begin = time_point[piece_num * 2 - 1]
        end = time_point[piece_num * 2 ]

    cut_idx = []
    for idx, val in enumerate(time_stamp):
        if val >= begin and val <= end:
            cut_idx.append(idx)
    return np.asarray(cut_idx)

def extract_target_data(full_data, pure_data, time_point, window, low_pass = False, prev=False):
    # extract 4 tasks and previous part and make them divisable by window size
    print('pure data shape', pure_data.shape[0])
    res = np.empty((pure_data.shape[0], 0))
    label = []
    if prev:
        prev_idx = extract_rest_idx(full_data, time_point, rest_name='0')
        modulo = np.mod(len(prev_idx), window)
        prev_idx = prev_idx[modulo:]
        print(pure_data.shape, prev_idx.shape)
        res = np.append(res, pure_data[:, prev_idx] , axis=1)
        label.append(len(prev_idx))
    if low_pass:
        pure_data = low_pass_filter(10, 0.2, pure_data)
    for i in range(4):
        task_idx = extract_task_idx(full_data, time_point, task_name = str(i))
        print('length of task idex is ',len(task_idx))
        modulo = np.mod(len(task_idx), window)
        task_idx = task_idx[modulo:]
        print(res.shape)
        temp = pure_data[:, task_idx]
        res = np.append(res, temp,axis=1)
        label.append(len(task_idx))
    return res, label
        
# generate label     
def generate_color(window, label):
    color = np.empty(0)
    for idx, ele in enumerate(label):
        color = np.append(color, np.ones(int(ele)//window) * idx)     
    return color








