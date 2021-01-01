#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 22:48:32 2019

@author: boyanglyu
"""


import numpy as np

import os
import util_nirs
from scipy.io import loadmat
import ot



group_num = '5'
test = '/home/boyang/AF/fnirs/' + group_num
save_path = '/home/boyang/AF/alignment/new_distance/' + group_num + '/' # norm

prefix = os.listdir(test)
   
    
remove_list = []


prefix = [x for x in prefix if x not in remove_list]

fnirs_path = [test+'/'+ele+'/'+ele+'_NIRSdata.csv' for ele in prefix]
time_path = [test+'/'+ele+'/'+ele+'_STIMdata.csv' for ele in prefix]


########
result_path = '/home/boyang/AF/alignment/clean_code/barycenter_folder/'
cleaned_file_path = '/home/boyang/AF/fnirs_new/'+ group_num +'_cl' 
clean_path = [cleaned_file_path +'/' + ele + '_all_back_cleaned.mat' for ele in prefix]
##########

def compute_barycenter(N, label_list, dist_list, alpha, loss_fun):
    ps = [ot.unif(N) for i in range(len(dist_list))]
   
    lam = [1 / len(dist_list) for i in range(len(dist_list))]
  
    X, Cs = ot.gromov.fgw_barycenters(N = N, Ys = label_list, Cs = dist_list,ps= ps, lambdas = lam, alpha = alpha, loss_fun = loss_fun, max_iter=500, tol=1e-3)
    return X, Cs


def subject_barycenter(shape_list, number_sess,save_path, prefix , label, alpha,loss_func):
    dist_list = []
    label_list = []
    for i in range(number_sess):
        # load distance matrix
        dis_cov_path = save_path + 'session_' + prefix[i] + 'win_' + str(window_size) + '_cov_mat.npy'
        dis_mat =  np.load(dis_cov_path) 
        ##########
        dis_mat /= shape_list[i]
        print('shape_list[i] is ', shape_list[i])
        ############
        dis_mat /= dis_mat.max()
        dist_list.append(dis_mat)
        label_list.append(np.asarray(label).reshape((dis_mat.shape[0],1)))
        print(label_list[-1].shape)
        print(dis_mat.shape)  #compute_barycenter(N, label_list, dist_list, alpha, loss_fun)
    X, Cs = compute_barycenter(dis_mat.shape[0], label_list, dist_list, alpha, loss_func)
    return X, Cs
        


def get_color(fnirs_path, time_path):
    all_back, crit_time, nirs_data = util_nirs.read_one_piece(fnirs_path, 
                                                          time_path)
    task_data, label = util_nirs.extract_target_data(nirs_data, all_back, 
                                                     crit_time, window_size,prev=False)
    color = util_nirs.generate_color(window_size, label)   
    return color,task_data.shape[0]

def get_task_data(fnirs_path, time_path, clean_path):
  
    _, crit_time, nirs_data = util_nirs.read_one_piece(fnirs_path, 
                                                          time_path)
    
    all_back = loadmat(clean_path)['to_save']
    print('shape of all back', all_back.shape)
    task_data, label = util_nirs.extract_target_data(nirs_data, all_back, 
                                                     crit_time, window_size)
    color = util_nirs.generate_color(window_size, label) 
    return color,task_data.shape[0]



# sub 1: alpha 0.45 times 3, 
# sub 2: alpha 0.5, times 1.5 
# sub 3: alpha 0.6, times 4
# sub 4: alpha 0.4, times 1
# sbu 5: alpha 0.6, times 5 
# sub 6: alpha 0.45 times 2
window_size = 60
total_files = 4
times = 5
alpha = 0.6
shape_list = []
for i in range(total_files):
    color, shape = get_task_data(fnirs_path[i], time_path[i], clean_path[i])
    shape_list.append(shape)
print(shape_list)      

label = color
label = [ele * times for ele in label]
X, Cs = subject_barycenter(shape_list, total_files ,save_path, prefix ,label, alpha, 'square_loss')  

X = X / times
print(X)

np.save(result_path + 'subject_' + group_num + '_times'+ str(times) + '_barycenter_feature.npy', X)
np.save(result_path + 'subject_' + group_num + '_times'+ str(times) + '_barycenter.npy', Cs)

