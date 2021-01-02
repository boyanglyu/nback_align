#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np

import os
import cal_dist
import util_nirs
from scipy.io import loadmat
import ot
import multiprocessing
from functools import partial


group_num = '4'
test = '/home/boyang/AF/fnirs/' + group_num
sava_path = '/home/boyang/AF/alignment/new_distance/' + group_num + '/'
cleaned_file_path = '/home/boyang/AF/fnirs_new/'+ group_num +'_cl'
res_path = '/home/boyang/AF/alignment/clean_code/session_result/' + group_num + '/'
prefix = os.listdir(test)
total_files = 4

if '.DS_Store' in prefix:
    prefix.remove('.DS_Store')
remove_list = []

        
prefix = [x for x in prefix if x not in remove_list]
comb = [(0,1),(0,2),(0,3),(1,0),(1,2),(1,3),(2,0),(2,1),(2,3),(3,0),(3,1),(3,2)]
#comb = [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]



fnirs_path = [test+'/'+ele+'/'+ele+'_NIRSdata.csv' for ele in prefix]
time_path = [test+'/'+ele+'/'+ele+'_STIMdata.csv' for ele in prefix]
clean_path = [cleaned_file_path +'/' + ele + '_all_back_cleaned.mat' for ele in prefix]
window_size = 60 # 59 for subject 2



def get_task_data(fnirs_path, time_path, clean_path):
  
    _, crit_time, nirs_data = util_nirs.read_one_piece(fnirs_path, 
                                                          time_path)
    
    all_back = loadmat(clean_path)['to_save']
    print('shape of all back', all_back.shape)
    task_data, label = util_nirs.extract_target_data(nirs_data, all_back, 
                                                     crit_time, window_size)
    return task_data
    

def calculate_cov(window, method, threshold,data):
    dist_mat = cal_dist.distance_cov_mat(data, window, method, threshold)
    return dist_mat


def save_cov_dist(data_1, data_2, data_3, data_4, window_size, method, threshold):
    method = method
    threshold=threshold
    window = window_size
    print('data to distance', data_1.shape)
    iterable = [data_1, data_2, data_3, data_4] 
    pool = multiprocessing.Pool(processes=4)
    func = partial(calculate_cov, window, method, threshold)
    dis_list = pool.map(func, iterable) 
    pool.close()
    pool.join()
    
    dis_mat1 = dis_list[0]
    dis_mat2 = dis_list[1]
    dis_mat3 = dis_list[2]
    dis_mat4 = dis_list[3]

    np.save(sava_path+ 'session_' + prefix[0] + 'win_' + str(window_size) + '_cov_mat.npy', dis_mat1)
    np.save(sava_path+ 'session_' + prefix[1] + 'win_' + str(window_size) + '_cov_mat.npy', dis_mat2)
    np.save(sava_path+ 'session_' + prefix[2] + 'win_' + str(window_size) + '_cov_mat.npy', dis_mat3)
    np.save(sava_path+ 'session_' + prefix[3] + 'win_' + str(window_size) + '_cov_mat.npy', dis_mat4)

def save_mean_dist(data, window_size):
    for idx, ele in enumerate(data):
        mean_mat = cal_dist.distance_mean_mat(ele, window_size)
        np.save(sava_path+ 'session_' + prefix[idx] + 'win_' + str(window_size) + '_mean_mat.npy', mean_mat)
    return

def mapping(dis_mat1, dis_mat2, window_size, method, threshold, eps):

    n1 = len(dis_mat1)
    n2 = len(dis_mat2)
    p = ot.unif(n1)
    q = ot.unif(n2)
    

    gw2, log = ot.gromov.entropic_gromov_wasserstein(
        dis_mat1, dis_mat2, p, q, 'square_loss', epsilon=eps, log=True, verbose=True)
    return log, gw2, dis_mat1, dis_mat2

def mask(couple, n1, n2):
    couple = couple.T
    idx = couple.argmax(axis=1)
#     print(idx)
    mask_mat = np.zeros((n2, n1))
    mask_mat[np.arange(n2),idx] = 1
    return mask_mat

def get_color(fnirs_path, time_path):
    all_back, crit_time, nirs_data = util_nirs.read_one_piece(fnirs_path, 
                                                          time_path)
    task_data, label = util_nirs.extract_target_data(nirs_data, all_back, 
                                                     crit_time, window_size,prev=False)
    color = util_nirs.generate_color(window_size, label)   
    return color

def evalutaion(couple_mat, color_1, color_2):
    correct = 0
    count = {0:0,1:0, 2:0,3:0}
    print('couple_mat shape is ', couple_mat.shape)
    print('color 1 shape is ', color_1.shape)
    print('color 2 shape is ', color_2.shape)
    pred_color = np.matmul(couple_mat,color_1)
    for i in range(len(color_2)):
        if pred_color[i] == color_2[i]:
            correct += 1
            count[color_2[i]] += 1
    rate = round(correct / len(color_2),2)
    for i in range(len(count.keys())):
        count[i] = round(count[i] / (len(color_2)/4) * 100,2)
    return correct, rate, count, pred_color
 

def sess_by_sess(shape_list, comb, eps):
    acc_list = []
    all_pred_list = []
    with open(res_path + 'accuracy_avg_' + group_num + '.txt', 'a+') as f:
        f.write('\n')
        f.write('mean  + cov , eps = '+ str(eps) + ' window is '+ str(window_size) +' cleaned data'+'\n')
    for ele in comb:
        # load distance matrix
        color_1 = get_color(fnirs_path[ele[0]], time_path[ele[0]])
        color_2 = get_color(fnirs_path[ele[1]], time_path[ele[1]])
        dis1_cov_path = sava_path+ 'session_' + prefix[ele[0]] + 'win_' + str(window_size) + '_cov_mat.npy'
        dis2_cov_path = sava_path+ 'session_' + prefix[ele[1]] + 'win_' + str(window_size) + '_cov_mat.npy'
        dis1_mean_path = sava_path+ 'session_' + prefix[ele[0]] + 'win_' + str(window_size) + '_mean_mat.npy'
        dis2_mean_path = sava_path+ 'session_' + prefix[ele[1]] + 'win_' + str(window_size) + '_mean_mat.npy'
        mean_1 = np.load(dis1_mean_path)
        mean_2 = np.load(dis2_mean_path)
        
        dis_mat1 =  np.load(dis1_cov_path) + mean_1 
        dis_mat2 = np.load(dis2_cov_path) + mean_2
        print(dis_mat1.shape)
        print(dis_mat2.shape)
        dis_mat1 /= shape_list[ele[0]]
        dis_mat2 /= shape_list[ele[1]]
        dis_mat1 /= dis_mat1.max()
        dis_mat2 /= dis_mat2.max()

        log, couple_2, dis_mat1, dis_mat2 = mapping(
                dis_mat1, dis_mat2, window_size, 3, 1e-13, eps)

        mask_2 = mask(couple_2, len(dis_mat1), len(dis_mat2))

        accuracy2, rate2, count2, pred_label = evalutaion(mask_2, color_1, color_2)
        all_pred_list.append(pred_label)
        print(accuracy2, rate2, count2)

        acc_list.append(rate2)
        with open(res_path + 'accuracy_avg_' + group_num + '.txt', 'a+') as f:
           
            f.write('session ' + prefix[ele[0]] + ' and ' + prefix[ele[1]]+'\n')
#             f.write(str(accuracy) + ' '+str(rate) + '--')
#             print(count, file=f)

    return acc_list, all_pred_list
            
'''
get distance matrices
'''   

task_list = []
shape_list = []

for i in range(total_files):
    temp = get_task_data(fnirs_path[i], time_path[i], clean_path[i])
    shape_list.append(temp.shape[0])
    task_list.append(temp)
print(shape_list)

# for subjects with 3 sessions
#save_cov_dist(task_list[0], task_list[1], task_list[2],task_list[2],  window_size, 3, 1e-13)
# for subjects with 4 sessions
save_cov_dist(task_list[0], task_list[1], task_list[2], task_list[3], window_size, 3, 1e-14)
save_mean_dist(task_list, window_size)

eps_list = np.append(np.around(np.arange(0.0005,0.001, 0.0001), 4),np.around(np.arange(0.001,0.011, 0.001), 4))

for ele in eps_list:
    acc, all_pred_list = sess_by_sess(shape_list, comb, ele)
    avg_acc = np.average(acc)
    
    with open(res_path + 'accuracy_avg_' + group_num + '.txt', 'a+') as f:
        f.write(str(shape_list)+ '\n')
        f.write('all prediction is ' + str(all_pred_list)+ '\n')
        f.write('average acc is ' + str(avg_acc)+ '\n')
        
        
