
import numpy as np
import os

import util_nirs
from scipy.io import loadmat
import ot



# sub 1: alpha 0.45 times 3, 
# sub 2: alpha 0.5, times 1.5 
# sub 3: alpha 0.6, times 4
# sub 4: alpha 0.4, times 1
# sbu 5: alpha 0.6, times 5 
# sub 6: alpha 0.45 times 2
# group_num_1  is source group, group_num_2 is the target group
group_num_1 = '6'
group_num_2 = '5'
times = 2

total_files = 4
sess_dist = 'cov'
window_size = 60 # window size is 60 for all subjects except for subject 2

test_2 = '/home/boyang/AF/fnirs/' + group_num_2
res_file = '/home/boyang/AF/alignment/clean_code/subject_result/'
bary_path = '/home/boyang/AF/alignment/clean_code/barycenter_folder/'
save_path_2 = '/home/boyang/AF/alignment/new_distance/' + group_num_2 + '/'
prefix_2 = os.listdir(test_2)

comb = [0,1,2,3] 
eps_list = np.around(np.arange(0.001,0.0031, 0.0001), 4)

if '.DS_Store' in prefix_2:
    prefix_2.remove('.DS_Store')

        
fnirs_path_2 = [test_2+'/'+ele+'/'+ele+'_NIRSdata.csv' for ele in prefix_2]
time_path_2 = [test_2+'/'+ele+'/'+ele+'_STIMdata.csv' for ele in prefix_2]

########
cleaned_file_path = '/home/boyang/AF/fnirs_new/'+ group_num_2 +'_cl' 
clean_path = [cleaned_file_path +'/' + ele + '_all_back_cleaned.mat' for ele in prefix_2]
##########

def mapping (dis_mat1, dis_mat2, window_size, method, threshold, eps):
    n1 = len(dis_mat1)
    n2 = len(dis_mat2)
    p = ot.unif(n1)
    q = ot.unif(n2)

    gw2, log = ot.gromov.entropic_gromov_wasserstein(
        dis_mat1, dis_mat2, p, q, 'square_loss', epsilon=eps, log=True, verbose=True)
    return gw2, gw2, dis_mat1, dis_mat2


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

    pred_color = np.matmul(couple_mat,color_1)
    for i in range(len(color_2)):
        if pred_color[i] == color_2[i]:
            correct += 1
            count[color_2[i]] += 1
    rate = round(correct / len(color_2),2)
    for i in range(len(count.keys())):
        count[i] = round(count[i] / (len(color_2)/4) * 100,2)
    return correct, rate, count, pred_color


def sess_by_sess(shape_list,comb,sess_dist, eps):
    acc_list = []
    all_pred_list = []
    with open(res_file + 'subject_accuracy_' + group_num_1 + '_and_' + group_num_2 + '_new.txt', 'a+') as f:
        f.write('\n')
        f.write(sess_dist + ', eps = ' +str(eps) +'window is '+ str(window_size) +' cleaned data, over channel for each window, hellinger, barycenter new ' +'\n')
    for ele in comb:
        # load distance matrix
        color_1 =  np.load(bary_path +'subject_' + group_num_1 + '_times'+ str(times) +'_barycenter_feature.npy')
        color_1 = color_1.flatten().tolist()
        color_2 = get_color(fnirs_path_2[ele], time_path_2[ele])
        
        dis1_bary_path = bary_path + 'subject_' + group_num_1 + '_times'+ str(times) +'_barycenter.npy'

        dis2_cov_path = save_path_2 + 'session_' + prefix_2[ele] + 'win_' + str(window_size) + '_cov_mat.npy'

        
        dis_mat1 =  np.load(dis1_bary_path) 
        dis_mat2 =  np.load(dis2_cov_path)
        
        print(dis_mat1.shape)
        print(dis_mat2.shape)
        dis_mat1 /= dis_mat1.max()
        ##########
        dis_mat2 /= shape_list[ele]
        ###########
        dis_mat2 /= dis_mat2.max()
        
        couple_1, couple_2, dis_mat1, dis_mat2 = mapping(
                dis_mat1, dis_mat2, window_size, 3, 1e-13, eps)
        

        mask_2 = mask(couple_2, len(dis_mat1), len(dis_mat2))


        accuracy2, rate2, count2, pred_label = evalutaion(mask_2, color_1, color_2)
        print(accuracy2, rate2, count2)
        acc_list.append(rate2)
        all_pred_list.append(pred_label)
#         with open(res_file +'subject_accuracy_'+ group_num_1 + '_and_' + group_num_2 + '_new.txt', 'a+') as f:
            
#             f.write('subject: ' + group_num_1 + ' barycenter and ' + group_num_2 + '\n')
#             f.write('session ' +  prefix_2[ele]+'\n')
#             f.write(str(accuracy2) + ' '+str(rate2) +'--')
#             print(count2, file=f)
    return acc_list, all_pred_list

def get_task_data(fnirs_path, time_path, clean_path):
  
    _, crit_time, nirs_data = util_nirs.read_one_piece(fnirs_path, 
                                                          time_path)
    
    all_back = loadmat(clean_path)['to_save']
    print('shape of all back', all_back.shape)
    task_data, label = util_nirs.extract_target_data(nirs_data, all_back, 
                                                     crit_time, window_size)
    color = util_nirs.generate_color(window_size, label) 
    return color,task_data.shape[0]

            
shape_list = []

for i in range(total_files):
    color, shape = get_task_data(fnirs_path_2[i], time_path_2[i], clean_path[i])
    shape_list.append(shape)
print(shape_list) 

 
all_acc = []
to_save_avg_acc = 0
for ele in eps_list:
    acc, all_pred_list = sess_by_sess(shape_list, comb, sess_dist,ele)
    avg_acc = np.average(acc)
    if to_save_avg_acc < avg_acc:
        to_save_acc = acc
        to_save_pred = all_pred_list
        to_save_avg_acc = np.average(to_save_acc)
        to_save_std = np.std(to_save_acc)
        to_save_eps = ele
    
    
with open(res_file +'subject_accuracy_'+ group_num_1 + '_and_' + group_num_2 + '_new.txt', 'a+') as f:
    f.write('times is ' + str(times)+ '\n')
    f.write('eps is ' + str(to_save_eps) + '\n')
    f.write('acc is ' + str(to_save_acc)+ '\n')
    f.write('average acc is ' + str(to_save_avg_acc)+ '\n')
    f.write('all prediction is ' + str(to_save_pred)+ '\n')






