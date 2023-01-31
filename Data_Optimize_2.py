

import torch
import numpy as np
from copy import deepcopy
import scipy.optimize as sco
import random
import os
import pickle
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def Load_Opdata(method_use_p = 'this_paper'):
    
    Work_Path = "...your path"
    os.chdir(Work_Path)
    File_LoadPath = "...your path"
    This_file = "100_Samples_4info.txt"  
    f = open(File_LoadPath + This_file, 'rb')
    dic = pickle.load(f)
    f.close()
    dic.keys()
  
    list_num_p, images_p, labels_p, IndexesSet_p = dic['list_x'], dic['images_x'], \
                                         dic['labels_x'], dic['IndexesSet_x ']

    return list_num_p, images_p, labels_p, IndexesSet_p


def text_save(content,filename,mode='a'):

    file = open(filename,mode)
    for i in range(len(content)):
        file.write(str(content[i])+'\n')
    file.close()

def initial_PHW():
    P_list = list()
    H_list = list()
    W_list = list()
    for i in range(0,10):
        np.random.seed(1)
        p_temp = np.random.uniform(low=0, high=1.0, size=(220,340,3))
        p_temp = p_temp.astype('float32')
        p_temp = p_temp.round(2)
        P_list.append(p_temp)
        W_list.append(p_temp)
        H_list.append(0.1*p_temp)
    return P_list, H_list, W_list

def get_Lx_op(P_list):
    '''
    :param P: to get L
    :return: L list include 10 L
    '''
    L_list = list()
    Belta_1 = 1.2
    for i in range(0,10):
        p_temp = P_list[i]
        
        l_temp = p_temp[:, :, 0]
        l_temp = l_temp[:, :, np.newaxis]
        for j in range(0,3):
            
            U, sigma, V = np.linalg.svd(p_temp[:, :, j],1,1)
            assert (len(sigma)==220)
            for k in range(0, len(sigma)):
                sigma[k] = np.sign(sigma[k]) * max(0, abs(sigma[k]) * Belta_1)
            sigma = np.concatenate((sigma,np.array([0.0]*120)) )
            sigma = np.diag(sigma)
            sigma = sigma[0:220]
            con_temp = np.dot(U, np.dot(sigma, V))
            con_temp = con_temp[:, :, np.newaxis]
            l_temp = np.concatenate((l_temp, con_temp ), -1)
        l_temp = l_temp[:,:,1:]
        l_temp = l_temp.astype('float32')
        l_temp = l_temp.round(2)
        L_list.append(l_temp )


    return L_list

  
def get_P_op(images_set, list_filenum, L_list, W_list, H_list):
    
    temp_list_filenum = list_filenum
    temp_images_set = images_set
   
    temp_P_list = list()
    P_list_num = list()
    P_list_num.extend([0]*10)
    for i in range(0,10):
        temp_p = np.array([np.zeros((220,340,3))]).reshape(220,340,3)
        temp_P_list.append(temp_p)
    Belta_1 = 100
    img_num = 0
    for f_num in temp_list_filenum:
        for i in range(0, f_num):
            temp_h1, temp_h2, temp_h3 = H_list[i][:,:,0],H_list[i][:,:,1],H_list[i][:,:,2]
            temp_hh1 = np.linalg.inv(np.dot(temp_h1,temp_h1.T)+np.identity(220) )
            temp_hh2 = np.linalg.inv(np.dot(temp_h2, temp_h2.T)+np.identity(220) )
            temp_hh3 = np.linalg.inv(np.dot(temp_h3, temp_h3.T)+np.identity(220) )
            
            temp_x1 = temp_images_set [img_num,:,:,0]
            temp_x2 = temp_images_set [img_num,:,:,1]
            temp_x3 = temp_images_set [img_num,:,:,2]
          
            temp_l1,temp_l2,temp_l3 = L_list[i][:,:,0],L_list[i][:,:,1],L_list[i][:,:,2]
           
            temp_w1, temp_w2, temp_w3 = W_list[i][:, :, 0], W_list[i][:, :, 1], W_list[i][:, :, 2]
            
            temp_p1 = np.dot(temp_hh1, ( np.dot(np.identity(340)[0:220], np.dot(-temp_h1.T, temp_x1))- \
                                         Belta_1 * temp_l1+temp_w1) )
            temp_p2 = np.dot(temp_hh2, (np.dot(np.identity(340)[0:220], np.dot(-temp_h2.T, temp_x2)) - \
                                        Belta_1 * temp_l2 + temp_w2))
            temp_p3 = np.dot(temp_hh3, (np.dot(np.identity(340)[0:220], np.dot(-temp_h3.T, temp_x3)) - \
                                        Belta_1 * temp_l3 + temp_w3))
            temp_P_list[i] = temp_P_list[i] + np.stack([temp_p1, temp_p2, temp_p3], axis=2)
            P_list_num[i] = P_list_num[i] + 1

            img_num = img_num+1
    for i in range(0,10):
        if P_list_num[i]!=0:
            temp_P_list[i] = temp_P_list[i]/P_list_num[i]
        temp_P_list[i] = temp_P_list[i].astype('float32')
        temp_P_list[i] = temp_P_list[i].round(2)
    return temp_P_list

def get_Wx_op(P_list, L_list, W_list):
    Belta_1 = 0
    gamma = 0.3
    temp_pminusl = [i - j for i, j in zip(P_list, L_list)]
    temp_pminusl = [gamma*(1+Belta_1)*i for i in temp_pminusl]
    W_list = [i - j for i, j in zip(W_list, temp_pminusl)]
    for i in range(0,10):
        W_list[i] = W_list[i].astype('float32')
        W_list[i] = W_list[i].round(2)

    return W_list

def get_H_op(images_set, list_filenum, P_list, H_list):
    temp_list_filenum = list_filenum
    temp_images_set = images_set
    
    temp_H_list = list()
    H_list_num = list()
    H_list_num.extend([0] * 10)
    for i in range(0, 10):
        temp_h = np.array([np.zeros((220, 340, 3))]).reshape(220, 340, 3) 
        temp_H_list.append(temp_h)
    Belta = 0.05e-11
    img_num = 0
    for f_num in temp_list_filenum:
      
        for i in range(0, f_num):  
           
            temp_h1, temp_h2, temp_h3 = H_list[i][:, :, 0], H_list[i][:, :, 1], H_list[i][:, :, 2]
           
            temp_x1 = temp_images_set[img_num, :, :, 0]
            temp_x2 = temp_images_set[img_num, :, :, 1]
            temp_x3 = temp_images_set[img_num, :, :, 2]
           
            temp_p1, temp_p2, temp_p3 = P_list[i][:, :, 0], P_list[i][:, :, 1], P_list[i][:, :, 2]
          
            temp_h1 = np.dot(Belta*(np.sum(temp_x1*temp_x1) - np.sum(temp_p1 * temp_h1)), temp_p1)
            temp_h2 = np.dot(Belta*(np.sum(temp_x2*temp_x2) - np.sum(temp_p2 * temp_h2)), temp_p2)
            temp_h3 = np.dot(Belta*(np.sum(temp_x3*temp_x3) - np.sum(temp_p3 * temp_h3)), temp_p3)
            
            temp_H_list[i] = temp_H_list[i] + np.stack([temp_h1, temp_h2, temp_h3], axis=2)
            H_list_num[i] = H_list_num[i] + 1

            img_num = img_num + 1 
    for i in range(0, 10):
        if H_list_num[i] != 0:
            temp_H_list[i] = temp_H_list[i] / H_list_num[i]
        temp_H_list[i] = temp_H_list[i].astype('float32')
        temp_H_list[i] = temp_H_list[i].round(2)
    assert(len(H_list)==len(temp_H_list))
    H_list = [i+j for i, j in zip(H_list, temp_H_list)] 

    return H_list


def initial_QMW():
    Q_list = list()
    M_list = list()
    W_list = list()
    np.random.seed(1)
    for i in range(0,10):
        q_temp = np.random.uniform(low=0, high=1.0, size=(220, 340))
        q_temp = q_temp.astype('float32')
        q_temp = q_temp.round(2)
        Q_list.append(q_temp)
        W_list.append(q_temp)
        M_list.append(0.1*q_temp)
    return Q_list, M_list, W_list

def get_Lh_op(Q_list):
    
    L_list = list()
    Belta_1 = 1.2
    for i in range(0,len(Q_list)):
        q_temp = Q_list[i]
      
        U, sigma, V = np.linalg.svd(q_temp,1,1)
        assert (len(sigma) == 220)
        for k in range(0, len(sigma)):
            sigma[k] = np.sign(sigma[k]) * max(0, abs(sigma[k]) * Belta_1)
        sigma = np.concatenate((sigma,np.array([0.0]*120)))
        sigma = np.diag(sigma)
        sigma = sigma[0:220]
        l_temp = np.dot(U, np.dot(sigma, V))
        l_temp = l_temp.astype('float32')
        l_temp = l_temp.round(2)

        L_list.append(l_temp )

    return L_list

def get_Q_op(M_list, list_filenum, L_list, W_list, H_list):
    AL_H_list = [np.mean(h, axis=2) for h in H_list]
    temp_list_filenum = list_filenum

    
    temp_Q_list = list()
    Q_list_num = list()
    Q_list_num.extend([0]*10)
    for i in range(0,10):
        temp_q = np.array([np.zeros((220,340))]).reshape(220,340)
        temp_Q_list.append(temp_q)
    Belta_1 = 100

    for f_num in temp_list_filenum:
     
        for i in range(0, f_num):
           
            temp_m1 = M_list[0]
            temp_mm1 = np.linalg.inv(np.dot(temp_m1,temp_m1.T)+np.identity(220) )
          
            temp_h1 = AL_H_list[i]
           
            temp_l1 = L_list[i]
           
            temp_w1 = W_list[i]
           
            temp_q1 = np.dot(temp_mm1, ( np.dot(np.identity(340)[0:220], np.dot(-temp_m1.T, temp_h1))- \
                                         Belta_1 * temp_l1+temp_w1) )

            temp_Q_list[i] = temp_Q_list[i] + temp_q1
            Q_list_num[i] = Q_list_num[i] + 1

    for i in range(0,10):
        if Q_list_num[i]!=0:
            temp_Q_list[i] = temp_Q_list[i]/Q_list_num[i]
    temp_Q_list = [i.astype('float32').round(2) for i in temp_Q_list]

    return temp_Q_list

def get_Wh_op(Q_list, L_list, W_list):
   
    Belta_1 = 0
    gamma = 0.3
    temp_qminusl = [i - j for i, j in zip(Q_list, L_list)]
    temp_qminusl = [gamma*(1+Belta_1)*i for i in temp_qminusl]
    W_list = [i - j for i, j in zip(W_list, temp_qminusl)]

    W_list = [i.astype('float32').round(2) for i in W_list]

    return W_list

def get_M_op(H_list, list_filenum, Q_list, M_list):
    
    AL_H_list = [np.mean(h, axis=2) for h in H_list]
    temp_list_filenum = list_filenum
   
    temp_M_list = list()
    M_list_num = list()
    M_list_num.extend([0] * 10)
    for i in range(0, 10):
        temp_m = np.array([np.zeros((220, 340))]).reshape(220, 340) 
        temp_M_list.append(temp_m)
    Belta = 0.05e-11

    for f_num in temp_list_filenum:
      
        for i in range(0, f_num):  
            
            temp_m1 = temp_M_list[i]
           
            temp_h1 = AL_H_list[i]
          
            temp_q1 = Q_list[i]
           
            temp_m1 = np.dot(Belta*(np.sum(temp_h1*temp_h1) - np.sum(temp_q1 * temp_m1)), temp_q1)
            
            temp_M_list[i] = temp_M_list[i] + temp_m1
            M_list_num[i] = M_list_num[i] + 1

    for i in range(0, 10):
        if M_list_num[i] != 0:
            temp_M_list[i] = temp_M_list[i] / M_list_num[i]
    temp_M_list = [i.astype('float32').round(2) for i in temp_M_list]

    assert(len(M_list)==len(temp_M_list))
    M_list = [i+j for i, j in zip(M_list, temp_M_list)] 

    return M_list


def get_alpha(x_series, M_list, list_filenum, list_2_classes):
    
    temp_list_filenum = list_filenum
    temp_sample_labels = list_2_classes
  
    alpha_matrix = np.array(x_series[0:10]).reshape(1,10)
    u_matrix = np.array(x_series[10*1:]).reshape(1,340,10)
    M_list = [np.mean(m, axis=0).reshape(340,1) for m in M_list]

    sum_2_norm = 0  
    label_index =0
    for f_num in temp_list_filenum:
        temp_sum = 0.0
        for i in range(0,f_num):
            temp_matrix = alpha_matrix[:,i] * np.dot(u_matrix[:,:,i], M_list[i])
            temp_sum = temp_sum + np.sum(temp_matrix)
            sum_2_norm = sum_2_norm + (temp_sum-float(list_2_classes[label_index]) )**2   
        label_index = label_index + 1

    return sum_2_norm

def cons_final_1(x):
    sum_total = 0.0
    u_matrix = np.array(x[10:]).reshape(1,340,10)
    for i in range(0, 10):  
           
        temp_u_matrix = np.dot(u_matrix[ :, :, i], u_matrix[ :, :, i].T)-np.identity(340)

        sum_total = sum_total + np.power(np.linalg.norm(temp_u_matrix, ord=2), 1)
    return sum_total

if __name__ == '__main__':
  
    method = 'this_paper'  
    list_num, images, labels, IndexesSet = Load_Opdata(method_use_p=method)
    images_set_train = images  
    list_filenum_train = list_num  
    list_2_classes = labels 
    
    P_list, H_list, Wx_list = initial_PHW()
    for Eq_Loop in range(0, 100000):  
        Lx_list = get_Lx_op(P_list)
        P_list = get_P_op(images_set_train, list_filenum_train, Lx_list, Wx_list, H_list)
        Wx_list = get_Wx_op(P_list, Lx_list, Wx_list)
        H_list = get_H_op(images_set_train, list_filenum_train, P_list, H_list)

    
    Q_list, M_list, Wh_list = initial_QMW()
    for Eq_Loop in range(0, 100000):  
        Lh_list = get_Lh_op(Q_list)
        Q_list =  get_Q_op(M_list, list_filenum_train, Lh_list, Wh_list, H_list)
        Wh_list = get_Wh_op(Q_list, Lh_list, Wh_list)
        M_list =  get_M_op(H_list, list_filenum_train, Q_list, M_list)
   
    cons_f_1 = ({'type': 'eq', 'fun': cons_final_1})
    conf =[cons_f_1]
    x_0 = [random.uniform(0, 2) for _ in range(0, 1*10 + 1 * 340 * 10)]
    sol = sco.minimize(fun = get_alpha, x0 = x_0, args=(M_list, list_filenum_train, list_2_classes), method='SLSQP', constraints=conf, \
                       options={'maxiter':1000,'disp':True, })
   