
import torch
import numpy as np
import yaml
from copy import deepcopy
import scipy.optimize as sco
import random
import os
import pickle
import os
from scipy import linalg
import platform
from itertools import product
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


print("code begin !")
def Load_data(method_use_p = 'this_paper',data_to_use = 'YHD_Hos'):
   
    if platform.system().lower() == 'windows':
        File_LoadPath = '....................your path' 
    else:
        File_LoadPath =".........your path"    
    if data_to_use == 'YHD_Hos':
        This_file = "100_Samples_4info.txt"  
    elif data_to_use == 'QILU_Hos':
        This_file = "medicalyuan_100sample.txt"  
    elif data_to_use == 'race_and_riding':
        This_file = "race_and_riding.txt"  
    elif data_to_use == 'Handstand':
        This_file = "pu_and_wa.txt"  

    f = open(File_LoadPath + This_file, 'rb')
    dic = pickle.load(f)
    f.close()
    dic.keys()
    

    
    if method_use_p == 'this_paper':
        file_name = File_LoadPath + 'this_paper.yaml'
    elif method_use_p == 'pcf':
        file_name = File_LoadPath + 'pcf.yaml'
    with open(file_name, 'r', encoding='utf8') as file:  
        parm_yaml_p = yaml.safe_load(file)
    
    if data_to_use == 'YHD_Hos':
        yaml_index = 0 
    elif data_to_use == 'QILU_Hos':
        yaml_index = 1  
    elif data_to_use == 'race_and_riding':
        yaml_index = 2  
    elif data_to_use == 'Handstand':
        yaml_index = 3  
    else:  
        pass
    for each_key in parm_yaml_p.keys():
        parm_yaml_p[each_key] = parm_yaml_p[each_key][yaml_index]

   

    if data_to_use == 'YHD_Hos':
        list_num_p, images_p, labels_p, IndexesSet_p = dic['list_x'], dic['images_x'], \
                                         dic['labels_x'], dic['IndexesSet_x ']
    if data_to_use == 'QILU_Hos':
        list_num_p, images_p, labels_p = dic['list_x'], dic['images_x'], dic['labels']
        IndexesSet_p = np.zeros(shape=(len(dic['list_x']), 7), dtype=np.float64)  
    if data_to_use == 'race_and_riding':
        list_num_p, images_p, labels_p = dic['list_x'], dic['images_x'], dic['labels']
        IndexesSet_p = np.zeros(shape=(len(dic['list_x']), 7), dtype=np.float64)  
    if data_to_use == 'Handstand':
        list_num_p, images_p, labels_p = dic['list_x'], dic['images_x'], dic['labels']
        IndexesSet_p = np.zeros(shape=(len(dic['list_x']), 7), dtype=np.float64)  

    return parm_yaml_p, list_num_p, images_p, labels_p, IndexesSet_p


def text_save(content,filename, mode='a'):
    
    file = open(filename, mode)

    for i in range(len(content)):
        if filename == "Iteration.txt":
            file.write(str(content[i])+" ")
        else:
            file.write(str(content[i])+'\n')
    file.write('\n')
    file.close()
def Del_na(Origin_array):
    Origin_array[np.isnan(Origin_array)] = 0
    return Origin_array


def initial_PHW(yaml_p):
    P_list = list()
    H_list = list()
    W_list = list()
    for i in range(0, yaml_p["image_maxlen"]):
        np.random.seed(1)
        p_temp = np.random.uniform(low=0, high=1.0, size=(yaml_p["image_height"],yaml_p["image_width"],3))
        p_temp = p_temp.astype('float32')
        p_temp = p_temp.round(2)
        P_list.append(p_temp)
        W_list.append(p_temp)
        H_list.append(0.1*p_temp)
    return P_list, H_list, W_list

def get_Lx_op(P_list, yaml_p):
    '''
    :param P: to get L
    :return: L list include 10 L
    '''
    L_list = list()
    Belta_1 = 1.2
    for i in range(0, yaml_p["image_maxlen"]):
        p_temp = P_list[i]
 
        l_temp = p_temp[:, :, 0]
        l_temp = l_temp[:, :, np.newaxis]
        for j in range(0,3):
            
            p_temp[:, :, j][np.isinf(p_temp[:, :, j])] = 0
            p_temp[:, :, j][np.isnan(p_temp[:, :, j])] = 0
            U, sigma, V = np.linalg.svd(p_temp[:, :, j],1,1)
            U[np.isinf(U)] = 0
            V[np.isinf(V)] = 0
            assert (len(sigma)==yaml_p["image_height"])
            for k in range(0, len(sigma)):
                sigma[k] = np.sign(sigma[k]) * max(0, abs(sigma[k]) * Belta_1) 
            sigma = np.concatenate((sigma,np.array([0.0]*400)) )
            sigma = np.diag(sigma)
            sigma = sigma[0:yaml_p["image_height"], 0:yaml_p["image_width"]]
            con_temp = np.dot(U, np.dot(sigma, V)) 
            con_temp = con_temp[:, :, np.newaxis]
            l_temp = np.concatenate((l_temp, con_temp ), -1)
        l_temp = l_temp[:,:,1:]
        l_temp = l_temp.astype('float32')
        l_temp = l_temp.round(2)
        L_list.append(l_temp )


    L_list = [Del_na(i) for i in L_list]
    L_list = [i * 100 / np.mean(i) for i in L_list]
    L_list = [np.maximum(i, -i) for i in L_list]

    return L_list

def get_P_op(images_set, list_filenum, L_list, W_list, H_list,yaml_p):
    '''
    :param images_set:  x
    :param list_filenum:  number of views in each sample
    :param L_list: lagrange
    :param W_list: lagrange
    :param H_list: reconstruction of x
    :return:
    '''
    
    temp_list_filenum = list_filenum
    temp_images_set = images_set
    temp_P_list = list()
    P_list_num = list()
    P_list_num.extend([0]*yaml_p["image_maxlen"])
    for i in range(0,yaml_p["image_maxlen"]):
        temp_p = np.array([np.zeros((yaml_p["image_height"],yaml_p["image_width"],3))]).reshape\
            (yaml_p["image_height"],yaml_p["image_width"],3)
        temp_P_list.append(temp_p)
    Belta_1 = 100
    img_num = 0
    for f_num in temp_list_filenum:
        for i in range(0, f_num):
            
            temp_h1, temp_h2, temp_h3 = H_list[i][:,:,0],H_list[i][:,:,1],H_list[i][:,:,2]
            temp_hh1 = np.linalg.inv(np.dot(temp_h1,temp_h1.T)+np.identity(yaml_p["image_height"]) )
            temp_hh2 = np.linalg.inv(np.dot(temp_h2, temp_h2.T)+np.identity(yaml_p["image_height"]) )
            temp_hh3 = np.linalg.inv(np.dot(temp_h3, temp_h3.T)+np.identity(yaml_p["image_height"]) )
            temp_x1 = temp_images_set [img_num,:,:,0]
            temp_x2 = temp_images_set [img_num,:,:,1]
            temp_x3 = temp_images_set [img_num,:,:,2]
          
            temp_l1,temp_l2,temp_l3 = L_list[i][:,:,0],L_list[i][:,:,1],L_list[i][:,:,2]
            temp_w1, temp_w2, temp_w3 = W_list[i][:, :, 0], W_list[i][:, :, 1], W_list[i][:, :, 2]
            temp_p1 = np.dot(temp_hh1, ( np.dot(np.identity(yaml_p["image_width"])[0:yaml_p["image_height"]], np.dot(-temp_h1.T, temp_x1))- \
                                         Belta_1 * temp_l1+temp_w1) )
            temp_p2 = np.dot(temp_hh2, (np.dot(np.identity(yaml_p["image_width"])[0:yaml_p["image_height"]], np.dot(-temp_h2.T, temp_x2)) - \
                                        Belta_1 * temp_l2 + temp_w2))
            temp_p3 = np.dot(temp_hh3, (np.dot(np.identity(yaml_p["image_width"])[0:yaml_p["image_height"]], np.dot(-temp_h3.T, temp_x3)) - \
                                        Belta_1 * temp_l3 + temp_w3))
            temp_P_list[i] = temp_P_list[i] + np.stack([temp_p1, temp_p2, temp_p3], axis=2)
            P_list_num[i] = P_list_num[i] + 1

            img_num = img_num+1
    for i in range(0,yaml_p["image_maxlen"]):
        if P_list_num[i]!=0:
            temp_P_list[i] = temp_P_list[i]/P_list_num[i]
        temp_P_list[i] = temp_P_list[i].astype('float32')
        temp_P_list[i] = temp_P_list[i].round(2)

    temp_P_list = [Del_na(i) for i in temp_P_list]
    temp_P_list = [i * 100 / np.mean(i) for i in temp_P_list]
    temp_P_list = [np.maximum(i, -i) for i in temp_P_list]

    return temp_P_list
def get_Wx_op(P_list, L_list, W_list,yaml_p):
    '''
    :param P_list:
    :param L_list:
    :return:
    '''
    Belta_1 = 0
    gamma = 0.3
    temp_pminusl = [i - j for i, j in zip(P_list, L_list)]
    temp_pminusl = [gamma*(1+Belta_1)*i for i in temp_pminusl]
    W_list = [i - j for i, j in zip(W_list, temp_pminusl)]
    for i in range(0, yaml_p["image_maxlen"]):
        W_list[i] = W_list[i].astype('float32')
        W_list[i] = W_list[i].round(2)

    W_list = [Del_na(i) for i in W_list]
    W_list = [i * 100 / np.mean(i) for i in W_list]
    W_list = [np.maximum(i, -i) for i in W_list]

    return W_list
def get_H_op(images_set, list_filenum, P_list, H_list,yaml_p):
    '''
    :param: P_list：shape(10*220*340*3)  other shape for other datasets
    :param images_set:  x
    :return: H(k+1)=H(k)+Belta * [tr(X'X)-tr(P'H)]*P
    '''
   
    temp_list_filenum = list_filenum
    temp_images_set = images_set
    temp_H_list = list()
    H_list_num = list()
    H_list_num.extend([0] * yaml_p["image_maxlen"])
    for i in range(0, yaml_p["image_maxlen"]):
        temp_h = np.array([np.zeros((yaml_p["image_height"], yaml_p["image_width"], 3))]).reshape\
            (yaml_p["image_height"], yaml_p["image_width"], 3)  
        temp_H_list.append(temp_h)
    Belta = 1 
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
            Belta = Belta/2
            temp_H_list[i] = temp_H_list[i] + np.stack([temp_h1, temp_h2, temp_h3], axis=2)
            H_list_num[i] = H_list_num[i] + 1

            img_num = img_num + 1  
    for i in range(0, yaml_p["image_maxlen"]):
        if H_list_num[i] != 0:
            temp_H_list[i] = temp_H_list[i] / H_list_num[i]
        temp_H_list[i] = temp_H_list[i].astype('float32')
        temp_H_list[i] = temp_H_list[i].round(2)
    assert(len(H_list)==len(temp_H_list))
    H_list = [i+j for i, j in zip(H_list, temp_H_list)] 

    H_list = [np.maximum(i,-i) for i in H_list]
    H_list = [Del_na(i) for i in H_list]
    H_list = [i*100/np.mean(i) for i in H_list]


    return H_list

def initial_QMW(yaml_p):
    Q_list = list()
    M_list = list()
    W_list = list()
    np.random.seed(1)
    for i in range(0,yaml_p["image_maxlen"]):
        q_temp = np.random.uniform(low=0, high=1.0, size=(yaml_p["image_height"] , yaml_p["image_width"]))
        q_temp = q_temp.astype('float32')
        q_temp = q_temp.round(2)
        Q_list.append(q_temp)
        W_list.append(q_temp)
        M_list.append(0.1*q_temp)
    return Q_list, M_list, W_list

def get_Lh_op(Q_list,yaml_p):
    '''
    :param Q: to get Lh
    :return: Lh_list list include 10 Lh
    '''
    L_list = list()
    Belta_1 = 1.2
    for i in range(0,len(Q_list)):
        q_temp = Q_list[i]

        q_temp[np.isinf(q_temp)] = 0
        q_temp[np.isnan(q_temp)] = 0
        U, sigma, V = np.linalg.svd(q_temp,1,1)

        U[np.isinf(U)] = 0
        U[np.isnan(U)] = 0
        V[np.isinf(V)] = 0
        V[np.isnan(V)] = 0


        assert (len(sigma)==yaml_p["image_height"] )
        for k in range(0, len(sigma)):
            if np.isnan(sigma[k]):
                sigma[k] = 0
            sigma[k] = np.sign(sigma[k]) * max(0, abs(sigma[k]) * Belta_1)
        sigma = np.concatenate((sigma,np.array([0.0]*400)))
        sigma = np.diag(sigma)
        sigma = sigma[0:yaml_p["image_height"], 0:yaml_p["image_width"] ]
        l_temp = np.dot(U, np.dot(sigma, V))
        l_temp = l_temp.astype('float32')
        l_temp = l_temp.round(2)
        L_list.append(l_temp )

    L_list = [i * 100 / np.mean(i) for i in L_list]

    return L_list
def get_Q_op(M_list, list_filenum, L_list, W_list, H_list,yaml_p):
    '''
    :param H_list:  x
    :param list_filenum:  number of views in each sample
    :param L_list: lagrange
    :param W_list: lagrange
    :param H_list: reconstruction of x
    :param M_list: reconstruction of h
    :return:
    '''
 
    AL_H_list = [np.mean(h, axis=2) for h in H_list]
    temp_list_filenum = list_filenum
    temp_Q_list = list()
    Q_list_num = list()
    Q_list_num.extend([0]*yaml_p["image_maxlen"])
    for i in range(0,yaml_p["image_maxlen"]):
        temp_q = np.array([np.zeros((yaml_p["image_height"] ,yaml_p["image_width"]))]).reshape\
            (yaml_p["image_height"], yaml_p["image_width"])
        temp_Q_list.append(temp_q)

    Belta_1 = 1

    for f_num in temp_list_filenum:
        for i in range(0, f_num):
            temp_m1 = M_list[i]
            temp_m1 = temp_m1 * 10 / np.mean(temp_m1)
            temp_mm1 = np.linalg.inv(np.dot(temp_m1,temp_m1.T)+np.identity(yaml_p["image_height"] ) )
            if np.isnan(temp_mm1).sum() > 1000:
                print("temp_mm1")
                break
            temp_mm1 = temp_mm1 * 10 / np.mean(temp_mm1)
            temp_h1 = AL_H_list[i] * 10 / np.mean(AL_H_list[i])
           
            temp_l1 = L_list[i] * 10 / np.mean(L_list[i])
          
            temp_w1 = W_list[i] * 10 / np.mean(W_list[i] )

            temp_q1 = np.dot(temp_mm1, ( np.dot(np.identity(yaml_p["image_width"]) \
                                         [0:yaml_p["image_height"]], np.dot(-temp_m1.T, temp_h1))- \
                                         Belta_1 * temp_l1-temp_w1) )
            if np.isnan(temp_q1).sum() > 1000:
                temp_q1[np.isinf(temp_q1)] = 0
                print("temp_q1")

                break
            temp_Q_list[i] = temp_Q_list[i] + temp_q1
            Q_list_num[i] = Q_list_num[i] + 1

    for i in range(0,yaml_p["image_maxlen"]):
        if Q_list_num[i]!=0:
            temp_Q_list[i] = temp_Q_list[i]/Q_list_num[i]
    temp_Q_list = [i.astype('float32').round(2) for i in temp_Q_list]
    temp_Q_list = [i * 100 / np.mean(i) for i in temp_Q_list]

    return temp_Q_list

def get_Wh_op(Q_list, L_list, W_list,yaml_p):
    '''
    :param Q_list:
    :param L_list:
    :return:
    '''
    Belta_1 = 0
    gamma = 0.3
    temp_qminusl = [i - j for i, j in zip(Q_list, L_list)]
    temp_qminusl = [gamma*(1+Belta_1)*i for i in temp_qminusl]
    W_list = [i - j for i, j in zip(W_list, temp_qminusl)]

    W_list = [i.astype('float32').round(2) for i in W_list]
    W_list = [i * 100 / np.mean(i) for i in W_list]

    return W_list

def get_M_op(H_list, list_filenum, Q_list, M_list,yaml_p):
    '''
    :param: P_list：shape(10*220*340*3)
    :param images_set:  x
    :return: M(k+1)=M(k)+Belta * [tr(Ha'Ha)-tr(Q'M)]*P
    '''
 
    AL_H_list = [np.mean(h, axis=2) for h in H_list]
    temp_list_filenum = list_filenum

    temp_M_list = list()
    M_list_num = list()
    M_list_num.extend([0] * yaml_p["image_maxlen"])
    for i in range(0, yaml_p["image_maxlen"]):
        temp_m = np.array([np.zeros((yaml_p["image_height"], yaml_p["image_width"]))]).\
            reshape(yaml_p["image_height"], yaml_p["image_width"])  
        temp_M_list.append(temp_m)
    Belta = 0.05e-2

    for f_num in temp_list_filenum:

        for i in range(0, f_num):  
            
            temp_m1 = M_list[i]
            
            temp_h1 = AL_H_list[i]
            
            temp_q1 = Q_list[i]
          
            temp_m1 = np.dot(Belta*(np.sum(temp_h1*temp_h1) - np.sum(temp_q1 * temp_m1)), temp_q1)
            Belta = Belta/2
            assert(temp_m1.shape == temp_M_list[i].shape)
            temp_M_list[i] = temp_M_list[i] + temp_m1

            M_list_num[i] = M_list_num[i] + 1

    for i in range(0, yaml_p["image_maxlen"]):
        if M_list_num[i] != 0:
            temp_M_list[i] = temp_M_list[i] / M_list_num[i]
    temp_M_list = [i.astype('float32').round(2) for i in temp_M_list]

    assert(len(M_list)==len(temp_M_list))
    M_list = [i+j for i, j in zip(M_list, temp_M_list)]
    M_list = [i * 100 / np.mean(i) for i in M_list]

    return M_list


def get_alpha(x_series, M_list, list_filenum, list_2_classes,yaml_p):
    '''
    :param x_series: include alpha:(1,10), u:(1,340,10)  x_series is to be optimized
    :param m:(220,340,10)-->(340,1,10)
    :param list_filenum: length of each sample in a list, The number of images of each sample len:130
    :param list_2_classes:  2 classes result for each sample  len:130
    :return: alpha * u * m
    '''
    temp_list_filenum = list_filenum
    temp_sample_labels = list_2_classes
 
    alpha_matrix = np.array(x_series[0:yaml_p["image_maxlen"]]).reshape(1,yaml_p["image_maxlen"])
    u_matrix = np.array(x_series[yaml_p["image_maxlen"]*1:]).reshape(1,yaml_p["image_width"],yaml_p["image_maxlen"])
    M_list = [np.mean(m, axis=0).reshape(yaml_p["image_width"],1) for m in M_list]

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
    yaml_p, list_num, images, labels, IndexesSet \
        = Load_data(method_use_p=method, data_to_use=dataset_use)
    sum_total = 0.0
    u_matrix = np.array(x[yaml_p["image_maxlen"]:]).reshape(1,yaml_p["image_width"], \
                                                            yaml_p["image_maxlen"])
    for i in range(0, yaml_p["image_maxlen"]): 
         
        temp_u_matrix = np.dot(u_matrix[ :, :, i], u_matrix[ :, :, i].T)-np.identity(yaml_p["image_width"])

        sum_total = sum_total + np.power(np.linalg.norm(temp_u_matrix, ord=2), 1)
    return sum_total

if __name__ == '__main__':

    print("Load data")
    method = 'this_paper'  
    dataset_use = 'YHD_Hos'  
    parm_yaml, list_num, images, labels, IndexesSet \
        = Load_data(method_use_p=method, data_to_use = dataset_use)
    images_set_train = images  
    list_filenum_train = list_num  
    list_2_classes = labels 
    
    print("step 1")
    P_list, H_list, Wx_list = initial_PHW(yaml_p = parm_yaml)
    for Eq1_Loop in range(0, 10000):  
        print(Eq1_Loop)
        Lx_list = get_Lx_op(P_list,yaml_p = parm_yaml)
        P_list = get_P_op(images_set_train, list_filenum_train, Lx_list, \
                          Wx_list, H_list, yaml_p = parm_yaml)
        Wx_list = get_Wx_op(P_list, Lx_list, Wx_list, yaml_p = parm_yaml)
        H_list = get_H_op(images_set_train, list_filenum_train, P_list, \
                          H_list,yaml_p = parm_yaml)

     
        Q_list, M_list, Wh_list = initial_QMW(yaml_p = parm_yaml)
    for Eq2_Loop in range(0, 10000):  
        print(Eq2_Loop)
        Lh_list = get_Lh_op(Q_list,yaml_p = parm_yaml)
        if np.isnan(Lh_list[0]).sum() > 1000:
            print("Lh_list nan!")
            break
        Q_list = get_Q_op(M_list, list_filenum_train, Lh_list, Wh_list, H_list,yaml_p = parm_yaml)
        if np.isnan(Q_list[0]).sum() > 1000:
            print("Q_list nan!")
            break
        Wh_list = get_Wh_op(Q_list, Lh_list, Wh_list,yaml_p = parm_yaml)
        if np.isnan(Wh_list[0]).sum() > 1000:
            print("Wh_list nan!")
            break

        M_list =  get_M_op(H_list, list_filenum_train, Q_list, M_list,yaml_p = parm_yaml)
        if np.isnan(M_list[0]).sum() > 1000:
            print("M_list nan!")
            break
        

    for Iter_steps in range(1, 10000):
        cons_f_1 = ({'type': 'eq', 'fun': cons_final_1})
        conf =[cons_f_1]
        x_0 = [random.uniform(0, 2) for _ in range(0, 1*parm_yaml["image_maxlen"] + \
                                      1 * parm_yaml["image_width"] * parm_yaml["image_maxlen"])]
        M_list = [np.maximum(i, -i) for i in M_list]
        method_op = 'L-BFGS-B'
        sol = sco.minimize(fun = get_alpha, x0 = x_0, args=(M_list, list_filenum_train, list_2_classes,parm_yaml), method=method_op, constraints=conf, \
                           options={'maxiter':Iter_steps, 'disp':True, })

        

