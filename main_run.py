
import os
import datetime
import random
import yaml
import pickle
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  
from torch.utils.data import DataLoader
import torchvision.datasets as normal_datasets
import torchvision.transforms as transforms
from sklearn import *
import sklearn
from torch.optim.lr_scheduler import * 
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import cross_val_score,KFold  


class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans,
                              kernel_size=3, padding=1,
                              bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)  
        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity='relu')  
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)  
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        
        return out + x

class NetResDeep(nn.Module):
    def __init__(self, n_chans1=32, n_blocks=5, inchannel=3, fc1_dim = 149600):
        super(NetResDeep, self).__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(inchannel, n_chans1, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(
            *(n_blocks * [ResBlock(n_chans=n_chans1)])
        )  
        self.fc1 = nn.Linear(fc1_dim, 32)
        self.fc2 = nn.Linear(32+7, 2)

    def forward(self, x,x2):
        in_size = x.size(0)  
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        out = out.view(in_size, -1)
        print(out.shape)
        out = torch.relu(self.fc1(out))
        out = torch.cat((out, x2), dim=1)
        out = self.fc2(out)
        return out
def training_loop(n_epochs, optimizer, model_resnet, loss_fn, train_loader):
    device = (torch.device('cpu') if torch.cuda.is_available()
              else torch.device('cpu'))

    scheduler1 = ExponentialLR(optimizer, gamma=0.9)
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, x2, labels in train_loader:
            imgs = imgs.to(device=device)
            x2 = x2.to(device=device)
            labels = labels.to(device=device)
            model_resnet = model_resnet.to(device=device)

            outputs = model_resnet(imgs,x2)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        scheduler1.step()
        if epoch == 1 or epoch % 2 == 0:
            print("{} Epoch {}, Training loss {:.6f}".format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)
            ))
    return model_resnet

def Load_Data(method_use_p = 'this_paper', yaml_path = '..your path',data_to_use = 'YHD_Hos'):
    os.getcwd()
    Work_Path = "...your path"
    os.chdir(Work_Path)

    if method_use_p == 'this_paper':
        file_name = yaml_path + '/this_paper.yaml'
    elif method_use_p == 'pcf':
        file_name = yaml_path + '/pcf.yaml'
    with open(file_name,'r',encoding='utf8') as file: 
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
 
    File_LoadPath = os.getcwd() + "\\Program_Data\\"
    if data_to_use == 'YHD_Hos':
        This_file = "100_Samples_4info.txt"  
        f = open(File_LoadPath + This_file, 'rb')
        dic = pickle.load(f)
        f.close()
        dic.keys()
        list_num_p, images_p, labels_p, IndexesSet_p = dic['list_x'], dic['images_x'], \
                                             dic['labels_x'], dic['IndexesSet_x ']
    elif data_to_use == 'QILU_Hos':
        This_file = "medicalyuan_100sample.txt"  
        f = open(File_LoadPath + This_file, 'rb')
        dic = pickle.load(f)
        f.close()
        dic.keys()
        list_num_p, images_p, labels_p = dic['list_x'], dic['images_x'], dic['labels']
        IndexesSet_p = np.zeros(shape=(len(dic['list_x']), 7), dtype=np.float64)
    elif data_to_use == 'race_and_riding':
        This_file = "race_and_riding.txt"  
        f = open(File_LoadPath + This_file, 'rb')
        dic = pickle.load(f)
        f.close()
        dic.keys()
        list_num_p, images_p, labels_p = dic['list_x'], dic['images_x'], dic['labels']
        IndexesSet_p = np.zeros(shape=(len(dic['list_x']), 7), dtype=np.float64)  
    elif data_to_use == 'Handstand':
        This_file = "pu_and_wa.txt"  
        f = open(File_LoadPath + This_file, 'rb')
        dic = pickle.load(f)
        f.close()
        dic.keys()
        list_num_p, images_p, labels_p = dic['list_x'], dic['images_x'], dic['labels']
        IndexesSet_p = np.zeros(shape=(len(dic['list_x']), 7), dtype=np.float64)  


    return parm_yaml_p, list_num_p, images_p, labels_p, IndexesSet_p
def Make_Data(method_use_p = 'this_paper', images_p = '', \
              list_filenum_p = [],data_to_use = 'YHD_Hos',yaml_p='',):
    yaml_index = 0
    if (method_use_p == 'this_paper') and (data_to_use == 'YHD_Hos'):
        alpha_list = [0.68513654, -0.14204, 0.04513944, -0.2870916, \
                      0.14992187, 1.05251051, \
                      1.16734609, -1.65828772, 3.23830385, -0.65209451]
        yaml_index = 0
    if (method_use_p == 'this_paper') and (data_to_use == 'QILU_Hos'):
        alpha_list = [0.24, -0.31, 0.03, 0.21, 0.12, 0.28, 1.26, 0.14]
        yaml_index = 1
    if (method_use_p == 'this_paper') and (data_to_use == 'race_and_riding'):
        alpha_list = [-0.14, -0.07, 0.17, 0.87, -0.18, -0.41, 0.27, 0.87]
        yaml_index = 2
    if (method_use_p == 'this_paper') and (data_to_use == 'Handstand'):
        alpha_list = [0.95, -0.7, -0.33, 1.18, -0.35, 0.13, 0.04, 0.41]
        yaml_index = 3

    if (method_use_p == 'this_paper'):
        img_list = []
        img_index = 0  
        for f_num in list_filenum_p:
            temp_sum = np.ones((yaml_p["image_height"], yaml_p["image_width"], yaml_p["inchannel"]), dtype=float)
            for img_num in range(0, f_num):
                temp_sum = temp_sum + images_p[img_index + img_num, :, :, :] * alpha_list[img_num]
            temp_sum = temp_sum / f_num
            img_list.append(temp_sum)
            img_index = img_index + f_num
        images_p = np.stack(img_list, axis=0)
    print(method_use_p)
    if (method_use_p == 'pcf'): 
        kernels_num_p = yaml_p["inchannel"]
        FunID_p = yaml_p["FunID"]
        img_list = []
        img_index = 0  
        images_set_pre = np.transpose(images_p, (0, 3, 1, 2))
        for f_num in list_filenum_p:  
            w = yield_kernels(FunID_p, kernels_num_p, f_num * 3)
            temp_img = images_set_pre[img_index:img_index + f_num:, :, :, :]
            temp_img = temp_img.reshape(f_num * 3, yaml_p["image_height"], yaml_p["image_width"])
            temp_img = torch.tensor(temp_img, dtype=torch.float)
            temp_img = temp_img.unsqueeze(0)
            
            out = F.conv2d(temp_img, weight=w, stride=1, padding=1) / f_num
            img_list.append(out)
            img_index = img_index + f_num
        images_p = torch.cat(img_list, dim=0)

    return images_p

def yield_kernels(FunID, kernels_num, channels):

    size_temp = (kernels_num, channels, 3, 3)
    functions_dict = {
        1: np.random.normal(loc=0.0, scale=1.0, size=size_temp),
        2: np.random.beta(1, 2, size=size_temp),
        3: np.random.dirichlet((2, 3, 4), size=(kernels_num, channels, 3)),  
        4: np.random.standard_exponential(size=size_temp),
        5: np.random.f(1, 2, size=size_temp),
        6: np.random.gamma(0.5, 0.5, size=size_temp),
        7: np.random.laplace(0.5, 0.1, size=size_temp),
        8: np.random.logistic(0.5, 0.1, size=size_temp),
        9: np.random.lognormal(0.5, 0.1, size=size_temp),
        10: np.random.uniform(0, 1, size=size_temp),
        11: np.random.vonmises(0, 1, size=size_temp),
        12: np.random.wald(0.5, 0.1, size=size_temp),
        13: np.random.standard_cauchy(size=size_temp)
    }
    kernels = functions_dict.get(FunID)
    kernels = torch.FloatTensor(kernels)
    return kernels


def Deep_Train(method_use_p='this_paper',train_x_p='',
               train_y_p = '',yaml_p='',train_IndexesSet_p=[]):
    """
    :param method_use_p:  method to use
    :param train_x:
    :param train_y:
    :param yaml: parameters in the network
    :return: model_resnet after training
    """
    import torch.utils.data as Data
    if method_use_p == 'this_paper':
        train_x_p = np.transpose(train_x_p, (0, 3, 1, 2))
    else:
        pass

    if train_IndexesSet_p == []: 
        train_IndexesSet_p = list([0]*7)

    
    train_x_p = torch.tensor(train_x_p, dtype=torch.float32).clone()
    train_y_p = torch.tensor(train_y_p, dtype=torch.int64).clone()
    train_IndexesSet_p = torch.tensor(train_IndexesSet_p, dtype=torch.float32).clone()
    torch.manual_seed(1)
    torch_dataset = Data.TensorDataset(train_x_p, train_IndexesSet_p, train_y_p)

    batch_size_p = yaml_p["batch_size"]
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size_p,
        shuffle=True, 
        num_workers=0,  
        
    )
    train_loader = loader
    fc1_dim_p = yaml_p["fc1_dim_p"]
    This_kernels_num = yaml_p["inchannel"]
    model_resnet = NetResDeep(inchannel=This_kernels_num, fc1_dim=fc1_dim_p) 
   
    optimizer = torch.optim.NAdam(model_resnet.parameters(), weight_decay=0.01)

    loss_fn = nn.CrossEntropyLoss()
    n_epochs_p = yaml_p["n_epoch"]
    model_resnet = training_loop(
        n_epochs=n_epochs_p,
        optimizer=optimizer,
        model_resnet=model_resnet,
        loss_fn=loss_fn,
        train_loader=train_loader
                                )
    model_resnet.eval()

    return model_resnet


def Deep_Test(model_resnet,test_x1,test_x2, method_use_p='this_paper'):
    model_resnet.eval()
    assert(test_x1.shape[0] == test_x2.shape[0])
    print("测试开始！")
    y_pred_list = []
    for i in range(0, test_x1.shape[0]):
        temp_x1 = torch.tensor(test_x1[i, :, :, :], dtype=torch.float32).clone()
        temp_x1 = temp_x1.unsqueeze(0).clone() 

        if method_use_p == 'this_paper':
            
            temp_x1 = temp_x1.permute(0, 3, 1, 2).clone()
        temp_x2 = torch.tensor(test_x2[i, :], dtype=torch.float32).clone()
        temp_x2 = temp_x2.unsqueeze(0).clone()
        output = model_resnet.forward(temp_x1, temp_x2)
        _, pred = torch.max(output, 1)
        y_pred_list.append(pred.item())

    return y_pred_list

def ACC_Series(y_true, y_pred):
    TP = 0 
    FP = 0 
    FN = 0 
    TN = 0 
    Err_list = []
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 1:
            TP += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            FP += 1
            Err_list.append(i)
        elif y_pred[i] == 0 and y_true[i] == 0:
            TN += 1
        else:
            FN += 1
            Err_list.append(i)

    print('TP:'+str(TP), ' FP:'+str(FP), ' FN:'+str(FN),' TN:'+str(TN) )
    ACC = sklearn.metrics.accuracy_score(y_true, y_pred)
    Precision = TP / (TP + FP+0.00001)
    return [ACC, Precision,  Err_list]


if __name__ == '__main__':
    
    method = 'pcf' 
    dataset_use = 'YHD_Hos'  
                                 
    parm_yaml, list_num, images, labels, IndexesSet \
        = Load_Data(method_use_p = method, data_to_use = dataset_use)
    train_x = Make_Data(method_use_p = method, \
                       images_p = images, \
                       list_filenum_p = list_num, \
                       data_to_use = dataset_use, \
                        yaml_p=parm_yaml)
    train_x_2 = copy.deepcopy(train_x)
    IndexesSet_2 = copy.deepcopy(IndexesSet)
    labels_2 = copy.deepcopy(labels)
    seed_id = 1
    np.random.seed(seed_id)
    np.random.shuffle(train_x_2)
    np.random.seed(seed_id)
    np.random.shuffle(IndexesSet_2)
    np.random.seed(seed_id)
    np.random.shuffle(labels_2)

    mb,me = 0, int((IndexesSet_2/5)*4), 
    train_x_1 = train_x_2[mb:me,:,:,:]
    train_y_1 = labels_2[mb:me]
    IndexesSet_x_1 = IndexesSet_2[mb:me,:]
    test_x_1 = train_x_2[me:,:,:,:]
    test_y_1 = labels_2[me:]
    IndexesSet_y_1 = IndexesSet_2[me:,:]

    model_resnet = Deep_Train(method_use_p=method, train_x_p=train_x_1, \
        train_y_p=train_y_1, yaml_p=parm_yaml, train_IndexesSet_p=IndexesSet_x_1)
    y_pred_list = Deep_Test(model_resnet, test_x1=test_x_1, \
                test_x2=IndexesSet_y_1, method_use_p=method)

