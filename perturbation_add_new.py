import os
import torch
import numpy as np
import time
from PIL import Image
from matplotlib import pyplot as plt
from torch.optim import lr_scheduler
from torch.nn import DataParallel
import torch.nn.functional as F
# import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
# from facenet_pytorch import MTCNN, InceptionResnetV1
import argparse
import random
from model.burstDF import burstDFNet
from model.burstAWF import burstAWFNet
from model.burstVarCNN import burstVarCNN
# from model.burstTransformer import burstTransformerNet
from model.new_burst_Transformer import burstTransformerNet
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torch.utils.data import Dataset,DataLoader
import torch.utils.data as Data
from torch.autograd import Variable
import pandas as pd
from torch.autograd import Variable
from collections import defaultdict

random.seed(0)

def generate_mask(sequence):
    # 使用列表推导生成掩码序列
    mask=[]
    # print(sequence)
    for i in range(len(sequence)):
        if(sequence[i]!=0):
            mask.append(1)
        else:
            mask.append(0)
        
    # mask = [1 if sequence[i]!=0 else 0 for i in range(len(sequence))]
    return mask

def load_model(model_name,device):
    if(model_name=="DF"):
        model_path="save_model/DF_user.pth"
        model=burstDFNet((1,800),100)
        model=torch.load(model_path,map_location=torch.device('cpu'))
        model.eval()
        model.to(device)
    elif(model_name=="AWF"):
        model_path="save_model/AWF_user.pth"
        model=burstAWFNet((1,800),100)
        model=torch.load(model_path,map_location=torch.device('cpu'))
        model.eval()
        model.to(device)
    elif(model_name=="VarCNN"):
        model_path="save_model/VarCNN_user.pth"
        model=burstVarCNN((1,800),100)
        model=torch.load(model_path,map_location=torch.device('cpu'))
        model.eval()
        model.to(device)
    elif(model_name=="Transformer"):
        model_path="save_model/Transformer_user.pth"
        model=burstTransformerNet((1,800),100)
        model=torch.load(model_path,map_location=torch.device('cpu'))
        model.eval()
        model.to(device)
    return model


def load_target_model(model_name,device):
    if(model_name=="DF"):
        model_path="save_model/DF_attacker.pth"
        model=burstDFNet((1,800),100)
        model=torch.load(model_path,map_location=torch.device('cpu'))
        model.eval()
        model.to(device)
    elif(model_name=="AWF"):
        model_path="save_model/AWF_attacker.pth"
        model=burstAWFNet((1,800),100)
        model=torch.load(model_path,map_location=torch.device('cpu'))
        model.eval()
        model.to(device)
    elif(model_name=="VarCNN"):
        model_path="save_model/VarCNN_attacker.pth"
        model=burstVarCNN((1,800),100)
        model=torch.load(model_path,map_location=torch.device('cpu'))
        model.eval()
        model.to(device)
    elif(model_name=="Transformer"):
        model_path="save_model/Transformer_attacker.pth"
        model=burstTransformerNet((1,800),100)
        model=torch.load(model_path,map_location=torch.device('cpu'))
        model.eval()
        model.to(device)
    return model


model_names=['AWF','VarCNN','Transformer']
threat_name='DF'
# num_classes=99
start_label=0
end_label=99
device_number=0
device = torch.device('cuda:'+str(device_number) if torch.cuda.is_available() else 'cpu')
print(device)

parser = argparse.ArgumentParser(
    description='RLpatch')
# parser.add_argument('-t', '--train', action='store_true',
#     help='Training DNN model for Deep Website Fingerprinting.')
parser.add_argument('-l', '--label', type=int, default=0,
    help='website label')
parser.add_argument('-f', '--fr_model', action='append',default=None,
    help='website label')
parser.add_argument('-t', '--target_model', default=threat_name,
    help='website label')
parser.add_argument('-s', '--start_label', type=int,default=start_label,
        help='start_label') 
parser.add_argument('-e', '--end_label', type=int,default=end_label,
        help='end_label') 

# 读取数据集
# target=0说明训练的是用户的模型，加载用户数据集和验证集
args=parser.parse_args()
model_names=args.fr_model
threat_name=args.target_model
start_label=args.start_label
end_label=args.end_label


train_data=pd.read_csv('./dataset/trainburst100.csv',index_col=0)
train_data=np.array(train_data)

val_data=pd.read_csv('./dataset/valburst100.csv',index_col=0)
val_data=np.array(val_data)

models=[]
for i in range(len(model_names)):
    print(model_names[i])
    fr_model=load_model(model_names[i],device)
    models.append(fr_model)
target_model=load_target_model(threat_name,device)


epsilon1=0.01
BATCH_SIZE=128
all_BWO_length=0
# start_label=0
# end_label=2
acc_list=[]
BWO_list=[]
final_BWO_list=[]
transformer_perturbation_list=[]

BWO_length=0
all_length=0




for test_category in range(start_label,end_label):
    file_path='./experiment_comparsion/'+'3to'+threat_name+'burst/label'+str(test_category)+'.npy'
    # file_path='./experiment_comparsion/'+'3to'+threat_name+'/label'+str(test_category)+'.npy'
    # file_path='./experiment_comparsion/'+'3to'+threat_name+'burstnew/label'+str(test_category)+'.npy'
    new_data=np.load(file_path,allow_pickle=True)
    # print(new_data)

    new_perturbation=np.zeros((800))
    count_vector=np.zeros_like(new_perturbation)
    for i in range(len(new_data)):
        for j in range(len(new_data[i])):
            if(new_data[i][j]!=0):
                new_perturbation[j]+=new_data[i][j]
                count_vector[j]+=1
    
    for i in range(len(new_perturbation)):
        if(count_vector[i]!=0):
            new_perturbation[i]=np.int32(new_perturbation[i]/count_vector[i])

     # 使用argsort函数获取降序索引  
    sorted_indices = count_vector.argsort()[::-1]  
    nonzero_indices = np.nonzero(new_perturbation)[0]  
    max_sampled_length=len(nonzero_indices)

    test_category_acc_list=[]
    min_acc=1
    print('---------------------------------')
    print(test_category)


    for N in range(1,min(30,max_sampled_length)):
    # for N in range(1,max_sampled_length):
        # 取前20个最大的值的索引  
        top_N_indices = sorted_indices[:N]  
        sampled_perbution = np.zeros_like(new_perturbation)  
        sampled_perbution[top_N_indices] = new_perturbation[top_N_indices] 
        adv_seq_array=[]
        BWO_list=[]

        for i in range(len(train_data)):
            sample=train_data[i][:-1]
            label=train_data[i][-1]
            if(label==test_category):
                # burst_array=Transform_ori2burst(sample,800)
                mask=generate_mask(sample)
                new_burst_array=(sample+sampled_perbution)*np.array(mask)
                per_adv_seq=np.append(new_burst_array,np.array(label))
                adv_seq_array.append(per_adv_seq)

        adv_seq_array=np.array(adv_seq_array)
        x_test = adv_seq_array[:,:-1].astype('float32')
        x_test = x_test.reshape((len(x_test), 1, 800))
        y_test = adv_seq_array[:,-1].astype('float32')

        x_test=torch.from_numpy(x_test)
        y_test=torch.from_numpy(y_test)
        val_dataset = Data.TensorDataset(x_test, y_test)
        val_loader = Data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=2)

        correct=0
        total=0
        predict_list0=[]
        predict_list1=[]
        predict_list2=[]
        y_label=[]
        models[0].eval()
        models[1].eval()
        # models[2].eval()
        for step, (xtest, ytest) in enumerate(val_loader):
            xtest = Variable(xtest.float())
            xtest = xtest.to(device)
            out0=models[0](xtest)
            out1=models[1](xtest)
            # out2=models[2](xtest)
            _, predicted0 = torch.max(out0.data, 1)
            _, predicted1 = torch.max(out1.data, 1)
            # _, predicted2 = torch.max(out2.data, 1)
            predict_list0.extend(predicted0.cpu().data)
            predict_list1.extend(predicted1.cpu().data)
            # predict_list2.extend(predicted2.cpu().data)
            y_label.extend(ytest)
        acc0=metrics. accuracy_score(y_label, predict_list0)
        acc1=metrics. accuracy_score(y_label, predict_list1)
        # acc2=metrics. accuracy_score(y_label, predict_list2)
        # acc=(acc0+acc1+acc2)/3
        # print(acc0,acc1,acc2,acc)
        acc=(acc0+acc1)/2

        if(min_acc-acc>0.02):
        # if(acc<min_acc):
            min_acc=acc
            # print(acc)
            test_category_acc_list.append(top_N_indices[N-1])
    print(test_category_acc_list)
    sampled_perbution = np.zeros_like(new_perturbation)  
    sampled_perbution[test_category_acc_list] = new_perturbation[test_category_acc_list]
    transformer_perturbation_list.append(sampled_perbution)

    # print(sampled_perbution)


    print("--------start test----------")
    adv_seq_array=[]
    BWO_list=[]
    ori_seq_array=[]
    for i in range(len(val_data)):
        sample=val_data[i][:-1]
        label=val_data[i][-1]
        if(label==test_category):
            # burst_array=Transform_ori2burst(sample,800)
            mask=generate_mask(sample)
            # new_burst_array=(burst_array+new_sampled_perbution)*np.array(mask)
            new_burst_array=(sample+sampled_perbution)*np.array(mask)
            BWO_length+=np.sum(np.abs(new_burst_array-sample))
            all_length+=np.sum(np.abs(sample))

            BWO=(np.sum(np.abs(new_burst_array))/np.sum(np.abs(sample)))-1
            BWO_list.append(BWO)
            per_adv_seq=np.append(new_burst_array,np.array(label))
            adv_seq_array.append(per_adv_seq)
                # print(per_adv_seq)
    adv_seq_array=np.array(adv_seq_array)
    ori_seq_array=np.array(ori_seq_array)
    print(N,"-th","BWO",np.mean(np.array(BWO_list)))
    final_BWO_list.append(np.mean(np.array(BWO)))
    x_test = adv_seq_array[:,:-1].astype('float32')
    x_test = x_test.reshape((len(x_test), 1, 800))
    y_test = adv_seq_array[:,-1].astype('float32')

    x_test=torch.from_numpy(x_test)
    y_test=torch.from_numpy(y_test)
    val_dataset = Data.TensorDataset(x_test, y_test)
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=2)


    correct=0
    total=0
    predict_list=[]
    y_label=[]
    target_model.eval()
    for step, (xtest, ytest) in enumerate(val_loader):
        xtest = Variable(xtest.float())
        # print(xtest.shape)
        xtest = xtest.to(device)
        out=target_model(xtest)
        # print(out.shape)
        _, predicted = torch.max(out.data, 1)
    predict_list.extend(predicted.cpu().data)
    y_label.extend(ytest)
    acc=metrics. accuracy_score(y_label, predict_list)
    print('label',test_category,"ACC  TPR/RC  FPR   PR   F1")
    print('{:.3f} '.format(acc))
    print("--------end test----------")
    acc_list.append(acc)
print('acc',np.mean(np.array(acc_list)))
print('BWO',np.mean(np.array(final_BWO_list)))
print('BWO2',BWO_length/all_length)

transformer_perturbation_list=np.array(transformer_perturbation_list)
save_path='./experiment_comparsion/'+'10302frmodelburstdata3to'+ threat_name+'acc'+str(start_label)+'-'+str(end_label)+'.csv'
pd.DataFrame(transformer_perturbation_list).to_csv(save_path)

# all_transformer_pebution.append(transformer_pebution)
# all_label.append(transformer_label_list)


        
    


    