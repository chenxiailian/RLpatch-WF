import os
import time
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset,DataLoader
import torch.utils.data as Data
from torch.autograd import Variable
from sklearn import preprocessing
import torch
import os
import random
import pickle
import argparse
from model.burstDF import burstDFNet
from model.burstAWF import burstAWFNet
from model.burstVarCNN import burstVarCNN
# from model.burstTransformer import burstTransformerNet
from model.new_burst_Transformer import burstTransformerNet


parser = argparse.ArgumentParser(
    description='RLpatch')
# parser.add_argument('-t', '--train', action='store_true',
#     help='Training DNN model for Deep Website Fingerprinting.')
parser.add_argument('-m', '--model', default='DF',
    help='Target DNN model. Supports ``AWF``, ``DF`` and ``VarCNN``')

parser.add_argument('-t', '--target', type=int, default=0,
    help='Train surrogate model or Target_model 0 surrogate model 1 Target_model')

parser.add_argument('-w', '--website_num', type=int, default=100,
    help='website_num')

parser.add_argument('-s', '--save', type=int, default=0,
    help='0 means not save while 1 means save model')

args=parser.parse_args()
if(args.model):
    model_name=args.model
if(args.website_num):
    num_classes=args.website_num

save_tag=args.save
target_tag=args.target

# target=0说明训练的是用户的模型，加载用户数据集和验证集
if(target_tag==0):
    train_data=pd.read_csv('./dataset/trainburst100.csv',index_col=0)
    train_data=np.array(train_data)
    val_data=pd.read_csv('./dataset/valburst100.csv',index_col=0)
    val_data=np.array(val_data)
elif(target_tag==1):
    train_data=pd.read_csv('./dataset/testburst100.csv',index_col=0)
    train_data=np.array(train_data)
    val_data=pd.read_csv('./dataset/valburst100.csv',index_col=0)
    val_data=np.array(val_data)


print(train_data.shape,val_data.shape)
input_shape=(1,800)
if(model_name=='DF'):
    # input_shape=(1,5000)
    # num_classes=99
    lr=0.002
    NB_EPOCH = 50   # Number of training epoch
    print ("Number of Epoch: ", NB_EPOCH)
    model = burstDFNet(input_shape,num_classes)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    BATCH_SIZE = 128 # Batch size
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr, betas=(0.9, 0.99))
elif(model_name=='AWF'):
    # input_shape=(1,5000)
    # num_classes=99
    lr=0.0011
    NB_EPOCH = 30   # Number of training epoch
    print ("Number of Epoch: ", NB_EPOCH)

    model = burstAWFNet(input_shape,num_classes)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    BATCH_SIZE = 256 # Batch size
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
elif(model_name=='VarCNN'):
    # input_shape=(1,5000)
    # num_classes=99
    lr=0.001
    NB_EPOCH = 80   # Number of training epoch
    print ("Number of Epoch: ", NB_EPOCH)

    model = burstVarCNN(input_shape,num_classes)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    BATCH_SIZE = 128 # Batch size
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
elif(model_name=='Transformer'):
    # input_shape=(1,5000)
    # num_classes=99
    lr=0.001
    NB_EPOCH = 50   # Number of training epoch
    print ("Number of Epoch: ", NB_EPOCH)

    model = burstTransformerNet(input_shape,num_classes)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    BATCH_SIZE = 128 # Batch size
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))


# 将训练集数据转为所需形式的输入[length,1,800],并将训练数据放入train_loader中
fr_xtrain= train_data[:,:-1].astype('float32')
fr_xtrain = fr_xtrain.reshape((len(fr_xtrain), 1, 800))
fr_ytrain=train_data[:,-1].astype('float32')

fr_xtrain=torch.from_numpy(fr_xtrain)
fr_ytrain=torch.from_numpy(fr_ytrain)
train_dataset = Data.TensorDataset(fr_xtrain, fr_ytrain)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=2)


# 在最终的测试集上衡量性能，
# 首先导入测试集
x_test = val_data[:,:-1].astype('float32')
x_test = x_test.reshape((len(x_test), 1, 800))
y_test = val_data[:,-1].astype('float32')


x_test=torch.from_numpy(x_test)
y_test=torch.from_numpy(y_test)
val_dataset = Data.TensorDataset(x_test, y_test)
val_loader = Data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=2)



# 模型训练
start_time=time.time()
train_losses = []
train_acces = []
# 用数组保存每一轮迭代中，在测试数据上测试的损失值和精确度，也是为了通过画图展示出来。
eval_losses = []
eval_acces = []
for epoch in range(NB_EPOCH):
    losses=[]
    accuracy=[]
    loop = tqdm(enumerate(train_loader), total =len(train_loader))
    model.train()
    for index,(x,target) in loop:
        x = Variable(x.float())
        target = Variable(target.long())
        x = x.to(device)
        target = target.to(device)
        predict = model(x)
        loss = criterion(predict,target)
        losses.append(loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _,predictions = predict.max(1)
        num_correct = (predictions == target).sum()
        running_train_acc = float(num_correct) / float(x.shape[0])
        accuracy.append(running_train_acc)
        losses.append(loss.item())
        # losses.append(loss.item())
        
        # writer.add_scalar('Training loss',loss ,global_step=step)
        # writer.add_scalar('Training accuracy',running_train_acc,global_step= step)
        # step+=1
        
        #更新信息
        loop.set_description(f'Epoch [{epoch}/{NB_EPOCH}]')
        loop.set_postfix(loss = loss.item(),acc = running_train_acc)
    train_acces.append(sum(accuracy)/len(accuracy))
    train_losses.append(sum(losses)/len(losses))


    # #测试模式
    # model.eval()
    # val_loss=[]
    # val_acc=[]
    # with torch.no_grad():
    #     loop2=tqdm(enumerate(test_loader), total =len(test_loader))
    #     for index,(x,target) in loop2:
    #         x = Variable(x.float())
    #         target = Variable(target.long())
    #         x = x.to(device)
    #         target = target.to(device)
    #         predict = model(x)
    #         loss = criterion(predict,target)
            
    #         _,predictions = predict.max(1)
    #         num_correct = (predictions == target).sum()
    #         running_train_acc = float(num_correct) / float(x.shape[0])
    #         val_acc.append(running_train_acc)
    #         val_loss.append(loss.item())
            
    #         #更新信息
    #         loop2.set_description(f'test[{epoch}/{NB_EPOCH}]')
    #         loop2.set_postfix(loss = loss.item(),acc = running_train_acc)
    # eval_acces.append(sum(val_acc)/len(val_acc))
    # eval_losses.append(sum(val_loss)/len(val_loss))
    # print('Val: '+ 'acc=' + str(eval_acces[epoch])+ '  '+'loss='+str(eval_losses[epoch]))

end_time=time.time()

print("time cost=",end_time-start_time)

# print(target,save_tag)
# 模型存储
if(target_tag==0 and save_tag==1):
    model_path='save_model/'+model_name+'_user.pth'
    torch.save(model,model_path)
elif(target_tag==1 and save_tag==1):
    model_path='save_model/'+model_name+'_attacker.pth'
    torch.save(model,model_path)


# 在最终的测试集上衡量性能，
# 首先导入测试集
x_test = val_data[:,:-1].astype('float32')
x_test = x_test.reshape((len(x_test), 1, 800))
y_test = val_data[:,-1].astype('float32')


x_test=torch.from_numpy(x_test)
y_test=torch.from_numpy(y_test)
val_dataset = Data.TensorDataset(x_test, y_test)
val_loader = Data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=2)


correct=0
total=0
predict_list=[]
y_label=[]
model.eval()
for step, (xtest, ytest) in enumerate(val_loader):
    xtest = Variable(xtest.float())
    # print(xtest.shape)
    xtest = xtest.to(device)
    out=model(xtest)
    # print(out.shape)
    _, predicted = torch.max(out.data, 1)
    predict_list.extend(predicted.cpu().data)
    y_label.extend(ytest)
    
    
from sklearn import metrics
# print(predict_list)
precision=metrics.precision_score(y_label, predict_list,average='macro',zero_division=1)
recall=metrics.recall_score(y_label, predict_list,average='macro',zero_division=1)
F1_score1=metrics.f1_score(y_label, predict_list,average='macro')
acc=metrics. accuracy_score(y_label, predict_list)
F1_score2= 2 * precision * recall / (precision + recall)
print("ACC  TPR/RC  FPR   PR   F1")
print('{:.3f}   {:.3f}  {:.3f}   {:.3f}'.format(acc,recall,precision,F1_score1))








