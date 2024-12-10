import os
import torch
import numpy as np
import time
import pandas as pd
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
from torch.optim import lr_scheduler
from torch.distributions import Normal, Categorical
# from model.DF import DFNet
# from model.AWF import AWFNet
# from model.VarCNN import VarCNN
# from model.Transformer import TransformerNet
from model.burstDF import burstDFNet
from model.burstAWF import burstAWFNet
from model.burstVarCNN import burstVarCNN
# from model.burstTransformer import burstTransformerNet
from model.new_burst_Transformer import burstTransformerNet
from torch.autograd import Variable
import argparse

random.seed(10)

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

def check_all(adv_seq,threat_model,threat_name,device):
    percent = []
    typess = []
    threat = threat_model.to(device)
    adv_seq=Variable(adv_seq.float())
    adv_seq=adv_seq.to(device)
    threat.eval()
    # adv_seq
    out=threat(adv_seq)
    indices = torch.argsort(out, descending=True, dim=1)
    # 获取前 top_k 个类别的索引和概率值
    top_indices = indices[:, :2]
    top_percent = torch.nn.functional.softmax(out, dim=1)[:, top_indices]
    # 将索引和概率值存储在 typess 和 percent 列表中
    # print(top_indices)
    typess.append(top_indices.tolist())
    percent.append(top_percent.tolist())
    
    return typess, percent


from torch import nn
class UNet(nn.Module):
    def __init__(self,inputdim = 1,sgmodel = 3,feature_dim=20):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5_1 = nn.Conv1d(128, 64, 1)
        self.bn5_1 = nn.BatchNorm1d(64)
        self.conv6_1 = nn.Conv1d(64, 1, 1)
        self.bn6_1 = nn.BatchNorm1d(1)
        self.conv5_2 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn5_2 = nn.BatchNorm1d(256)
        self.conv6_2 = nn.Conv1d(256, 128, 1)
        self.bn6_2 = nn.BatchNorm1d(128)
        self.conv7_2 = nn.Conv1d(128, sgmodel, 1)
        self.bn7_2 = nn.BatchNorm1d(sgmodel)
        

        self.fc = nn.Linear(800, feature_dim)
        self.fc2 = nn.Linear(inputdim * inputdim, feature_dim*10)
        self.last_bn = nn.BatchNorm1d(feature_dim)
        
        self.maxpool = nn.MaxPool1d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self._initialize_weights()

    def forward(self, xs):
        x1 = F.relu(self.bn1(self.conv1(xs)))
        # print('x1',x1.shape)
        x2 = F.relu(self.bn2(self.conv2(self.maxpool(x1))))
        # print('x2',x2.shape)
        x3 = F.relu(self.bn3(self.conv3(self.maxpool(x2))))
        # print('x3',x3.shape)
        x4 = F.relu(self.bn4(self.conv4(self.upsample(x3))))
        # print('x4',x4.shape)
        x5_1 = F.relu(self.bn5_1(self.conv5_1(self.upsample(x4))))
        # print('x5_1',x5_1.shape)
        x6_1 = F.relu(self.bn6_1(self.conv6_1(x5_1)))
        # print('x6_1',x6_1.shape)
        # x6_1 = F.relu((self.conv6_1(x5_1)))
        x5_2 = F.relu(self.bn5_2(self.conv5_2(self.upsample(x4))))  # x7in
        # print('x5_2',x5_2.shape)
        x6_2 = F.relu(self.bn6_2(self.conv6_2(self.maxpool(x5_2))))
        # print('x6_2',x6_2.shape)
        x7_2 = F.relu(self.conv7_2(self.upsample(x6_2)))
        # print('x7_2',x7_2.shape)

        
        e1 = torch.softmax(x5_2,dim=1)           # (bt,n_models,h,w)
        # print(e1.shape)
        e2 = torch.mean(e1,dim=1)
        # print(e2.shape)
        e3 = e2.view(e2.size(0), -1)
        # print(e3.shape)
        #print('e.shape = ',e1.shape,e2.shape,e3.shape)
        e4 = self.fc(e3)
        # print(e4.shape)

        # a1 = torch.softmax(x5_2,dim=1)           # (bt,n_models,h,w)
        # a2 = torch.mean(a1,dim=1)
        # a3 = a2.view(a2.size(0), -1)
        # a4 = self.fc2(a3)
        
        return x6_1, self.bn7_2(x7_2),e4#,a4   
    
def clip(x,lower,upper):
    if(lower>upper):
        return 0
    if(x<lower):
        return lower
    if(x>upper):
        return upper
    return x

# change1
def actions2params(actions):
    params_slove = []                                                                  # [[x,y],[w1,w2,...,wn],eps]
    for i in range(len(actions)):
        if(i==0):
            ind = actions[i].cpu().detach().item()#.numpy()#
            #print('ind = ',ind)
            x=ind
            params_slove.append(x)
            temp =[]
            accmw = 1
        elif(i==len(actions)-1):
            new_temp=[]
            temp_sum=sum(temp)
            for j in range(len(temp)):
                new_temp.append(temp[j]/temp_sum)
            params_slove.append(new_temp)
            eps = actions[i].cpu().detach().item()
            # print(eps)
            #params_slove.append(clip(eps,0.009,0.189)) #.copy()
            # eps_sets = np.arange(1/255,21/255,1/255)
            eps_sets = np.arange(0.11,0.51,0.02)
            # eps_sets = np.arange(1,6)
            # print('sets',eps_sets)
            params_slove.append(eps_sets[eps]) #.copy()
        # elif(i==len(actions)-1):
        #     eps = actions[i].cpu().detach().item()
            
        #     eps_sets = np.arange(0.05,10.05,0.05)
        #     params_slove.append(eps_sets[eps]) #.copy()
        else:
            w = actions[i].cpu().detach().item()       #.numpy()[0]
            # print(w,0,accmw,clip(w,0,accmw))
            # clip_w = clip(w,0,accmw)
            # print('clip_w = ',clip_w,accmw)
            temp.append(w) #.copy()
            # accmw -=clip_w
        #print(i,' temp = ',temp)
    return params_slove



def make_burst_mask(burst_tensor,location,length):
    mask = torch.zeros_like(burst_tensor, dtype=torch.float32)  
    mask[:,:,location:location+length] = 1
    # print(mask[location])
    return mask

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


model_names=['Transformer','DF','VarCNN']
threat_name='AWF'
num_classes=95
device_number=0
# device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
start_label=0
end_label=30
test_sample_count=30


parser = argparse.ArgumentParser(
        description='Minipatch: Undermining DNN-based Website Fingerprinting with Adversarial Patches')
parser.add_argument('-m', '--model', action='append',default=None,
        help='fr_model')
parser.add_argument('-t', '--threatmodel', default=threat_name,
        help='threat_model')   
parser.add_argument('-s', '--start_label', type=int,default=start_label,
        help='start_label') 
parser.add_argument('-e', '--end_label', type=int,default=end_label,
        help='end_label') 
parser.add_argument('-c', '--sample_count', type=int,default=test_sample_count,
        help='test_sample_count') 

args=parser.parse_args()
model_names=args.model
threat_name=args.threatmodel
start_label=args.start_label
end_label=args.end_label
test_sample_count=args.sample_count 
# target_tag=0

print(model_names)
print(threat_name)
print(start_label,end_label,device_number,test_sample_count)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 读取数据集
# target=0说明训练的是用户的模型，加载用户数据集和验证集

train_data=pd.read_csv('./dataset/trainburst100.csv',index_col=0)
train_data=np.array(train_data)

# 数据集处理





fr_models = []
for name in model_names:
    print('ensemble model:',name)
    model = load_model(name, device)
    fr_models.append(model)
threat_model = load_target_model(threat_name, device)
print("threat model",threat_name)


targeted=False
# for website_category in range(99):
test_category=60
mean_NQ=[]


start_time=time.time()
'''------------------------Agent initialization--------------------------'''
for web_category in range(start_label,end_label):
    torch.cuda.empty_cache()
    perbution_location=[]
    # perbution_size=[]
    location_param_list=[]
    print('Initializing the agent......')
    print('----------web_category=------------',web_category)
    # start_time=time.time()
    agent = UNet(inputdim = 800,sgmodel = 3).to(device)
    optimizer = torch.optim.Adam(agent.parameters(),lr=1e-03,weight_decay=5e-04)       # optimizer
    scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)                  # learning rate decay
    sample_count=0
    iter_count=0
    for i in range(len(train_data)):
        idx=i
        sample=train_data[idx][:-1]
        label=train_data[idx][-1].astype(int)
        if(label!=web_category):
            continue
        sample_count+=1
        if(sample_count==test_sample_count):
            break
        print('----------web_category=------------',web_category,'----------count=------------',sample_count)
        # burst_array=Transform_ori2burst(sample,800)
        burst_tensor=torch.from_numpy(sample)
        burst_tensor=burst_tensor.view(1, 1, -1)
        burst_tensor = Variable(burst_tensor.float())
        burst_tensor=burst_tensor.to(device)
        
        # print(torch.sum(torch.abs(burst_tensor)))
        #非目标     
        target=label
        baseline = 0.0
        
        #生成掩码矩阵     
        mask=generate_mask(sample)
        # print(mask)
        mask=np.array(mask)
        mask=torch.from_numpy(mask)
        mask=mask.to(device)
        
        print("-----------Initialization with random parameters-------------------")
        last_score = []                                                                    # predicted similarity
        all_final_params = []
        all_best_reward = -2.0
        all_best_face = burst_tensor
        all_best_adv_face_ts = burst_tensor
        num_iter=0
        
        while num_iter<10:
            '''--------------------Agent output feature maps-------------------'''  
            featuremap1, featuremap2, eps_logits = agent(burst_tensor)

            fm_op=featuremap2[0]  # ((n_models) * length,模型重要性或者说模型贡献度)
            n,l = fm_op.shape
            n_models = n
            pre_actions = []
            '''-------------------location action-------------------'''
            op = featuremap1.reshape(1, 1, -1)  # (1, 1, length)
            loct_resp = torch.softmax(op,dim=2)           # (1,1,length)
            loct_probs = torch.mean(loct_resp,dim=0)[0]    
            space=mask
            loct_pbspace = space.reshape(-1) * loct_probs
            loct_preaction = Categorical(loct_pbspace)
            pre_actions.append(loct_preaction)

            '''-------------------weights action-------------------'''
            op = fm_op[:n_models].reshape(n_models,1,-1)  # (n_models,1,h*w)
            weg_resp = torch.mean(op,dim=2).t()             # (1,n_models)
            weg_probs = torch.softmax(weg_resp,dim=1)      # (1,n_models)
            for i in range(n_models):
                dist_weg = Normal(weg_probs[0][i], torch.tensor(0.02).to(device))
                pre_actions.append(dist_weg)
            '''-----------------new epsilon action-------------------''' # value range (0.01,0.2)
            eps_probs = torch.softmax(eps_logits,dim=1)      # (bt,eps_dim)
            dist_eps = Categorical(eps_probs[0])
            pre_actions.append(dist_eps)
            
            cost=0
            '''----------------Policy gradient and Get reward----------------'''
            pg_rewards = []
            phas_final_params = []
            phas_best_reward = -2.0
            phas_best_face = burst_tensor
            phas_best_adv_face_ts = burst_tensor
            N=5 #采样次数
            width=32
            
            for j in range(5):
                iter_count+=1
                log_pis, log_sets = 0, []
                actions=[]
                for j in range(len(pre_actions)):
                    ac=pre_actions[j].sample()
                    actions.append(ac)
                for t in range(len(actions)):
                    log_prob=pre_actions[t].log_prob(actions[t])
                    log_pis+=log_prob
                    log_sets.append(log_prob)
                #攻击参数化
                params_slove = actions2params(actions)  # [x,[w1,w2,...,wn],eps]
                print(params_slove)
                
                mw=width

                if(len(last_score)>2):
                    if(abs(last_score[-1]-last_score[-2])<0.001):
                        if(random.randint(1,2)==1 and len(location_param_list)>0):
                            print("******************************************")
                            print(params_slove)
                            params_slove[0]=random.choice(location_param_list)
                            print(params_slove)
                            print("******************************************")
                                
                x=params_slove[0]
                weights=params_slove[1]
                epsilon=params_slove[2]
                flag=1 if targeted else -1
                X_ori=burst_tensor  #原始序列
                delta = torch.zeros_like(X_ori,requires_grad=True).to(device) #生成与原始突发序列等张量的扰动序列
                
                
                # 扰动序列的掩码序列        
                Pre_mask=make_burst_mask(burst_tensor,x,mw)
                grad_momentum = 0
                
                for itr in range(100):
                    X_adv = X_ori + delta
                    X_adv.requires_grad_(True)
                    X_adv.retain_grad()
                    
                    unique_tensor = torch.zeros_like(Pre_mask)  
                    
                    accm=0
                    
                    for(i,name) in enumerate(model_names):
                        frmodel_out=fr_models[i](X_adv)
                        indices = torch.argsort(frmodel_out, descending=True, dim=1)
                        top_indices = indices[:, :1]
                        top_percent = torch.nn.functional.softmax(frmodel_out, dim=1)[:, label]
                        # print(itr,top_indices,label,top_percent)
                        accm+=top_percent*weights[i]
                    loss = flag *accm
                    loss.backward()
                    
                    # MI operation
                    grad_c = X_adv.grad.clone()  
                    grad_a = grad_c / torch.mean(torch.abs(grad_c), (1), keepdim=True)+1.0*grad_momentum   # 1
                    grad_momentum = grad_a
                    
                    X_adv.grad.zero_()
                    X_adv.data=X_adv.data+epsilon * torch.sign(grad_momentum)* Pre_mask.to(device)
                    
                    delta.data=X_adv-X_ori
                
                                    
                for seq_len in range(len(X_adv[0][0])):
                    if(X_adv[0][0][seq_len]>=0):
                        if(X_ori[0][0][seq_len]<=0):
                            X_adv[0][0][seq_len]=X_ori[0][0][seq_len]
                        else:
                            if(X_adv[0][0][seq_len]<X_ori[0][0][seq_len]):
                                X_adv[0][0][seq_len]=X_ori[0][0][seq_len]
                    else:
                        if(X_ori[0][0][seq_len]>=0):
                            X_adv[0][0][seq_len]=X_ori[0][0][seq_len]
                        else:
                            if(X_adv[0][0][seq_len]>X_ori[0][0][seq_len]):
                                X_adv[0][0][seq_len]=X_ori[0][0][seq_len]
                    
                #计算威胁模型的得分，作为奖励
                # x=
                threat_out=threat_model(X_adv)
                threat_percent=torch.nn.functional.softmax(threat_out, dim=1)
                target_prob=threat_percent[:, label]

                reward_m = target_prob

                reward_g=0
                if(not targeted): reward_m=-reward_m
                reward_f=reward_m +0.1*reward_g
                expected_reward = log_pis * (reward_f - baseline)

                cost -= expected_reward

                pg_rewards.append(reward_m)
                if reward_f > phas_best_reward:
                    phas_final_params = params_slove
                    phas_best_reward = reward_f
                    phas_Ori_face = X_ori
                    phas_best_adv_face_ts = X_adv

            reward_numpy=[reward.cpu().detach().numpy() for reward in pg_rewards]
            observed_value = np.mean(reward_numpy)
            print('{}-th: Reward is'.format(num_iter))

            '''-------------------------Update Agent---------------------------'''
            print("----------------------Update Agent---------------------------")
            optimizer.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(),5.0)
            optimizer.step()

            scheduler.step()

            '''-------------------------Check Result---------------------------'''
            print("-------------------------Check Result--------------------------- ")
            if phas_best_reward > all_best_reward:
                all_final_params = phas_final_params
                all_best_reward = phas_best_reward
                all_Ori_face = phas_Ori_face
                all_best_adv_face_ts = phas_best_adv_face_ts

            sim_labels, sim_probs = check_all(all_best_adv_face_ts,threat_model,threat_name,device)
            succ_label = sim_labels[0][0][0]
            succ_gap = sim_probs[0][0][0][0] 
            print(sim_labels, sim_probs)

                    # #early stop
            if ((targeted and sim_labels[0][0][0] == target) or
                (not targeted and sim_labels[0][0][0] != target)):
                print('early stop at iterartion {},succ_label={},succ_gap={}'.format(num_iter,succ_label,succ_gap))
                print('final_pall_final_paramsall_final_params',all_final_params)
                location_param_list.append(all_final_params[0])
                delta_perbution=(all_best_adv_face_ts-all_Ori_face).cpu().detach().numpy()
                perbution_location.append(delta_perbution[0][0].round())
                print("BWO=",torch.sum(torch.abs(all_best_adv_face_ts-all_Ori_face))/torch.sum(torch.abs(all_Ori_face)))                      
                break
                
            last_score.append(observed_value)    
            last_score = last_score[-200:]   
            print("last_score=",last_score)
            if last_score[-1] <= last_score[0] and len(last_score) == 200:
                print('FAIL: No Descent, Stop iteration')
                break
            num_iter+=1
    location_array=np.array(perbution_location)
    # size_array=np.array(perbution_size)

    print('mean iter count=',iter_count/sample_count)
    # 如果你想要保存多个数组到一个压缩文件中  
    if(len(model_names)>1):
        np.save('./experiment_comparsion/3to'+threat_name+'burstnew/label'+str(web_category)+'.npy', location_array) 
    else:
        np.save('./experiment_comparsion/'+model_names[0]+'to'+threat_name+'burstnew/label'+str(web_category)+'.npy', location_array) 
    end_time=time.time()
    print("cost_time:",end_time-start_time)
    mean_NQ.append(iter_count/sample_count)

mean_NQ=np.array(mean_NQ)
if(len(model_names)>1):
    np.save('./experiment_comparsion/3to'+threat_name+'burstnew/'+str(start_label)+'-'+str(end_label)+'NQ.npy', mean_NQ) 
else:
    np.save('./experiment_comparsion/'+model_names[0]+'to'+threat_name+'burstnew/'+str(start_label)+'-'+str(end_label)+'NQ.npy', mean_NQ) 
# np.save('./experiment1/'+str(start_label)+'-'+str(end_label)+'NQ.npy', mean_NQ) 









