import numpy as np
import pickle
import io
from tqdm import tqdm
import torch
dataset = 'yiliao_juzi'

path2 = dataset + '/data_all_new.pkl'
path3 = dataset + '/RNN_ci_all_GRU.pkl'

f2 = open(path2, 'rb')
shuju = pickle.load(f2)
f3 = open(path3, 'rb')
fea_ci = pickle.load(f3)
ju_feature = []

for m,n in tqdm(zip(shuju,fea_ci)):
    zidian_temp = {}
    for ppp,qqq in zip(m[0],n):
        if ppp in zidian_temp:
            zidian_temp[ppp] = (torch.tensor(zidian_temp[ppp]) + torch.tensor(qqq))/2
        else:
            zidian_temp[ppp] = torch.tensor(qqq)
    ju_pian = []

    for ju in m[1]:
        temp = ju.split()
        num = 0

        for k in temp:
            if k in zidian_temp:
                if num ==0:
                    ww = zidian_temp[k]
                elif num>2000:
                    break
                else:
                    ww = ww + zidian_temp[k]
                num+=1
            else:
                continue
        ww = ww/num
        ju_pian.append(ww.tolist())
    ju_feature.append(ju_pian)

path = dataset + '/RNN_ju_all_GRU.pkl'
f = open(path, 'wb')
pickle.dump(ju_feature, f)
f.close()


