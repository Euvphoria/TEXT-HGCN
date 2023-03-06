import pickle
from tqdm import tqdm
dataset = 'yiliao_juzi'




__ = open(dataset + '/' +  dataset + '_labels.txt', 'r', encoding='utf-8-sig')
label_zidian = {i.strip():n for n,i in enumerate(__)}

path = dataset + '/data_all_new.pkl'
f = open(path, 'rb')
data = pickle.load(f)
label_shuzi = []
for i in data:
    label_shuzi.append(label_zidian[i[2]])

path = dataset + '/lable_all.pkl'
f = open(path, 'wb')
pickle.dump(label_shuzi, f)
f.close()