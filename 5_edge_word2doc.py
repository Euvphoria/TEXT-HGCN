import pickle
from tqdm import tqdm


dataset = 'ohsumed'

path =   dataset + '/Bert_all_data.pkl'
f = open(path, 'rb')
data = pickle.load(f)


ci2doc_all = []

for i in tqdm(data):
    zidian_ci_temp = {}
    ci2doc_pian = []
    ci = []
    ju = []
    for n,k in enumerate(i[0]):
        zidian_ci_temp[k] = n
    for n,k in enumerate(i[1]):
        qqq = k.split()
        for g in qqq:
            if g in zidian_ci_temp:
                ci.append(zidian_ci_temp[g])
                ju.append(n)
            else:
                continue
    ci2doc_pian.append(ci)
    ci2doc_pian.append(ju)
    ci2doc_all.append(ci2doc_pian)




path =  dataset + '/Bert_ci2doc_bian.pkl'
f = open(path, 'wb')
pickle.dump(ci2doc_all, f)
f.close()
