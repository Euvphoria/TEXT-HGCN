import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

dataset = 'yiliao_juzi'

path1 = dataset + '/RNN_ci_all_GRU.pkl'
f1 = open(path1, 'rb')
fea = pickle.load(f1)

ci2ci = []
for i in tqdm(fea):

    temp_ci = [[],[]]
    for num1 , m in enumerate(i):
        for num2 , n in enumerate(i):
            if num1 == num2:
                temp_ci[0].append(num1)
                temp_ci[1].append(num2)
                continue
            m = np.array(m)
            n = np.array(n)
            ma = np.linalg.norm(m)
            mb = np.linalg.norm(m)
            sim = (np.matmul(m, n)) / (ma * mb)
            if sim > 0.6:
                temp_ci[0].append(num1)
                temp_ci[1].append(num2)
    ci2ci.append(temp_ci)


path = dataset + '/ci2ci_bian_all_RNN.pkl'
f = open(path, 'wb')
pickle.dump(ci2ci, f)
f.close()




