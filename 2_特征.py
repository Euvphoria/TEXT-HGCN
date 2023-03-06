import numpy as np
import pickle
from tqdm import tqdm

dataset = 'yiliao_juzi'

path = dataset + '/data_all.pkl'
f1 = open(path, 'rb')
data = pickle.load(f1)

cibiao = open('glove.6B.300d.txt','r',encoding='utf-8')
n = 0
vocab = {}
for i in cibiao:
    n+=1
    temp = i.strip().split(' ')
    a = temp[0]
    b = np.asarray(temp[1:], dtype='float32')
    vocab[a] = b

feature = []
all_new = []
for i in tqdm(data) :
    temp = []
    temp1 = []
    ttt = 0
    for j in i[0]:
        if ttt > 500:
            break
        if j in vocab and j not in temp:
            temp.append(j)
            temp1.append(vocab[j])
        ttt+=1
    if ttt==0:
        print(i)
        continue
    all_new.append([temp, i[1], i[2]])
    feature.append(temp1)


path = dataset + '/data_all_new.pkl'
f = open(path, 'wb')
pickle.dump(all_new, f)
f.close()

import torch
import torch.nn as nn
import numpy as np
import pickle

class DynamicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='GRU'):
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type

        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, x_len, h0=None):
        x_sort_idx = torch.argsort(-x_len)
        x_unsort_idx = torch.argsort(x_sort_idx).long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx.long()]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)

        if self.rnn_type == 'LSTM':
            if h0 is None:
                out_pack, (ht, ct) = self.RNN(x_emb_p, None)
            else:
                out_pack, (ht, ct) = self.RNN(x_emb_p, (h0, h0))
        else:
            if h0 is None:
                out_pack, ht = self.RNN(x_emb_p, None)
            else:
                out_pack, ht = self.RNN(x_emb_p, h0)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type == 'LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]
                ct = torch.transpose(ct, 0, 1)

            return out

max = 0
for i in feature:
    if len(i)>max:
        max =    len(i)
print(max)
qqqq = []
chang = []
jjj = 0
for i in tqdm(feature):
    if len(i) < max:
        try:
            qqqq.append(np.concatenate([i, np.zeros((max-len(i),300), dtype='float64')], axis=0))
            chang.append(len(i))
        except:
            continue
    else:
        qqqq.append(np.array(i))
        chang.append(len(i))

qqqq = torch.tensor(qqqq).to(torch.float32)
chang2= torch.tensor(chang).to(torch.float32)
chushi = 0
batch = 11
LSTM = DynamicLSTM(300,250,bidirectional = True)

for i in tqdm(range(len(qqqq)//batch + 1)):
    if i == (len(qqqq)//batch + 1):
        temp = qqqq[0 + batch*i :]
        temp1 = chang2[0 + batch*i :]
    else:
        temp = qqqq[0 + batch*i : batch * (i+1)]
        temp1 = chang2[0 + batch*i : batch * (i+1)]
    pppp = LSTM(temp,temp1).tolist()
    if i==0:
        q = pppp
    else:
        q.extend(pppp)

c=[]
for i,n in zip(q,chang):
    c.append(i[:n])

path = dataset + '/RNN_ci_all_GRU.pkl'
f = open(path, 'wb')
pickle.dump(c, f)
f.close()


