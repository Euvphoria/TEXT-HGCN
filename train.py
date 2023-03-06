import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import random
import pickle
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


batch_size = 256

dataset_name = '20ng' #  lr_num = 0.05 drop_num = 0.2
lr_num = 0.5
drop_num = 0.2


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    dgl.seed(seed)


def create_graph():
    path = dataset_name + '/ci2ci_bian_all_RNN.pkl'
    f = open(path, 'rb')
    ci2ci = pickle.load(f)

    path1 = dataset_name + '/ci2doc_bian_all_RNN.pkl'
    f1 = open(path1, 'rb')
    ci2bian = pickle.load(f1)

    path2 = dataset_name + '/RNN_ci_all.pkl'

    f2 = open(path2, 'rb')
    fea_ci = pickle.load(f2)

    path3 = dataset_name + '/RNN_ju_all.pkl'
    f3 = open(path3, 'rb')
    fea_ju = pickle.load(f3)

    path4 = dataset_name + '/lable_all.pkl'
    f4 = open(path4, 'rb')
    lable = pickle.load(f4)

    data = []
    for qqq,(a,b,c,d,e) in enumerate(zip(ci2ci,ci2bian,fea_ci,fea_ju,lable)):

        hetero_graph = dgl.heterograph({
            ('ci', '1', 'ci'): (a[0], a[1]),
            ('ci', '2', 'ci'): (a[1], a[0]),
            ('ci', '3', 'ju'): (b[0], b[1]),
            ('ju', '4', 'ci'): (b[1], b[0]),
         })
        try:
            hetero_graph.nodes['ci'].data['feat'] = torch.tensor(c)
            hetero_graph.nodes['ju'].data['feat'] = torch.tensor(d)
            e = torch.tensor(e)
            e = torch.unsqueeze(e,0)
            data.append((hetero_graph,e))
        except:
            continue
    return data,len(set(lable))

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')
    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()
        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        # self.classify = nn.Linear(hidden_dim, n_classes)
        self.classify = nn.Linear(in_dim, n_classes)
        self.sig = nn.Sigmoid()
    def forward(self, g):
        h = g.ndata['feat']
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['feat'] = h
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'feat', ntype=ntype)
            hg = F.normalize(F.relu(hg),1)
            hg1 =self.sig(hg)
            hg = F.dropout(hg,p=drop_num)
            hg = self.classify(hg)

            return hg,hg1

def collate(samples):
    graphs, labels = zip(*samples)
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)
    return batched_graph, batched_labels

origina_X = 0
origina_Y = []

dataset,lable_number = create_graph()
etypes = dataset[0][0].etypes
random.shuffle(dataset)
num = len(dataset)
train_size = int(num*0.9)
test_size = num - train_size
train_data = dataset[:train_size]
test_data = dataset[train_size:]


for n,i in enumerate(test_data):
    hg = 0
    for ntype in i[0].ntypes:
        hg = hg + dgl.readout_nodes(i[0], 'feat',op = 'min' ,ntype=ntype)
    if n == 0:
        origina_X = hg
    else:
        origina_X = torch.cat((origina_X, hg), dim=0)
    origina_Y.append(i[1].item())


def hautu(X,yy,temp_name,dataset) :
    tsne = TSNE(n_components=2,init='pca')
    Y = tsne.fit_transform(X)
    plt.scatter(Y[:,0], Y[:,1], c = yy)
    plt.savefig(dataset + '/' + dataset + '_' + temp_name + '.jpg',dpi = 300)
    plt.show()

# hautu(origina_X,origina_Y,'qian',dataset_name)


def test():

    After_processing_Y = []
    test_pred, test_label = [], []
    with torch.no_grad():
        for it, (batchg, label) in enumerate(test_loader):
            batchg, label = batchg.to(DEVICE), label.to(DEVICE)
            pre,temp =  model(batchg)

            pred = torch.softmax(pre,1)
            pred = torch.max(pred, 1)[1].view(-1)
            test_pred += pred.detach().cpu().numpy().tolist()
            test_label += label.cpu().numpy().tolist()
            accc =  accuracy_score(test_label, test_pred)


            After_processing_Y += label.tolist()
            if it == 0:
                After_processing_X = temp.cpu()
            else:
                After_processing_X = torch.cat((After_processing_X, temp.cpu()), dim=0)


    return  accc


train_data = DataLoader(train_data,batch_size=batch_size,collate_fn=collate,drop_last=False,shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True,collate_fn=collate)
etypes = dataset[0][0].etypes
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bestttt =[]
for ffff in range(10):
    model = HeteroClassifier(500, 600 , lable_number, etypes)
    model.to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr_num)
    model.train()
    epoch_losses = []
    best_acc = 0
    for epoch in range(100):
       epoch_loss = 0
       test_pred, test_label = [], []
       for iter,(batched_graph, labels) in enumerate(train_data):
           batched_graph, labels = batched_graph.to(DEVICE), labels.to(DEVICE)
           logits ,_= model(batched_graph)
           loss = F.cross_entropy(logits, labels.squeeze(-1))
           opt.zero_grad()
           loss.backward()
           opt.step()
           pred = torch.softmax(logits, 1)
           pred = torch.max(pred, 1)[1].view(-1)
           test_pred += pred.detach().cpu().numpy().tolist()
           test_label += labels.cpu().numpy().tolist()
           epoch_loss += loss.detach().item()
       epoch_loss /= (iter + 1)
       q = accuracy_score(test_label, test_pred)
       print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss),end='\t')
       print("Train accuracy: ", accuracy_score(test_label, test_pred),end='\t')
       epoch_losses.append(epoch_loss)
       temp_acc = test()
       if temp_acc > best_acc:
           best_acc = temp_acc
       print("BestTest accuracy: ", best_acc)
    model.eval()
    test_pred, test_label = [], []
    After_processing_Y = []
    bestttt.append(best_acc)
    print(bestttt)

    
# with torch.no_grad():
#     for it, (batchg, label) in enumerate(test_loader):
#         batchg, label = batchg.to(DEVICE), label.to(DEVICE)
#         pre,temp =  model(batchg)
#         pred = torch.softmax(pre,1)
#         pred = torch.max(pred, 1)[1].view(-1)
#         test_pred += pred.detach().cpu().numpy().tolist()
#         test_label += label.cpu().numpy().tolist()
#         After_processing_Y += label.tolist()
#         if it == 0:
#             After_processing_X = temp.cpu()
#         else:
#             After_processing_X = torch.cat((After_processing_X, temp.cpu()), dim=0)

#    hautu(After_processing_X, After_processing_Y,'hou',dataset_name)

print("best accuracy: ", bestttt)


