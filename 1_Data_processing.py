from nltk.corpus import stopwords
from utils import clean_str
import re
import pickle
from tqdm import tqdm
dataset = 'R52'
stop_words = set(stopwords.words('english'))

doc_content_list = []
ci_content_list = []
f = open(dataset + '/' +  dataset + '_all.txt', 'rb')
f1 = open(dataset + '/' +  dataset + '.clean.txt', 'rb')

num = 0
for line in f.readlines():
    num+=1
    doc_content_list.append(line.strip().decode('latin1'))

for line1 in f1.readlines():
    num+=1
    ci_content_list.append(line1.strip().decode('latin1'))

f.close()

#分词分句
all = []
pattern = r'Lines: \d{1,3}'
for n,(i,k) in tqdm(enumerate(zip(doc_content_list,ci_content_list))):
    i = re.split(pattern, i)[-1]
    doc1 = i.split('.')
    ci = k.split(' ')
    doc2 = []
    num = 0
    for j in doc1:
        if len(j.split()) > 5:
            num+=1
            j = clean_str(j)
            doc2.append(j)
    if doc2 == []:
        doc2.append(clean_str(i))
    all.append([ci, doc2])


q = open(dataset + '/' +  dataset + '_vocab.txt','r',encoding='utf-8')
vocab= []
for i in q:
    vocab.append(i.strip())


lable = open(dataset + '/' +  dataset + '.txt', 'r', encoding='utf-8-sig')
all_new=[]
for aaa,bbb in tqdm(zip(all,lable.readlines())):
    temp = []
    biaoqian = bbb.split()[2].strip()
    for i in aaa[0]:
        temp.append(i)
    ju = []
    for j in aaa[1]:
        temp2 = []
        temp1 = j.split()
        for k in temp1:
            if  k  in vocab and k not in stop_words:
                temp2.append(k)
        qqqqqq = ' '.join(temp2)
        ju.append(qqqqqq)
    if ju ==[]:
        print(aaa)
        print('ju')
        continue
    if temp==[] and ju ==[]:
        continue
    all_new.append([temp, ju,biaoqian])


num = 0

#保存
path = dataset + '/data_all.pkl'
f = open(path, 'wb')
pickle.dump(all_new, f)
f.close()



