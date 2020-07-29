#!/usr/bin/env python
# coding: utf-8
import sys

import pandas as pd
import numpy as np
import os
from collections import Counter, defaultdict
from nltk.tokenize import TweetTokenizer

tkzr = TweetTokenizer()

import multiprocessing as mp

from torch import nn, tensor
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader

import time
from torch.utils.data.dataset import random_split

# get exp dataset
if len(sys.argv) != 2:
    print("Run as : CUDA_VISIBLE_DEVICES=0 python3 train_textlevel_gnn.py DATASET_NAME")
    exit()

experiment_dataset = sys.argv[1]

print("Experiment on :", experiment_dataset) #, sys.argv)

# # Read File

DATA_PATH = 'dataset'
DATA = {'R8':'R8', 'R52':'R52', '20ng':'20ng', 'Ohsumed':'ohsumed_single_23'}


def read_data(data_name):
    dataset = []
    if data_name=='R8' or data_name =='R52':
        targets = ['train.txt', 'test.txt']
        for target in targets:
            trainind_data_path = os.path.join(DATA_PATH, data_name, target)
            lines = open(trainind_data_path).readlines()

            for line in lines:
                label, text = line.strip().split('\t')

                # add doc
                dataset.append((target[:-4], label, text))

    # ohsumed_single_23
    if data_name == 'Ohsumed':
        targets = {'training':'train', 'test':'test'}
        for target in targets:
            trainind_data_path = os.path.join(DATA_PATH, DATA[data_name], target)
            for label in os.listdir(trainind_data_path):
                for doc in os.listdir(os.path.join(trainind_data_path, label)):
                    lines = open(os.path.join(trainind_data_path, label, doc)).readlines()
                    text = " ".join([line.strip() for line in lines])

                    # add doc
                    dataset.append((targets[target], label, text))

    # 20 ng
    if data_name =='20ng':
        from sklearn.datasets import fetch_20newsgroups

        for target in ['train', 'test']:
            data = fetch_20newsgroups(subset=target, shuffle=True, random_state=42, remove = ('headers', 'footers', 'quotes'))


            dataset += list(map(lambda sample: (target, data['target_names'][sample[0]], sample[1].replace("\n", " ")), zip(data['target'], data['data'])))

    print("\tDataset Loaded! Total:", len(dataset))
    return dataset
print("\nLoading dataset..")
data_pd = pd.DataFrame(read_data(experiment_dataset), columns=["target", "label", "text"])
data_pd.head()


# # Build Graph

# ## Nodes

print("\nFeature Transforming..")

data_pd['tokens'] = data_pd.text.apply(lambda text: [w.lower() for w in tkzr.tokenize(text)])
data_pd.head()


print("\tRemove empty/too long text data..")
data_pd = data_pd[data_pd.tokens.apply(lambda tokens: (len(tokens)>0) & (len(tokens)< 1000))]
distribution = data_pd['tokens'].apply(lambda tokens: len(tokens)).describe()
print(distribution)

MAX_LEN = int(distribution['mean']+ 3*distribution['std'])
print("\t\tRemove text > {} tokens (mean + 3x std of word counts)".format(MAX_LEN))

data_pd = data_pd[data_pd.tokens.apply(lambda tokens: len(tokens) < distribution['mean'] + 3 * distribution['std'])]

print("\t\tFinish, Total Rest:", len(data_pd))
word_nodes = Counter([w for tokens in data_pd.tokens.to_list() for w in tokens])
print("\tMost common words:", word_nodes.most_common(10))
# remove rare words
word_limit = 15
word_nodes = [w for w,v in word_nodes.items() if v > word_limit]
print("\tTotal words:", len(word_nodes))


word2idx = {word:i+1 for i,word in enumerate(word_nodes)}
word2idx[0] = '_PAD_'


with mp.Pool(mp.cpu_count()) as p:
    data_pd['X'] = data_pd.tokens.apply(lambda tokens: [word2idx.get(w) if word2idx.get(w) else 0 for w in tokens])

data_pd.head()


skip_size = 2 # from paper


def get_word_neighbor(text_tokens, skip=skip_size):
    text_len = len(text_tokens)

    edge_neighbors = []
    for w_idx in range(text_len):
        skip_neighbors = []
        # check before
        for sk_i in range(skip):
            before_idx = w_idx -1 - sk_i
            skip_neighbors.append(text_tokens[before_idx] if before_idx > -1 else 0)

        # check after
        for sk_i in range(skip):
            after_idx = w_idx +1 +sk_i
            skip_neighbors.append(text_tokens[after_idx] if after_idx < text_len else 0)

        edge_neighbors.append(skip_neighbors)
    return edge_neighbors


with mp.Pool(mp.cpu_count()) as p:
    data_pd['X_Neighbors'] = p.map(get_word_neighbor, data_pd.X.to_list())
data_pd.head()


data_pd['y'] = pd.Categorical(data_pd.label).codes
data_pd.head()

data_pd['y'].values

print("\tFinish")

# ## Dataset


NUM_NODES = len(word2idx) # with padding
WINDOW_SIZE = skip_size
NUM_CLASSES = len(set(data_pd['y'].values))


class TextDataset(Dataset):

    def __init__(self, data_pd, num_nodes, max_len=None, transform=None):
        """
        Args:
            num_nodes : number of nodes including padding

        """
        self.X = data_pd['X'].values
        self.NX = data_pd['X_Neighbors'].values
        self.y = data_pd['y'].values

        self.num_words = num_nodes-1

        if not max_len:
            self.max_len = max([len(text) for text in self.X])
        else:
            self.max_len = max_len

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # padding with 0
        X = np.zeros(self.max_len, dtype=np.int)
        X[:min(len(self.X[idx]), self.max_len)] = np.array(self.X[idx])[:min(len(self.X[idx]), self.max_len)]
        NX = np.zeros((self.max_len, 4), dtype=np.int)
        NX[:min(len(self.NX[idx]), self.max_len),:] = np.array(self.NX[idx])[:min(len(self.NX[idx]), self.max_len),:]


        # calculate edge weight index (EW_idx), 0 is for padding with padding (0,0)
        EW_idx = ((X-1)*self.num_words).reshape(-1,1)+NX
        EW_idx[NX == 0] = 0 # set neighbor with PAD or unknow node as 0
        EW_idx[X==0] = 0 # set unknow word node as 0

        y = torch.tensor(self.y[idx], dtype=torch.long)

        sample = {'X': X,
                  'NX': NX,
                  'EW': EW_idx,
                  'y': y}

#         if self.transform:
#             sample = self.transform(sample)

        return sample


# # Model

WORD_EMBED_DIM = 300

word_embedding_file = 'glove.twitter.27B.{}d.txt'.format(WORD_EMBED_DIM) if WORD_EMBED_DIM < 300 else 'glove.6B.{}d.txt'.format(WORD_EMBED_DIM)

print("\nLoading pretrain word embedding as:", word_embedding_file)


with open('embeddings/'+word_embedding_file, encoding="utf8") as f:
    lines = f.readlines()
    embedding = np.random.random((len(lines), WORD_EMBED_DIM))
    for line in f.readlines():
        line_split = line.strip().split()
        if line_split[0] in word_nodes:
            embedding[word_nodes.index(line_split[0])] = line_split[1:]

    embedding[0] = 0

print("\tFinish")


class TextLevelGNN(nn.Module):

    def __init__(self, num_nodes, node_feature_dim, class_num, window_size, embeddings=0):
        super(TextLevelGNN, self).__init__()

        if type(embeddings) != int:
            print("\tConstruct pretrained embeddings")
            self.node_embedding = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=0)
        else:
            self.node_embedding = nn.Embedding(num_nodes, node_feature_dim, padding_idx = 0)

        self.edge_weights = nn.Embedding((num_nodes-1) * (num_nodes-1) + 1, 1, padding_idx=0) # +1 is padding
        self.node_weights = nn.Embedding(num_nodes, 1, padding_idx=0) # Nn, node weight for itself

        self.fc = nn.Sequential(
            nn.Linear(node_feature_dim, class_num, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Softmax(dim=1)
        )

    def forward(self, X, NX, EW):
        """
        INPUT:
        -------
        X  [tensor](batch, sentence_maxlen)               : Nodes of a sentence
        NX [tensor](batch, sentence_maxlen, window_size*2): Neighbor nodes of each nodes in X
        EW [tensor](batch, sentence_maxlen, window_size*2): Neighbor weights of each nodes in X

        OUTPUT:
        -------
        y  [list] : Predicted Probabilities of each classes
        """
        ## Neighbor
        # Neighbor Messages (Mn)
        Mn = self.node_embedding(NX) # (BATCH, SEQ_LEN, NEIGHBOR_SIZE, EMBED_DIM)

        # EDGE WEIGHTS
        En = self.edge_weights(NX) # (BATCH, SEQ_LEN, NEIGHBOR_SIZE )

        # get representation of Neighbors
        Mn = torch.sum(En * Mn, dim=2) # (BATCH, SEQ_LEN, EMBED_DIM)

        # Self Features (Rn)
        Rn = self.node_embedding(X) # (BATCH, SEQ_LEN, EMBED_DIM)

        ## Aggregate information from neighbor
        # get self node weight (Nn)
        Nn = self.node_weights(X)
        Rn = (1 - Nn) * Mn + Nn * Rn

        # Aggragate node features for sentence
        X = Rn.sum(dim=1)
        y = self.fc(X)
        return y


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Model Running on device:", device)

model = TextLevelGNN(NUM_NODES, WORD_EMBED_DIM, NUM_CLASSES,WINDOW_SIZE, embeddings=torch.tensor(embedding, dtype=torch.float)).to(device)
print(model)


# # Trainning

def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_,
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      num_workers=8
                    )
    model.train()
    for i, sampled_batch in enumerate(data):
        X = sampled_batch['X'].to(device)
        NX = sampled_batch['NX'].to(device)
        EW = sampled_batch['EW'].to(device)
        y = sampled_batch['y'].to(device)

        optimizer.zero_grad()

        output = model(X, NX, EW)

        loss = criterion(output, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == y).sum().item()

    # Adjust the learning rate
#     scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, num_workers=8)

    model.eval()
    for i, sampled_batch in enumerate(data):
        X = sampled_batch['X'].to(device)
        NX = sampled_batch['NX'].to(device)
        EW = sampled_batch['EW'].to(device)
        y = sampled_batch['y'].to(device)

        with torch.no_grad():
            output = model(X, NX, EW)
            loss = criterion(output, y)
            loss += loss.item()
            acc += (output.argmax(1) == y).sum().item()

    return loss / len(data_), acc / len(data_)


SEQ_LEN = None

data_tr = TextDataset(data_pd[data_pd.target=='train'],len(word2idx), max_len=SEQ_LEN)
data_te = TextDataset(data_pd[data_pd.target=='test'],len(word2idx), max_len=SEQ_LEN)

train_len = int(len(data_tr) * 0.9)
sub_train_, sub_valid_ = random_split(data_tr, [train_len, len(data_tr) - train_len])
print("Data size: {} train, {} valid, {} test".format(len(sub_train_), len(sub_valid_),len(data_te)))


N_EPOCHS = 500
BATCH_SIZE = 32

criterion = torch.nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.1)

best_var_loss = 5
best_var_acc = 0
step_from_best = 0
best_test = []
print("Start Training...")
for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)

    print(f'Epoch:{epoch+1:4d} ({secs:2d} sec)\t|\tLoss: {train_loss:.4f}(train)\t{valid_loss:.4f}(valid)\t|\tAcc: {train_acc * 100:.1f}%(train)\t{valid_acc * 100:.1f}%(valid)')

    # if valid_loss > best_var_loss:
    if valid_acc < best_var_acc:
        step_from_best +=1
        if step_from_best > 25:
            print("Early Stop!")
            step_from_best =0
            # break
    else:
        # best_var_loss = valid_loss
        best_var_acc = valid_acc
        step_from_best = 0
        test_loss, test_acc = test(data_te)
        best_test.append((epoch+1, test_loss.cpu().item(), test_acc))



print('Checking the results of test dataset...')
test_loss, test_acc = test(data_te)
print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')
print("\nPrevious best accuracy records:", best_test)

"""
R8 (8 cls)
Acc: 62.6%(train)   96.3%(valid)    96.4%(test)

R52 (52 cls)
Acc: 56.8%(train)   90.7%(valid)    91.0%(test)

20ng (20 cls)
Acc: 44.8%(train)   60.7%(valid)    55.6%(test)

Ohsumed (23 cls)
Acc: 16.0%(train)   16.3%(valid)    23.4%(test)
"""

