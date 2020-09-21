
import torch
from torch import nn, tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
import time
import numpy as np
import pandas as pd
from pprint import pprint


class TextLevelGNN_Model:
    def __init__(self, word2idx: dict,
                       nums_classes: int,
                       word_embedding_dim: int,
                       pretrain_embedding: bool = False,
                       pretrain_embedding_fix: bool = False,
                       seq_len: int=0,
                       ):
        
        print("\nModel Preparing..")

        self.word2idx = word2idx
        self.nums_node = len(word2idx) # all the distinct words including _PAD_ and _UNKNOWN_

        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Model Running on device:", self.device)
        self.word_embedding_dim = word_embedding_dim
        if pretrain_embedding:
            self.word_embedding_file = 'glove.twitter.27B.{}d.txt'.format(self.word_embedding_dim) if self.word_embedding_dim < 300 else 'glove.6B.{}d.txt'.format(self.word_embedding_dim)
            
        self.model = TextLevelGNN(self.nums_node,
                                  word_embedding_dim,
                                  nums_classes,
                                  embeddings=(self.load_pretrain_embedding() if pretrain_embedding else 0),
                                  embedding_fix=pretrain_embedding_fix
                                  ).to(self.device)
        print("\n", self.model)
        
        self.data_train = None
        self.data_valid = None
        self.data_test = None
        self.seq_len = seq_len

    
    def load_pretrain_embedding(self):
        print("\tLoading pretrain word embedding from:", self.word_embedding_file)

        with open('embeddings/'+self.word_embedding_file, encoding="utf8") as f:
            lines = f.readlines()
            embedding = np.random.random((self.nums_node, self.word_embedding_dim))
            for line in f.readlines():
                line_split = line.strip().split()
                if line_split[0] in self.word2idx:
                    embedding[self.word2idx[line_split[0]]] = line_split[1:]

            embedding[0] = 0 # set the _PAD_ as 0
        return torch.tensor(embedding, dtype=torch.float)

    def set_dataset(self, data_pd: pd.DataFrame):
        
        data_tr = TextLevelGNNDataset(data_pd[data_pd.target == 'train'], self.nums_node, max_len=self.seq_len)
        
        # Split validation from trainind dataset if there is no validation dataset
        if len(data_pd[data_pd.target == 'valid']) == 0:
            train_len = int(len(data_tr) * 0.9)
            sub_train_, sub_valid_ = random_split(data_tr, [train_len, len(data_tr) - train_len])

            self.data_train = sub_train_
            self.data_valid = sub_valid_
        else:
            self.data_train = data_tr
            self.data_valid = TextLevelGNNDataset(data_pd[data_pd.target == 'valid'], self.nums_node, max_len=self.seq_len)

        self.data_test = TextLevelGNNDataset(data_pd[data_pd.target=='test'], self.nums_node, max_len=self.seq_len)

        print("\tData size: {} train, {} valid, {} test".format(len(self.data_train), len(self.data_valid),len(self.data_test)))

    def train_eval(self,epochs: int,
                        batch_size: int,
                        learning_rate: float = 0.001,
                        weight_decay: float = 0.0001,
                        early_stop_epochs: int = 15,
                        early_stop_monitor: str = "loss",
                        warmup_epochs: int = 150,
                        ):
        
        # Define Learning Criteria
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_var_loss = 5
        best_var_acc = 0
        valid_loss = 0        
        step_from_best = 0
        best_test = []
        print("\nStart Training and monitor on model {}...".format(early_stop_monitor))
        
        if warmup_epochs > 0:
            print("\nWARM UP training for {} epochs..".format(warmup_epochs))
            for epoch in range(warmup_epochs):
                start_time = time.time()
                train_loss, train_acc = self.train_func(self.data_train, batch_size, criterion, optimizer)
                valid_loss, valid_acc = self.test_func(self.data_valid, batch_size, criterion)

                secs = int(time.time() - start_time)

                print(f'[WARM UP] Epoch:{epoch+1:4d} ({secs:2d} sec)\t|\tLoss: {train_loss:.4f}(train)\t{valid_loss:.4f}(valid)\t|\tAcc: {train_acc * 100:.1f}%(train)\t{valid_acc * 100:.1f}%(valid)')

            print("WARMUP finished!")
        print("Training..")
        for epoch in range(epochs):

            start_time = time.time()
            train_loss, train_acc = self.train_func(self.data_train, batch_size, criterion, optimizer)
            valid_loss, valid_acc = self.test_func(self.data_valid, batch_size, criterion)

            secs = int(time.time() - start_time)

            print(f'Epoch:{warmup_epochs+epoch+1:4d} ({secs:2d} sec)\t|\tLoss: {train_loss:.4f}(train)\t{valid_loss:.4f}(valid)\t|\tAcc: {train_acc * 100:.1f}%(train)\t{valid_acc * 100:.1f}%(valid)')

            if early_stop_epochs > 0:
                if early_stop_monitor == "loss":
                    if valid_loss > best_var_loss:
                        step_from_best +=1
                        if step_from_best > early_stop_epochs:
                            print("Early Stop!")
                            step_from_best =0
                            break
                    else:
                        best_var_loss = valid_loss
                        step_from_best = 0
                        test_loss, test_acc = self.test_func(self.data_test, batch_size, criterion)
                        best_test.append((warmup_epochs+epoch + 1, test_loss.cpu().item(), test_acc))
                        
                elif early_stop_monitor == "accuracy":
                    if valid_acc < best_var_acc:
                        step_from_best +=1
                        if step_from_best > early_stop_epochs:
                            print("Early Stop!")
                            step_from_best =0
                            break
                    else:
                        best_var_acc = valid_acc
                        step_from_best = 0
                        test_loss, test_acc = self.test_func(self.data_test, batch_size, criterion)
                        best_test.append((warmup_epochs+epoch + 1, test_loss.cpu().item(), test_acc))
                        
        print('\n End Training. Checking the results of test dataset...')
        test_loss, test_acc = self.test_func(self.data_test, batch_size, criterion)
        print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')
        print("Top 5 accuracy of testing record is found at (epoch, loss, accuracy):")
        pprint(list(sorted(best_test, key=lambda epoch_loss_acc: epoch_loss_acc[2], reverse=True))[:5])
        
    def train_func(self, data_train, batch_size, criterion, optimizer):

        # Train the model
        train_loss = 0
        train_acc = 0
        data = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=8)
        self.model.train()
        for i, sampled_batch in enumerate(data):
            X = sampled_batch['X'].to(self.device)
            NX = sampled_batch['NX'].to(self.device)
            EW = sampled_batch['EW'].to(self.device)
            y = sampled_batch['y'].to(self.device)

            optimizer.zero_grad()

            output = self.model(X, NX, EW)

            loss = criterion(output, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_acc += (output.argmax(1) == y).sum().item()

        return train_loss / len(data_train), train_acc / len(data_train)

    def test_func(self, data_test, batch_size, criterion):
        loss = 0
        acc = 0
        data = DataLoader(data_test, batch_size=batch_size, num_workers=8)

        self.model.eval()
        for i, sampled_batch in enumerate(data):
            X = sampled_batch['X'].to(self.device)
            NX = sampled_batch['NX'].to(self.device)
            EW = sampled_batch['EW'].to(self.device)
            y = sampled_batch['y'].to(self.device)

            with torch.no_grad():
                output = self.model(X, NX, EW)
                loss = criterion(output, y)
                loss += loss.item()
                acc += (output.argmax(1) == y).sum().item()

        return loss / len(data_test), acc / len(data_test)

    def get_torch_model(self):
        return self.model


    

class TextLevelGNN(nn.Module):

    def __init__(self, num_nodes, node_feature_dim, class_num, embeddings=0, embedding_fix=False):
        super(TextLevelGNN, self).__init__()

        if type(embeddings) != int:
            print("\tConstruct pretrained embeddings")
            self.node_embedding = nn.Embedding.from_pretrained(embeddings, freeze=embedding_fix, padding_idx=0)
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
        NX [tensor](batch, sentence_maxlen, neighbor_distance*2): Neighbor nodes of each nodes in X
        EW [tensor](batch, sentence_maxlen, neighbor_distance*2): Neighbor weights of each nodes in X

        OUTPUT:
        -------
        y  [list] : Predicted Probabilities of each classes
        """
        ## Neighbor
        # Neighbor Messages (Mn)
        Mn = self.node_embedding(NX) # (BATCH, SEQ_LEN, NEIGHBOR_SIZE, EMBED_DIM)

        # EDGE WEIGHTS
        En = self.edge_weights(EW) # (BATCH, SEQ_LEN, NEIGHBOR_SIZE )

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



class TextLevelGNNDataset(Dataset):

    def __init__(self, data_pd, num_nodes, max_len=0, transform=None):
        """
        Args:
            num_nodes : number of nodes including padding

        """
        self.X = data_pd['X'].values
        self.NX = data_pd['X_Neighbors'].values
        self.y = data_pd['y'].values

        self.num_words = num_nodes-1

        if max_len==0:
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
        EW_idx[NX == 0] = 0 # set neighbor with PAD 0
        EW_idx[X==0] = 0 # set PAD node as 0

        y = torch.tensor(self.y[idx], dtype=torch.long)

        sample = {'X': X,
                  'NX': NX,
                  'EW': EW_idx,
                  'y': y}

#         if self.transform:
#             sample = self.transform(sample)

        return sample