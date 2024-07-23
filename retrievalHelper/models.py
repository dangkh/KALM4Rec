from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_
import torch.optim as optim
import torch
import json
import numpy as np


class CBR(nn.Module):
    def __init__(self, dim_users, dim_items, embedding_dim):
        super(CBR, self).__init__()
        self.user_embeddings = nn.Linear(dim_users, embedding_dim)
        self.uAP = AttentionPooling(embedding_dim , embedding_dim // 4)
        self.item_embeddings = nn.Linear(dim_items, embedding_dim)
        self.iAP = AttentionPooling(embedding_dim , embedding_dim // 4)
        self.dropout = nn.Dropout(0.2)
        self.activate = nn.Sigmoid()
        # initial weight
        self.apply(xavier_normal_initialization)

    def forward(self, user_ft, item_ft):
        numBatch = len(user_ft)
        numRest = len(item_ft)
        user_embeddings = self.user_embeddings(user_ft)
        # user_embeddings = self.r1(user_embeddings)
        user_embeddings = self.dropout(user_embeddings)
        user_embeddings, _ = self.uAP(user_embeddings)
        item_embeddings = self.item_embeddings(item_ft)
        item_embeddings = self.dropout(item_embeddings)
        # item_embeddings = self.r2(item_embeddings)
        item_embeddings, _ = self.iAP(item_embeddings)
        item_embeddings = torch.permute(item_embeddings, (1, 0))
        pred = self.activate(torch.matmul(user_embeddings, item_embeddings))
        # using this to return 
        return pred   


    def prediction(self, user_ft, item_ft):
        return self.forward(user_ft, item_ft)

class AttentionPooling(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionPooling, self).__init__()

        # Linear layers for attention scoring
        self.V = nn.Linear(input_size, hidden_size)
        self.size = input_size
        self.w = nn.Linear(hidden_size, 1)
        self.tanh = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_features):
        # Calculate attention scores
        scores = self.tanh(self.V(input_features)) 
        scores = self.w(scores)
        
        # Apply softmax to get attention weights
        weights = self.softmax(scores)

        # Apply attention weights to input features
        pooled_features = torch.sum(weights * input_features, dim=1)

        return pooled_features, weights

class MFBPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num):
        super(MFBPR, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """     
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)

        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1)
        return prediction_i, prediction_j            


    def prediction(self, user, item):
        user = self.embed_user(user)
        item_i = self.embed_item(item)
        return (user * item_i).sum(dim=-1)

    def csPrediction(self, top3Users, item):
        tmpU = []
        for uid in top3Users:
            sc = self.prediction(uid, item[0])
            tmpU.append(sc.item())
        result = np.mean(np.asarray(tmpU))
        return result


def xavier_normal_initialization(module):
    r""" using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.
    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_
    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)


class JaccardSim(object):
    """docstring for JaccardSim"""
    def __init__(self, path, quantity):
        super(JaccardSim, self).__init__()
        self.path = path
        self.quantity = quantity
        f = open(path)
        keywordScore = json.load(f)
        self.rest_kw = {}
        self.l_rest = []
        for rest in keywordScore:
            self.rest_kw[rest] = []
            lw = keywordScore[rest]
            self.l_rest.append(rest)
            for kw, sc in lw:
                self.rest_kw[rest].append(kw)
        # read TFIUF contain rest: kw

        
    def pred(self, userkwList):
        sc = []
        for rest in self.rest_kw:
            sc.append(self.jscore(self.rest_kw[rest], userkwList))
        idxrest = np.argsort(sc)[::-1]
        result = [self.l_rest[x] for x in idxrest[:self.quantity]]
        return result

    def jscore(self, l1, l2):
        l1 = set(l1)
        l2 = set(l2)
        i = l1.intersection(l2)
        u = l1.union(l2)
        return len(i) / len(u)