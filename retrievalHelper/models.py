from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_, xavier_uniform_
import torch.optim as optim
import torch


class MatrixFactorization(nn.Module):
    def __init__(self, dim_users, dim_items, embedding_dim):
        super(MatrixFactorization, self).__init__()
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