import random
import time
import torch
import random
import numpy as np
from torch.utils.data import Dataset
import os
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")


def quick_eval(preds, gt, source = None):
    '''
    - preds: [list of restaurants]
    - GT: [('wrdLrTcHXlL4UsiYn3cgKQ', 4.0), ('uG59lRC-9fwt64TCUHnuKA', 3.0)]
    - 
    '''
    gt_list = set([a[0] for a in gt])
    preds_list = list(set(preds))
    ov = gt_list.intersection(preds_list)
    prec = len(ov)/len(preds_list)
    rec = len(ov)/len(gt_list)
    f1 = 0 if prec+rec == 0 else 2*prec*rec/(prec+rec)
    # if source != None :
    #     print("Precision: {}, Recall: {}, F1: {}".format(prec, rec, f1), file = source)
    return prec, rec, f1

class DataCF(Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data).type(torch.FloatTensor)
        self.label = torch.from_numpy(label).type(torch.FloatTensor)

    def __getitem__(self, index):
        '''
        return data and label
        '''
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

def evaluateModel(model, data_loader, rest_train, groundtruth, users, numRetrieval, rest_Label, getPred = False):
    model.eval()
    listPred = []
    for batch_idx, (data, label) in enumerate(data_loader):
        data = data.to(device)
        label = label.to(device)
        predictions = model.prediction(data, rest_train)
        listPred.append(predictions)
    listPred = torch.vstack(listPred)
    listPred = listPred.detach().cpu().numpy()
    if getPred:
        return evaluate2pred(users, groundtruth, listPred, numRetrieval, rest_Label)
    return evaluate(users, groundtruth, listPred, numRetrieval, rest_Label)

def evaluate2pred(users, groundtruth, listPred, numRetrieval, rest_Label):
    lResults = []
    for idx in range(len(users)):
        if idx > 3000:
            break
        testUser = users[idx]
        groundtruthUser = groundtruth[testUser]
        pred = listPred[idx]
        tmp = np.argsort(pred)[::-1][:numRetrieval]
        restPred = [rest_Label[x] for x in tmp]
        score = quick_eval(restPred, groundtruthUser)
        lResults.append(restPred)    
    return lResults

def evaluate(users, groundtruth, listPred, numRetrieval, rest_Label):
    lResults = []
    for idx in range(len(users)):
        if idx > 3000:
            break
        testUser = users[idx]
        groundtruthUser = groundtruth[testUser]
        pred = listPred[idx]
        tmp = np.argsort(pred)[::-1][:numRetrieval]
        restPred = [rest_Label[x] for x in tmp]
        score = quick_eval(restPred, groundtruthUser)
        lResults.append(score)    
    return lResults    