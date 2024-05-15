import argparse
from ast import parse
import json
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

print("*"*50)
print("Start loading config")
print("*"*50)

# setSeed(random_seed)

listcity = ['charlotte', 'edinburgh', 'lasvegas', 'london', 'phoenix', 'pittsburgh', 'singapore']

parser = argparse.ArgumentParser()
parser.add_argument('--city', type=str, default='edinburgh', help=f'choose city{listcity}')
parser.add_argument('--checkKeyword', action='store_true', help='check bad keyword, uniqueness')
parser.add_argument('--showCheck', action='store_true', help='show check bad keyword, uniqueness')
parser.add_argument('--quantity', type=int, default=20, help='number of keyword retrieval')
parser.add_argument('--seed', type=int, default=1001, help='number of keyword retrieval')
parser.add_argument('--edgeType', type=str, default='IUF', help='future work, current using TF-IDF')

'''
Export args
'''
parser.add_argument('--logResult', type=str, default='./log', help='write log result detail')
parser.add_argument('--export2LLMs', action='store_true', help='whether export list of data for LLMs or not. \
                                                                default = False')

'''
Model args
'''
parser.add_argument('--RetModel', type=str, default='MPG_old', help='jaccard, MF, MVAE, CBR, MPG_old, MPG')
parser.add_argument('--numKW4FT', type=int, default=20, help='number of keyword for feature')

'''
CBR args
'''
parser.add_argument('--hidden_dim', type=int, default=32, help='hidden_dim')
parser.add_argument('--learning_rate', type=int, default=1e-3, help='learning_rate')
parser.add_argument('--num_epochs', type=int, default=1, help='num_epochs')

args = parser.parse_args()

config = {}
config['city'] = args.city
config['checkKeyword'] = args.checkKeyword
config['showCheck']= args.showCheck
config['quantity'] = args.quantity
config['seed'] = args.seed
config['edgeType']  = args.edgeType
config['logResult'] = args.logResult
config['export2LLMs'] = args.export2LLMs
config['RetModel'] = args.RetModel
config['numKW4FT'] = args.numKW4FT

modelConfig = {}
modelConfig['hidden_dim'] = args.hidden_dim
modelConfig['learning_rate'] = args.learning_rate
modelConfig['num_epochs'] = args.num_epochs

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")

