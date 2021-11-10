import json
import ast
from parse_python3 import parse_file
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, BertTokenizer
import nltk
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data
from torch_scatter import scatter_mean
from torch_geometric.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GatedGraphConv, GAE, VGAE
import numpy as np

def tokenize(training_file, tokenizer):
    temp = []
    for word in training_file:
            temp = temp + tokenizer(word, return_tensors="pt")['input_ids'][0][1:-1].tolist()
    return temp


def list_code(data ,src_len, tokenizer):
    delete_list = []
    for i in range(len(data)):
        temp = json.loads(data[i])
        tokens = tokenize(temp['code_tokens'], tokenizer)
        if len(tokens) > src_len:
            delete_list.append(i)
        if i%1000 == 0:
             print(i)

    return delete_list



def list_lang(data ,src_len, tokenizer):
    delete_list = []
    for i in range(len(data)):
        temp = json.loads(data[i])
        tokens = tokenize(temp['docstring_tokens'], tokenizer)
        if len(tokens) > src_len:
            delete_list.append(i)
        if i%1000 == 0:
             print(i)

    return delete_list


def list_graph(data, ast_len, ast_min = 5):
    delete_list = []
    for i in range(len(data)):
        temp = json.loads(data[i])
        if isinstance(temp, list):
            if len(temp) > ast_len or len(temp)<ast_min:
                delete_list.append(i)
        else:
            delete_list.append(i)
        if i%1000 == 0:
            print(i)

    return delete_list

def list_delete_items(data_graph, data_code, src_len, ast_len, tokenizer, ast_min = 5):
    delete_code = list_code(data_code, src_len, tokenizer)
    delete_graph = list_graph(data_graph, ast_len, ast_min)
    delete_lang = list_lang(data_code, src_len, tokenizer) 
    temp = delete_code + list(set(delete_graph) - set(delete_code))
    temp = delete_lang + list(set(temp) - set(delete_lang))
    return temp





