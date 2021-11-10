import numpy as np
import json
import ast
from parse_python3 import parse_file
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, BertTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM
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
from utils.get_valid_ids import list_delete_items
#from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader
from torch.utils.data import Dataset
from torch import multiprocessing as mp
import time
#from torch.geometric.data import Batch
from torch_geometric.data import Batch
import matplotlib.pyplot as plt

def pad_sents(sents1, pad_token1 = 0):
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """
    sents_padded1 = sents1.copy()
    attention_masks = []

    max_length = 502
    """
    for i in range(len(sents1)):
        if len(sents1[i]) > max_length:
            max_length = len(sents1[i])
    print(max_length)
   """

    for i in range(len(sents1)):
        attention_mask = (np.zeros(len(sents1[i]))+1).tolist()
        for k in range(len(sents_padded1[i]),max_length):
            sents_padded1[i].append(pad_token1)
            attention_mask.append(0)
        attention_masks.append(attention_mask)

    return sents_padded1, attention_masks


def return_graph_attention(z, graph_batch, embedding_dim = 768, dev = 'cuda'):
  max_length = 0
  stack = torch.empty(1, embedding_dim).to(dev)

  for k in range(len(graph_batch['ptr'])):
    if k > 0:    
      if graph_batch['ptr'][k] - graph_batch['ptr'][k-1] > max_length:
        max_length = graph_batch['ptr'][k] - graph_batch['ptr'][k-1]

  mask = torch.empty(1, max_length).to(dev)

  for k in range(len(graph_batch['ptr'])):
    if k > 0:    
      a = z[graph_batch.ptr[k-1]:graph_batch.ptr[k]]
      b = torch.zeros(( max_length - graph_batch.ptr[k] + graph_batch.ptr[k-1]), embedding_dim).to(dev)
      c = torch.cat((a,b),0)
      stack = torch.cat((stack,c),0).to(dev)
      mask_length = graph_batch['ptr'][k] - graph_batch['ptr'][k-1]
      temp = torch.ones((mask_length,1))
      temp2 = torch.zeros((max_length - mask_length, 1))
      temp = torch.cat((temp, temp2), 0).to(dev)
      mask = torch.cat((mask, temp.view(1,-1)),0).to(dev)
  
  return_stack = stack[1:].view(len(graph_batch.ptr)-1,-1, embedding_dim)
  mask = mask[1:]#.view(len(graph_batch.ptr)-1,max_length)

  return return_stack, mask

def create_graph(ast, embedding, tokenizer, dev = 'cuda'):
    edge_head = []
    edge_tail = []
    node_emb = []
    node_id = 0
    for node in ast:
        node_type = torch.tensor(tokenizer(node['type'])['input_ids']).to(dev)
        node_type = torch.mean(embedding(node_type),0)
        if 'value' in node.keys():
            node_value = tokenizer(str(node['value']))['input_ids']
            if len(node_value) != 0:
                node_value = torch.tensor(tokenizer(str(node['value']))['input_ids']).to(dev)
                node_value = torch.mean(embedding(node_value),0)
            else:
                node_value = torch.zeros(embedding.embedding_dim).to(dev)
        else:
            node_value = torch.zeros(embedding.embedding_dim).to(dev)

        node_emb.append(torch.cat((node_type, node_value), 0))

        if 'children' in node.keys():
            for child in node['children']:
                edge_head.append(node_id)
                edge_head.append(child)
                edge_tail.append(child)
                edge_tail.append(node_id)
        node_id = node_id + 1

    graph = Data(x=torch.stack(node_emb), edge_index=torch.tensor([edge_head, edge_tail], dtype=torch.long))

    return graph


def create_labels(code_batch_tensors, attention_mask):
    rand = torch.rand(code_batch_tensor.shape)
    # where the random array is less than 0.15, we set true
    mask_arr = rand < 0.15* (code_batch_tensor != 101) * (code_batch_tensor != 102)
    arr = mask_arr.nonzero()
    labels = torch.clone(code_batch_tensor)
    input_tensors = torch.clone(code_batch_tensor)
    input_tensors[arr[:,0], arr[:,1]] = 103
    arr = attention_mask.nonzero()
    labels[arr[:,0], arr[:,1]] = -100

    return input_tensors, labels

def tokenize(training_file, tokenizer):
    temp = [101]
    for word in training_file:
            temp = temp + tokenizer(word, return_tensors="pt")['input_ids'][0][1:-1].tolist()
    temp.append(102)
    return temp

def create_cost_matrix(matrix_bert, matrix_graph, p = 2):
    batch, m, _ = matrix_bert.size()
    batch, n, _ = matrix_graph.size()
    cost_matrix = torch.cdist(matrix_bert, matrix_graph, p = p)
   # cost_matrix = torch.cosine_similarity(matrix_bert, matrix_graph)
    

    return cost_matrix


