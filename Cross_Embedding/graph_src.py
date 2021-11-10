import json
import ast
from parse_python3 import parse_file
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, BertTokenizer
#from transformers.models.bert.modeling_bert import BertForMaskedLM
from models.base_classes import BertForMaskedLM, Sinkhorn
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
from models.utils import pad_sents, create_graph, return_graph_attention, create_cost_matrix
import pytorch_lightning as pl

from torch_geometric.data import Data
from torch_scatter import scatter_mean
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from utils.utils import pad_sents, create_graph, return_graph_attention, create_cost_matrix


class graph_src_bert(pl.LightningModule):
  def __init__(self, model_graph, model_src, lr):
        super().__init__()
        self.model_graph = model_graph
        self.model_src = model_src
	      self.lr = lr

  
  def configure_optimizers(self):
    params = list(model_graph.parameters()) + list(model_src.parameters())
    optimizer = Adam(params, lr=self.lr)
    
  return optimizer

    def training_step(self, batch, batch_idx):
      
      graph_batch, code_batch = batch
      code_batch_tensor, attention_mask = pad_sents(code_batch)
      code_batch_tensor = torch.tensor(code_batch_tensor)
      attention_mask_code = torch.tensor(attention_mask)
      code_batch_tensor, labels_code = create_labels(code_batch_tensor, attention_mask_code)
      code_batch_tensor = code_batch_tensor.to('cuda')
      attention_mask_code = attention_mask_code.to('cuda')
      labels_code = labels_code.to('cuda')
      
      graph_batch = Batch.from_data_list(graph_batch).to('cuda')
      output_graph = model_graph.encode(graph_batch)
      loss_graph = model_graph.recon_loss(output_graph, graph_batch['edge_index'])
      graph_bert, mask_graph = return_graph_attention(output_graph, graph_batch, dev = 'cuda')
      
      outputs_lang, cross_lang = model_lang(input_ids = doc_batch_tensor, attention_mask = attention_mask_doc, labels = la
bels_doc)
      
      cost_matrix = torch.cdist(cross_src, graph_bert, p = 2)
      cost = Sinkhorn.apply(cost_matrix, attention_mask_code, mask_graph)
      
      loss_src = outputs_lang.loss
      loss = loss_src + cost
      return loss
