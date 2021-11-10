import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
#from pytorch_transformers import BertTokenizer, BertConfig
#from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
import logging
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import nltk
import torch.optim as optim
from torch.optim import Adam
import codecs
import collections
from itertools import chain
import os
import zipfile
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, BertTokenizer
import torch
from utils.utils import return_train, tokenize, pad_sents, load_train_files, return_train_single, tokenize_single
from utils.graph import Net, return_past, graph_preprocess, Dataset_graph

from torch_geometric.data import Data
from torch_scatter import scatter_mean
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import pytorch_lightning as pl


class graph_gpt2(pl.LightningModule):
  def __init__(self, model, model_graph, lr):
        super().__init__()
        self.model = model
        self.model_graph = model_graph
	self.lr = lr
 #   def forward(self, x):
 #       return self.model(x)

  
  def configure_optimizers(self):
    
    params = list(model.parameters()) + list(embedding.parameters()) + list(model_graph.parameters())
    optimizer = Adam(params, lr=self.lr)
    
  return optimizer

    def training_step(self, batch, batch_idx):
      
      text_batch, graph_batch = batch
      padded_inputs, attention_masks = pad_sents(text_batch, tokenizer.pad_token_id) # to check 
      tensor_inputs = torch.tensor(padded_inputs).to('cuda')
      tensor_attention_masks = torch.tensor(attention_masks).to('cuda')
      
      
      output_graph = self.model_graph.encode(graph_batch)
      past = return_past(output_graph, graph_batch)
      output_text = self.model(text_batch, past = past, labels=text_batch, attention_mask = attention_batch)
      loss_text = output_text[0]
			loss_graph = model_graph.recon_loss(output_graph, graph_batch['edge_index'])
      loss = loss_text + loss_graph
      
      return loss

@hydra.main(config_path="config", config_name="gpt2_graph")
def main(cfg: DictConfig):
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  model = GPT2LMHeadModel.from_pretrained('gpt2')
  special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
  tokenizer.add_special_tokens(special_tokens_dict)
  model.resize_token_embeddings(len(tokenizer))
  zip = zipfile.ZipFile(cfg.data_address)    #"/content/drive/My Drive/SemProjectFiles/data.tar.zip"
  train_files = load_train_files()
  embedding = nn.Embedding(len(tokenizer), cfg.embedding_dim) #768
  net = Net()
  model_graph = GAE(net)

  graphs = open(cfg.graph_address,) #'/content/drive/My Drive/SemProject/programs_eval.json'

  params = {'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': 1}

  training_set = Dataset(train_files, zip, graphs, tokenizer, embedding)
  training_generator = torch.geometric.data.DataLoader(training_set, **params)  # Might be wrong, please check

  loss_array = []
  max_epochs = 1
  # Loop over epochs
  model_train = graph_gpt2(model, model_graph, cfg.lr)
  
 # training
 trainer = pl.Trainer(gpus=4, precision=16, limit_train_batches=0.5)
 trainer.fit(model_train, training_generator)	
	

if__name__ = 'main':
  ma







