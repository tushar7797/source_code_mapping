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
import codecs
import collections
from itertools import chain
import os
import zipfile
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, BertTokenizer
import torch
from utils.utils import return_train, tokenize, pad_sents, load_train_files, return_train_single, tokenize_single, Dataset
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig



class model_gpt2(pl.LightningModule):
  def __init__(self, model, lr):
      super().__init__()
      self.model = model
      self.lr = lr
        
 #   def forward(self, x):
 #       return self.model(x)
  
  def configure_optimizers(self):
    
    params = list(self.model.parameters())
    optimizer = Adam(params, lr=self.lr)

  def training_step(self, batch, batch_idx): 
    padded_inputs, attention_masks = pad_sents(batch, tokenizer.pad_token_id) # to check 
    tensor_inputs = torch.tensor(padded_inputs).to('cuda')
    tensor_attention_masks = torch.tensor(attention_masks).to('cuda')
    batch_inputs, batch_attention = tensor_inputs, tensor_attention_masks
    loss = model(batch_inputs, labels = batch_inputs, attention_mask = batch_attention)[0]
      
      return loss
    
@hydra.main(config_path="config", config_name="gpt2")
def main(cfg: DictConfig):
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  model = GPT2LMHeadModel.from_pretrained('gpt2')
  special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
  tokenizer.add_special_tokens(special_tokens_dict)
  model.resize_token_embeddings(len(tokenizer))
  zip = zipfile.ZipFile(cfg.data_address)    #"/content/drive/My Drive/SemProjectFiles/data.tar.zip"
  train_files = load_train_files()
  
  params = {'batch_size': cfg.batch_size,
          'shuffle': True}
  
  training_set = Dataset(train_files, zip, tokenizer)
  training_generator = torch.utils.data.DataLoader(training_set, **params)
  
  loss_array = []
  max_epochs = 1
  model_train = model_gpt2(model, cfg.lr)
  
  # training
  trainer = pl.Trainer(gpus=4, precision=16, limit_train_batches=0.5)
  trainer.fit(model, train_loader)
      

if __name__ = main:
  main()
  
  
  
