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
import time

# Install trl
from trl.core import (logprobs_from_logits,
                         whiten,
                         clip_by_value,
                         entropy_from_logits,
                         flatten_dict,
                         average_torch_dicts,
                         stats_to_np,
                         stack_dicts,
                         add_suffix)
                         
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
import torch
import collections
import time
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model, GPT2PreTrainedModel
from transformers import top_k_top_p_filtering
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import torch
from src.utils import return_train, tokenize, pad_sents, load_train_files, return_train_single, tokenize_single, Dataset
from src.ppo import ValueHead, GPT2HeadWithValueModel, respond_to_batch, AdaptiveKLController, FixedKLController, PPOTrainer
        
if __name__ == '__main__':
  gpt2_model = GPT2HeadWithValueModel.from_pretrained('gpt2')
  gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('gpt2')
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
  tokenizer.add_special_tokens(special_tokens_dict)
  gpt2_model_ref.resize_token_embeddings(len(tokenizer))
  gpt2_model.resize_token_embeddings(len(tokenizer))
  
  zip = zipfile.ZipFile("/content/drive/My Drive/SemProjectFiles/data.tar.zip")
  train_files = load_train_files()
    
  n_epochs = 1# or whatever
  batch_size = 8
  loss_array = []
  file_size = 256
  
  #model = model.to('cuda')
  inputs = []
  rewards = []
  kl_div = []
  batch_size = 8
  loss_array = []
  file_size = 256
  
  params = {'batch_size': 256,
          'shuffle': True,
          'num_workers': 2}
  
  training_set = Dataset(train_files, zip, tokenizer)
  training_generator = torch.utils.data.DataLoader(training_set, **params)
  
  for epoch in range(max_epochs):
    # Training
    for batch in training_generator:
      padded_inputs, attention_masks = pad_sents(batch, tokenizer.pad_token_id) # to check 
      tensor_inputs = torch.tensor(padded_inputs).to('cuda')
      tensor_attention_masks = torch.tensor(attention_masks).to('cuda')
      query_tensors = padded_inputs[:,:30]
      response = []
      scores = []
      
      optimizer.zero_grad()
      batch_inputs, batch_attention = tensor_inputs, tensor_attention_masks
      loss = model(batch_inputs, labels = batch_inputs, attention_mask = batch_attention)[0]
      loss.backward()
      optimizer.step()
      loss_array.append(loss.detach())
      
      for i in range(int(256/4)):
        response_tensors = respond_to_batch(gpt2_model, query_tensors[i*4:(i+1)*4], txt_len=15, top_k=0, top_p=1.0)
        score = bleu(padded_inputs[:,:45][i*4:(i+1)*4], response_tensors, 15)
        response.append(response_tensors)
        scores.append(score*20)
      
    response_tensors = torch.cat(response, dim = 0)
    scores_tensors = torch.cat(scores, dim = 0)
    scores_tensors = scores_tensors.to('cuda')
    stats = ppo.step(query_tensors, response_tensors, scores_tensors)
    print(stats['objective/kl'], stats['objective/kl_coef'], stats['objective/entropy'], torch.mean(scores_tensors)*5)
    print(time.time() - start)
    kl_div.append(stats['objective/kl'].item())
    rewards.append((torch.mean(scores_tensors)*5).item())


