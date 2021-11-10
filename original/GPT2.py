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


def return_train(files, zip):
  training_file = []
  lengths = []
  count = 0
  # To change the range to 1000
  for i in range(100):
    if (count%1000 == 0):
      print(count)
    if files[i][0] in zip.namelist():
      file = zip.read(files[i][0]).decode("ISO-8859-1")
      temp2 = []
      temp = ""
      flag1 = 0
      flag2 = 0
      for j in range(len(file)):
        if file[j:j+2] == '//':
          flag1 = 1
        if file[j:j+2] == '/*':
          flag2 = 1
        if file[j:j+1] == '\n':
          flag1 = 0
        if file[j-1:j+1] == "*/":
          flag2 = 0

        if flag1 == 0 and flag2 == 0:
          temp2.append(file[j])
         
      temp = temp.join(temp2)
      
    training_file.append(temp)
    lengths.append(len(temp))
    count = count + 1
  return training_file
    
def tokenize(training_file, tokenizer)
  tokenizer.pad_token = '<PAD>'
  lengths = []
  inputs = []
  count = 0
  #dict_counts = dict()
  for x in training_file:
    k = tokenizer(x, return_tensors="pt")
    temp = k['input_ids'][0]
    temp = temp[temp!=198]
    temp = temp[temp!=220]
    temp = temp[temp!=197]
    temp2 = [tokenizer.bos_token_id] + temp.tolist() + [tokenizer.eos_token_id]
    if len(temp2) < 255:
      inputs.append(temp2)
    return inputs   

def pad_sents(sents1, pad_token1):
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
    
    max_length = 0
    for i in range(len(sents1)):
        if len(sents1[i]) > max_length:
            max_length = len(sents1[i]) 

    for i in range(len(sents1)):
        attention_mask = (np.zeros(len(sents1[i]))+1).tolist()
        for k in range(len(sents_padded1[i]),max_length):
            sents_padded1[i].append(pad_token1)
            attention_mask.append(0)
        attention_masks.append(attention_mask)


    return sents_padded1, attention_masks



if __name__ = main:
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  model = GPT2LMHeadModel.from_pretrained('gpt2')
  special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
  tokenizer.add_special_tokens(special_tokens_dict)
  model.resize_token_embeddings(len(tokenizer))
  zip = zipfile.ZipFile("/content/drive/My Drive/SemProjectFiles/data.tar.zip")
  file1 = open("/content/drive/My Drive/SemProjectFiles/programs_training.txt",encoding="utf8") 
  files = []
  count = 0
  for g in file1:
    temp = nltk.word_tokenize(g)
    files.append(temp)
  #training_file = return_train(files)
  #inputs = tokenizer(training_file, tokenizer)
  #padded_inputs, attention_masks = pad_sents(inputs, tokenizer.pad_token_id)
  
  file_size  = 1000
  n_epochs = 1# or whatever
  batch_size = 8
  loss_array = []
  import time
  model = model.to('cuda')
  for k in range(0,len(files), file_size):
    start = time.time()
    print(k)
    training_file = return_train(files[k:k+file_size], zip)
    inputs = tokenizer(training_file, tokenizer)
    padded_inputs, attention_masks = pad_sents(inputs, tokenizer.pad_token_id)
    tensor_inputs = torch.tensor(padded_inputs).to('cuda')
    tensor_attention_masks = torch.tensor(attention_masks).to('cuda')
    permutation = torch.randperm(tensor_inputs.size()[0])

    for i in range(0,tensor_inputs.size()[0], batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i+batch_size]
        batch_inputs, batch_attention = tensor_inputs[indices], tensor_attention_masks[indices]
        loss = model(batch_inputs, labels = batch_inputs, attention_mask = batch_attention)[0]
        loss.backward()
        optimizer.step()
        loss_array.append(loss.detach())

    print(loss)
    print(time.time() - start)



