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
from models.utils import pad_sents, create_graph, return_graph_attention, create_cost_matrix, tokenize
import pytorch_lightning as pl


class lang_src_bert(pl.LightningModule):
  def __init__(self, model_lang, model_src, lr):
        super().__init__()
        self.model_lang = model_lang
        self.model_src = model_src
	self.lr = lr
 #   def forward(self, x):
 #       return self.model(x)

  
  def configure_optimizers(self):
    
    params = list(model_lang.parameters()) + list(model_src.parameters())
    optimizer = Adam(params, lr=self.lr)
    
  return optimizer

    def training_step(self, batch, batch_idx):
      
      code_batch, doc_batch = batch
      code_batch_tensor, attention_mask = pad_sents(code_batch)
     code_batch_tensor = torch.tensor(code_batch_tensor)
      attention_mask_code = torch.tensor(attention_mask)
      code_batch_tensor, labels_code = create_labels(code_batch_tensor, attention_mask_code)
      doc_batch_tensor, attention_mask_doc = pad_sents(doc_batch)
      doc_batch_tensor = torch.tensor(doc_batch_tensor)
      attention_mask_doc = torch.tensor(attention_mask_doc)
      doc_batch_tensor, labels_doc = create_labels(doc_batch_tensor, attention_mask_doc)
      code_batch_tensor = code_batch_tensor.to('cuda')
      attention_mask_code = attention_mask_code.to('cuda')
      labels_code = labels_code.to('cuda')
      doc_batch_tensor = doc_batch_tensor.to('cuda')
      attention_mask_doc = attention_mask_doc.to('cuda')
      labels_doc = labels_doc.to('cuda')
      
      outputs_lang, cross_lang = model_lang(input_ids = doc_batch_tensor, attention_mask = attention_mask_doc, labels = la
bels_doc)
      outputs_src, cross_src = model_src(input_ids = code_batch_tensor, attention_mask = attention_mask_code, labels = labels_code)
      
      cost_matrix = torch.cdist(cross_lang, cross_src, p = 2)
      cost = Sinkhorn.apply(cost_matrix, attention_mask_doc, attention_mask_code)
      
      loss = loss_src + loss_lang + cost
      return loss


@hydra.main(config_path="config", config_name="lang_src_bert")
def main(cfg: DictConfig):
  tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
  model_src = BertForMaskedLM.from_pretrained('bert-base-cased')
  model_lang = BertForMaskedLM.from_pretrained('bert-base-cased')
  params = {'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': 1}
  train_files = load_train_files()
  training_set = Dataset(train_files, zip, graphs, tokenizer, embedding)
  training_generator = torch.utils.data.DataLoader(training_set, **params)

  loss_array = []
  max_epochs = 1
  # Loop over epochs
  model_train = lang_src_bert(model_src, model_graph, cfg.lr)
 # training
  trainer = pl.Trainer(gpus=4, precision=16, limit_train_batches=0.5)
  trainer.fit(model_train, training_generator)
	
if __name__ = main:
  main()
