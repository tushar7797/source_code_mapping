import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch
import logging
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import nltk
import torch.optim as optim
import codecs
import collections
import time
from itertools import chain

import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch
import logging
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import nltk
import codecs
import collections
from itertools import chain
import os
import zipfile


from torch_geometric.data import Data
from torch_scatter import scatter_mean
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn as nn
from torch.optim import Adam
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GatedGraphConv, GAE, VGAE
import sys
import os
dev = "cuda"
import sys
import os
#sys.path.append(os.path.abspath('/content/drive/My Drive/SemProject'))


from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, BertTokenizer
import torch

import copy

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
    sents_padded1 = copy.deepcopy(sents1)
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


    ### END YOUR CODE

    return sents_padded1, attention_masks

def return_past(z, graph_batch):
  max_length = 0
  stack = torch.empty(1, embedding.embedding_dim).to('cuda')
  for k in range(len(graph_batch['ptr'])):
    if k > 0:    
      if graph_batch['ptr'][k] - graph_batch['ptr'][k-1] > max_length:
        max_length = graph_batch['ptr'][k] - graph_batch['ptr'][k-1]

  for k in range(len(graph_batch['ptr'])):
    if k > 0:    
      a = z[graph_batch.ptr[k-1]:graph_batch.ptr[k]]
      b = torch.zeros(( max_length - graph_batch.ptr[k] + graph_batch.ptr[k-1]), embedding.embedding_dim).to('cuda')
      c = torch.cat((a,b),0)
      stack = torch.cat((stack,c),0).to('cuda')

  return_stack = stack[1:].view(len(graph_batch.ptr)-1,-1,embedding.embedding_dim)

  return return_stack

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(2*embedding.embedding_dim,embedding.embedding_dim)
        self.conv1 = GCNConv(embedding.embedding_dim, embedding.embedding_dim)
        self.conv2 = GatedGraphConv(embedding.embedding_dim, 8)
        #self.linear2 = nn.Linear(embedding.embedding_dim, 32)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.linear(x)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
				
        return  x #F.log_softmax(mean, dim=1)




if__name__ = 'main':
	graphs = open('/content/drive/My Drive/SemProject/programs_eval.json',)
	zip = zipfile.ZipFile("/content/drive/My Drive/SemProjectFiles/data.tar.zip")
	file1 = open("/content/drive/My Drive/SemProjectFiles/programs_eval.txt",encoding="utf8") 
	files = []

	count = 0
	for g in file1:
		temp = nltk.word_tokenize(g)
		files.append(temp)
		count = count + 1
	
	net = Net()
	model_graph = GAE(net)
	embedding = embedding.to('cuda')
	model_graph = model_graph.to('cuda')
	model = model.to('cuda')
	optimizer = Adam(list(model.parameters()) + list(embedding.parameters()) + list(model_graph.parameters()), lr=0.00005)

	text_inputs = []
	lengths = []
	count = 1
	graph_data = []
	batch_size = 8
	#net = net.to('cuda')
	model_graph = model_graph.to('cuda')
	#model = model.to('cuda')
	loss_graph_array = []
	loss_text_array = []
	loss_cross_array = []
	import time

	with open('/content/drive/My Drive/SemProject/programs_eval.json', encoding = "ISO-8859-1") as f:
		for line in f:
			if files[count-1][0] in zip.namelist():
				if count%30000 == 0:
					break
				if count % 500 == 1:
					start = time.time()
				file = zip.read(files[count-1][0]).decode("ISO-8859-1")
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
				k = tokenizer(temp, return_tensors="pt")
				temp = k['input_ids'][0]
				temp = temp[temp!=198]
				temp = temp[temp!=220]
				temp = temp[temp!=197]
				temp2 = [tokenizer.bos_token_id] + temp.tolist() + [tokenizer.eos_token_id]
				if len(temp2) < 255:
					text_inputs.append(temp2)
					graph_data.append(json.loads(line))
			count = count+1

			if count % 500 == 0:
				padded_inputs, attention_masks = pad_sents(text_inputs, tokenizer.pad_token_id)
				padded_text = torch.tensor(padded_inputs).to('cuda')
				attention_masks = torch.tensor(attention_masks).to('cuda')
				edges = []  
				ips = []
				batches = []
				high = []
				for i in range(len(graph_data)):
					node1 = []
					node2 = []
					ip = []
					highest = 0
					for a in graph_data[i][:-1]:
						temp_rand = torch.tensor(tokenizer(a['type'])['input_ids']).to('cuda')
						temp = torch.mean(embedding(temp_rand),0)
						if 'value' in a.keys():
							temp_token = tokenizer(str(a['value']))['input_ids']
							if len(temp_token) != 0:
								temp_token2 = torch.tensor(temp_token).to('cuda')
								temp2 = torch.mean(embedding(temp_token2),0)
							else:
								temp2 = torch.zeros(embedding.embedding_dim).to('cuda')           
						else:
							temp2 = torch.zeros(embedding.embedding_dim).to('cuda')
						ip.append(torch.cat((temp, temp2), 0))

						if highest < a['id']:
							highest = a['id']
						if 'children' in a.keys():
							for y in a['children']:
								node1.append(a['id'])
								node1.append(y)
								node2.append(y)
								node2.append(a['id'])
					high.append(highest)
					#zeros = torch.zeros(2*embedding.embedding_dim).to('cuda')
					#ip.append(zeros)
					#for k in range(highest+1):
					#  node1.append(k)
					#  node2.append(highest+1)        
					batches.append(Data(x=torch.stack(ip), edge_index=torch.tensor([node1, node2], dtype=torch.long)))

				graph_batches = DataLoader(batches, batch_size=batch_size)
				count_batch = 0
				for graph_batch in graph_batches:
					optimizer.zero_grad()
					text_batch =  padded_text[count_batch : min(count_batch + batch_size, len(graph_data))]
					attention_batch = attention_masks[count_batch:count_batch+batch_size]
					graph_batch = graph_batch.to('cuda')
					output_graph = model_graph.encode(graph_batch)

					past = return_past(output_graph, graph_batch)

					output_text = model(text_batch, past = past, labels=text_batch, attention_mask = attention_batch)

					loss_text = output_text[0]
					loss_graph = model_graph.recon_loss(output_graph, graph_batch['edge_index'])

					loss = loss_text + loss_graph

					loss_text_array.append(loss_text.detach())
					loss_graph_array.append(loss_graph.detach())

					loss.backward()
					optimizer.step()
					count_batch = count_batch + batch_size
				print(time.time()-start)
				text_inputs = []
				graph_data = []
				print(loss_text, loss_graph)

	
	
	
