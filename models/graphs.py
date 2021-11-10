
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


def graph_preprocess(graph_line, embedding):
	node1 = []
	node2 = []
	ip = []
	highest = 0
	for a in graph[:-1]:
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
				
	return ip, node1, node2


class Dataset_graph(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, files, zip, graph, tokenizer, embedding):
        'Initialization'
        self.zip = zip
        self.files = files
        self.tokenizer = tokenizer
	self.graph = graph 
	self.embedding = embedding

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample 
        file = self.zip.read(self.files[index]).decode("ISO-8859-1")
        train_file = return_train_single(file)
        tokenize_file = tokenize_single(train_file, self.tokenizer)
	graph_line = self.graph[index]       # Be careful with this one (Has to be changed for the type of data used)
	ip, node1, node2 = graph_preprocess(graph_line, self.embedding)
	
	graph_file(x=torch.stack(ip), edge_index=torch.tensor([node1, node2], dtype=torch.long)
        
        return tokenize_file, graph_file
