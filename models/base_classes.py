from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertOnlyMLMHead
import torch
import torch.nn as nn
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import MaskedLMOutput

class Sinkhorn(torch.autograd.Function):
    def forward(ctx, cost_matrix, mask_tensor1, mask_tensor2,  sinkhorn_reg = 0.5, n_iter = 50, dev = 'cuda'):
        batch, n, m = cost_matrix.shape
        sum_tensor1 = torch.sum(mask_tensor1, 1)
        a = mask_tensor1/sum_tensor1.view(batch, 1).to(dev)
        b = mask_tensor2/torch.sum(mask_tensor2).to(dev)#.view(batch, 1))
        k = torch.exp(-cost_matrix/sinkhorn_reg).double().to(dev)
        u = torch.ones((batch, n), requires_grad = True).double().to(dev)
        v = torch.ones((batch, m), requires_grad = True).double().to(dev)
       # print(a.size(), torch.matmul(k, v.view(batch, -1, 1)).size())

       # print(k[1,0:10, 0:10])

        for i in range(30):
            u = torch.divide(a, torch.matmul(k, v.view(batch, -1, 1)).view(batch,-1))
            v = torch.divide(b, torch.matmul(torch.transpose(k, 1, 2), u.view(batch, -1, 1)).view(batch,-1))

        T = torch.matmul(k, torch.diag_embed(v)).to(dev)
        T = torch.matmul(torch.diag_embed(u), T).to(dev)
        ctx.save_for_backward(T)

        return torch.sum(cost_matrix*T)  # , dim = [1,2])

    def backward(ctx, grad_output):
        #print(grad_output.size())
        (T,) = ctx.saved_tensors
        #print(T.size())
        return T, None, None#[:, None, None]


class BertForMaskedLM(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        #config.hidden_size = config.hidden_size + 12
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.embed_dim = config.hidden_size
        #config.hidden_size = config.hidden_size + 1
        self.init_weights()
        self.Q = nn.Linear(self.embed_dim, self.embed_dim)
        self.K = nn.Linear(self.embed_dim, self.embed_dim)
        self.V = nn.Linear(self.embed_dim, self.embed_dim)
       

        self.linear = nn.Linear(self.embed_dim, 1)
        self.tanh = nn.Tanh()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        graph_attention = None,
        return_embed = False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        if graph_attention is not None:
            Q_temp = self.Q(graph_attention)
            K_temp = self.K(sequence_output)
            V_temp = self.V(graph_attention)
            
            temp2 = torch.matmul(K_temp, torch.transpose(Q_temp, 1, 2))/np.sqrt(self.embed_dim)
            
            att_temp = torch.matmul(nn.Softmax(dim=2)(temp2), V_temp)
            
            #sequence_output = sequence_output + att_temp

        l2 = (torch.norm(sequence_output, dim = 2)).double()+ 1e-8
        #cross_head = torch.div(sequence_output, l2.view(sequence_output.size[0], -1, 1))
        sequence_output = sequence_output/l2.view(sequence_output.size()[0], -1, 1).double()
        gate = self.tanh(l2).double()
        sequence_output = sequence_output*gate.view((sequence_output.size()[0],-1,1)).float()
        #print(sequence_outputs.size())    

        if return_embed:
            return sequence_output

        prediction_scores = self.cls(sequence_output.float())

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            #l2 = (torch.norm(sequence_output, dim = 2) - 1).double()
           # penalty_loss = torch.sum(l2)
           # penalty_loss = torch.sum(torch.where(l2>0, l2, float(0))) 

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output


        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), sequence_output

    
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(2*embedding.embedding_dim,embedding.embedding_dim)
        self.number_layers = 30
        self.convs = nn.ModuleList()
        for i in range(self.number_layers):
            self.convs.append(GCNConv(embedding.embedding_dim, embedding.embedding_dim))

        #self.conv1 = GCNConv(embedding.embedding_dim, embedding.embedding_dim)
        self.conv2 = GatedGraphConv(embedding.embedding_dim, 8)
        #self.final_linear = nn.Linear(embedding.embedding_dim, 1)
        self.gating_func = torch.nn.Tanh()
        #self.linear2 = nn.Linear(embedding.embedding_dim, 32)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        #for i in range(len(x)):
        #  x[i] = self.linear(x[i])

        #self.linear(x)
        x = self.linear(x)
        x = F.relu(x)
        for i in range(self.number_layers):
            x = self.convs[i](x, edge_index)
            F.relu(x)
        x = self.conv2(x, edge_index)
        l2 = (torch.norm(x, dim = 1)).double()+ 1e-8
        gate = self.gating_func(l2)
        cross_graph = torch.div(x, l2.view(-1,1))
        return x
    
    
