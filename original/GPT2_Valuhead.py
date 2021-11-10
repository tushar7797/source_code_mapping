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
  
  class ValueHead(nn.Module):
    """The ValueHead class implements a head for GPT2 that returns a scalar for each output token."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.detach_head = False
        self.summary_type = config.summary_type if hasattr(config, "summary_type") else "last"
        if self.summary_type == "attn":
            raise NotImplementedError

        self.summary = Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if hasattr(config, "summary_proj_to_labels") and config.summary_proj_to_labels and config.num_labels > 0:
                num_classes = config.num_labels
            else:
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)

        self.activation = Identity()
        if hasattr(config, "summary_activation") and config.summary_activation == "tanh":
            self.activation = nn.Tanh()

        self.first_dropout = Identity()
        if hasattr(config, "summary_first_dropout") and config.summary_first_dropout > 0:
            self.first_dropout = nn.Dropout(config.summary_first_dropout)

        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)
            
        self.flatten = nn.Flatten()

    def forward(self, hidden_states, cls_index=None):
        if self.detach_head:
            output = hidden_states.detach()
        else:
            output = hidden_states
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)

        return output

class GPT2HeadWithValueModel(GPT2PreTrainedModel):
    """The GPT2HeadWithValueModel class implements a GPT2 language model with a secondary, scalar head."""
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.v_head = ValueHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def detach_value_head(self):
        self.v_head.detach_head = True

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
    ):
       
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)
        value = self.v_head(hidden_states).squeeze(-1)

        outputs = (lm_logits,) + transformer_outputs[1:] + (value,)
        
        return outputs      
      
def respond_to_batch(model, queries, txt_len=20, top_k=0, top_p=1.0):
    """Sample text from language model."""
    input_ids = queries
    for i in range(txt_len):
        # Get Logits
        outputs = model(input_ids)
        next_token_logits = outputs[0][:, -1, :]
        next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        # Sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    return input_ids[:, -txt_len:]
  
class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult

class FixedKLController:
    """Fixed KL controller."""
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass
class PPOTrainer:
    """
    The PPO_trainer uses Proximal Policy Optimization to optimise language models.
    """
    
    default_params = {
        "lr": 1.41e-5,
        "adap_kl_ctrl": True, 
        "init_kl_coef":0.2,
        "target": 6,
        "horizon":10000,
        "gamma":1,
        "lam":0.95,
        "cliprange": .2,
        "cliprange_value":.2,
        "vf_coef":.1,
        "batch_size": 256,
        "forward_batch_size": 4,
        "ppo_epochs": 4,    
    } 
    
    def __init__(self, model, ref_model, **ppo_params):
        """
        Initialize PPOTrainer.
        
        Args:
            model (torch.model): Hugging Face transformer GPT2 model with value head
            ref_model (torch.model): Hugging Face transformer GPT2 refrence model used for KL penalty
            ppo_params (dict or None): PPO parameters for training. Can include following keys:
                'lr' (float): Adam learning rate, default: 1.41e-5
                'batch_size' (int): Number of samples per optimisation step, default: 256
                'forward_batch_size' (int): Number of samples forward passed through model at a time, default: 16
                'ppo_epochs' (int): Number of optimisation epochs per batch of samples, default: 4
                'gamma' (float)): Gamma parameter for advantage calculation, default: 1.
                'lam' (float): Lambda parameter for advantage calcualation, default: 0.95
                'cliprange_value' (float): Range for clipping values in loss calculation, default: 0.2
                'cliprange' (float): Range for clipping in PPO policy gradient loss, default: 0.2
                'vf_coef' (float): Scaling factor for value loss, default: 0.1
                'adap_kl_ctrl' (bool): Use adaptive KL control, otherwise linear, default: True
                'init_kl_coef' (float): Initial KL penalty coefficient (used for adaptive and linear control), default: 0.2
                'target' (float): Target KL value for adaptive KL control, default: 6.0
                'horizon' (float): Horizon for adaptive KL control, default: 10000
                
        """
        self.ppo_params = self.default_params
        self.ppo_params.update(ppo_params)
        
        self.ref_model = ref_model
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=self.ppo_params['lr'])
     
        self.kl_ctl = AdaptiveKLController(self.ppo_params['init_kl_coef'],
                                           self.ppo_params['target'],
                                           self.ppo_params['horizon'])


    def step(self, query, response, scores):
        """
        Run a PPO optimisation step.
        
        args:
            query (torch.tensor): tensor containing the encoded queries, shape [batch_size, query_length]
            response (torch.tensor): tensor containing the encoded responses, shape [batch_size, response_length]
            scores (torch.tensor): tensor containing the scores, shape [batch_size]
            
        returns:
            train_stats (dict): a summary of the training statistics
        """

        bs = self.ppo_params['batch_size']
        timing = dict()
        t0 = time.time()
        
        gen_len = response.shape[1]
        model_input = torch.cat((query, response), axis=1)
        
        t = time.time()
        logprobs, ref_logprobs, values = self.batched_forward_pass(model_input, gen_len)
        timing['time/ppo/forward_pass'] = time.time()-t

        t = time.time()
        rewards, non_score_reward, kl_coef = self.compute_rewards(scores, logprobs, ref_logprobs)
        timing['time/ppo/compute_rewards'] = time.time()-t 
        
        t = time.time() 
        all_stats = []
        idxs = list(range(bs))
        for _ in range(self.ppo_params['ppo_epochs']):
            random.shuffle(idxs)
            for i in range(bs):
                idx = idxs[i]
                train_stats = self.train_minibatch(logprobs[idx:idx+1], values[idx:idx+1],
                                                   rewards[idx:idx+1], query[idx:idx+1],
                                                   response[idx:idx+1], model_input[idx:idx+1])
                all_stats.append(train_stats)
        timing['time/ppo/optimize_step'] = time.time()-t
        
        t = time.time()
       # train_stats = stack_dicts(all_stats)
        
        # reshape advantages/ratios such that they are not averaged.
       # train_stats['policy/advantages'] = torch.flatten(train_stats['policy/advantages']).unsqueeze(0)
       # train_stats['policy/ratio'] = torch.flatten(train_stats['policy/ratio']).unsqueeze(0)
        
        stats = self.record_step_stats(scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs,
                                       non_score_reward=non_score_reward,
                                       kl_coef=kl_coef)
        stats = stats_to_np(stats)
        timing['time/ppo/calc_stats'] = time.time()-t

        self.kl_ctl.update(stats['objective/kl'], self.ppo_params['batch_size'])

        timing['time/ppo/total'] = time.time()-t0
        stats.update(timing)
        return stats

    def batched_forward_pass(self, model_input, gen_len):
        """Calculate model outputs in multiple batches."""
        bs = self.ppo_params['batch_size']
        fbs = self.ppo_params['forward_batch_size']
        logprobs = []
        ref_logprobs = []
        values = []
        
        for i in range(int(self.ppo_params['batch_size']/fbs)):
            m_input = model_input[i*fbs:(i+1)*fbs]
            logits, _, v = self.model(m_input)
            ref_logits, _, _ = self.ref_model(m_input)
            
            values.append(v[:, -gen_len-1:-1].detach())
            logprobs.append(logprobs_from_logits(logits[:,:-1,:], m_input[:,1:])[:, -gen_len:].detach())
            ref_logprobs.append(logprobs_from_logits(ref_logits[:,:-1,:], m_input[:,1:])[:, -gen_len:].detach())
   
        return torch.cat(logprobs), torch.cat(ref_logprobs), torch.cat(values)
    
    def train_minibatch(self, logprobs, values, rewards, query, response, model_input):
        """Train one PPO minibatch"""
        loss_p, loss_v  = self.loss(logprobs, values, rewards, query, response, model_input)
        loss = loss_p + loss_v
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()
    
    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """Compute per token rewards from scores and KL-penalty."""
        kl = logprobs - ref_logprobs
        non_score_reward = -self.kl_ctl.value * kl
        rewards = non_score_reward.clone().detach()
        rewards[:, -1] += scores
        return rewards, non_score_reward, self.kl_ctl.value

    def loss(self, old_logprobs, values, rewards, query, response, model_input):
        """Calculate policy and value losses."""
        lastgaelam = 0
        advantages_reversed = []
        gen_len = response.shape[1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = rewards[:, t] + self.ppo_params['gamma'] * nextvalues - values[:, t]
            lastgaelam = delta + self.ppo_params['gamma'] * self.ppo_params['lam'] * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1]).transpose(0, 1)

        returns = advantages + values
        advantages = whiten(advantages)
        advantages = advantages.detach()

        logits, _, vpred = self.model(model_input)
        logprob = logprobs_from_logits(logits[:,:-1,:], model_input[:, 1:])
        
        #only the generation part of the values/logprobs is needed
        logprob, vpred = logprob[:, -gen_len:], vpred[:,-gen_len-1:-1]

        vpredclipped = clip_by_value(vpred,
                                     values - self.ppo_params["cliprange_value"],
                                     values + self.ppo_params["cliprange_value"])

        vf_losses1 = (vpred - returns)**2
        vf_losses2 = (vpredclipped - returns)**2
        vf_loss = .5 * torch.mean(torch.max(vf_losses1, vf_losses2))
        vf_clipfrac =  torch.mean(torch.gt(vf_losses2, vf_losses1).double())

        ratio = torch.exp(logprob - old_logprobs)
        
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio,
                                               1.0 - self.ppo_params['cliprange'],
                                               1.0 + self.ppo_params['cliprange'])

        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))
        pg_clipfrac = torch.mean(torch.gt(pg_losses2, pg_losses).double())
        
        loss = pg_loss + self.ppo_params['vf_coef'] * vf_loss

        entropy = torch.mean(entropy_from_logits(logits))
        approxkl = .5 * torch.mean((logprob - old_logprobs)**2)
        policykl = torch.mean(logprob - old_logprobs)
        return_mean, return_var = torch.mean(returns), torch.var(returns)
        value_mean, value_var = torch.mean(values), torch.var(values)

        #stats = dict(
        #    loss=dict(policy=pg_loss, value=vf_loss, total=loss),
        #    policy=dict(entropy=entropy, approxkl=approxkl,policykl=policykl, clipfrac=pg_clipfrac,
        #                advantages=advantages, advantages_mean=torch.mean(advantages), ratio=ratio),
        #    returns=dict(mean=return_mean, var=return_var),
        #    val=dict(vpred=torch.mean(vpred), error=torch.mean((vpred - returns) ** 2),
        #             clipfrac=vf_clipfrac, mean=value_mean, var=value_var),
        #)
        return pg_loss, self.ppo_params['vf_coef'] * vf_loss#, flatten_dict(stats)


    def record_step_stats(self, kl_coef, **data):
        """Record training step statistics."""
        kl = data['logprobs'] - data['ref_logprobs']
        mean_kl = torch.mean(torch.sum(kl, axis=-1))
        mean_entropy = torch.mean(torch.sum(-data['logprobs'], axis=1))
        mean_non_score_reward =torch.mean(torch.sum(data['non_score_reward'], axis=1))
        stats = {
            'objective/kl': mean_kl,
            'objective/kl_dist': kl,
            'objective/logprobs': data['logprobs'],
            'objective/ref_logprobs': data['ref_logprobs'],
            'objective/kl_coef': kl_coef,
            'objective/entropy': mean_entropy,
            'ppo/mean_non_score_reward': mean_non_score_reward,
        }

        #for k, v in data['train_stats'].items():
        #    stats[f'ppo/{k}'] = torch.mean(v, axis=0)
        #stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        return stats
      
      
if __name__ == '__main__':
  gpt2_model = GPT2HeadWithValueModel.from_pretrained('gpt2')
  gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('gpt2')
  tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  special_tokens_dict = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
  tokenizer.add_special_tokens(special_tokens_dict)
  gpt2_model_ref.resize_token_embeddings(len(tokenizer))
  gpt2_model.resize_token_embeddings(len(tokenizer))
  
  zip = zipfile.ZipFile("/content/drive/My Drive/SemProjectFiles/data.tar.zip")
  file1 = open("/content/drive/My Drive/SemProjectFiles/programs_training.txt",encoding="utf8") 
  files = []

  count = 0
  for g in file1:
    temp = nltk.word_tokenize(g)
    files.append(temp)
    
    n_epochs = 1# or whatever
  batch_size = 8
  loss_array = []
  file_size = 256
  import time
  #model = model.to('cuda')
  inputs = []
  count = 0
  rewards = []
  kl_div = []

  for k in range(0,len(files), file_size):
    start = time.time()
    print(k)
    training_file = return_train(files[k:k+file_size], zip)
    inputs = tokenizer(training_file, tokenizer)
    padded_inputs, attention_masks = pad_sents(inputs, tokenizer.pad_token_id)
    tensor_inputs = torch.tensor(padded_inputs).to('cuda')
    tensor_attention_masks = torch.tensor(attention_masks).to('cuda')
    permutation = torch.randperm(tensor_inputs.size()[0])
    query_tensors = padded_inputs[:,:30]
    response = []
    scores = []
    
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

    
   
