# Source code mapping to Natural Language

The relevant files are organized accordingly:

```
├── Source code mapping:
	├── models: Contains all the relevant models used
	   ├── base_classes.py:  BERT + attention with GNN, Sinkhorn iterations (Optimal Transport)
	   ├── graphs.py: GNN architecture and related functions
	   ├── gpt2.py: Base GPT2 File
	   ├── GPT2_valuehead.py: Value head for GPT2 reinforcement Learning
	   ├── ppo.py: Training for GPT2 reinforcement Learning
	├── Cross_Embedding:
	   ├── graph_src.py
	   ├── natural_lang.py: Source code, Natural language modeling and Mapping between the two
 
```
