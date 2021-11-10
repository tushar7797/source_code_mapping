# Source code mapping to Natural Language

The relevant files for mapping tasks are organized accordingly:

```
├── Source code mapping:
	├── models: Contains all the relevant models used
	   ├── base_classes.py:  BERT + attention with GNN, Sinkhorn iterations (Optimal Transport)
	   ├── graphs.py: GNN architecture and related functions
	   └── ...
	   
	├── Cross_Embedding:
	   ├── graph_src.py
	   ├── natural_lang.py: Source code, Natural language modeling and Mapping between the two
 
```

The relevant files for code completion tasks are organized accordingly:

```
├── Source code mapping:
	├── src: Contains all the relevant models used
	   ├── gpt2.py: Base GPT2 File
	   ├── GPT2_valuehead.py: Training for GPT2 reinforcement Learning
	   ├── ppo.py: Base file for GPT2 reinforcement Learning
	   ├── graph+gpt2.py: Training for Graph and GPT2 
	   └── ...
	
 
```

The original folder is old code and only used for reference
