# import torch

# s=torch.load('PAC-dataset/metamon_models/medium-rl/ckpts/policy_weights/policy_epoch_40.pt', map_location='cpu')

# print(s.keys())

import torch

state_dict= torch.load('model_weights.pt', map_location='cpu')
print(state_dict.keys())