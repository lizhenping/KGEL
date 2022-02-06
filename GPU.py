import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device:', device)