import torch

device = (
	'cuda' if torch.cuda.is_available()
	else
	'cpu'
)

BATCH_SIZE = 8
LEARNING_RATE = 1e-4
N_EPOCHS = 20