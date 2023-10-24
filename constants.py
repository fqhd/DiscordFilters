import torch

BATCH_SIZE = 4

device = (
	'cuda' if torch.cuda.is_available()
	else
	'cpu'
)

LEARNING_RATE = 2e-4
N_EPOCHS = 5