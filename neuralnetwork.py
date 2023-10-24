import torch
from torch import nn
import matplotlib.pyplot as plt
import time
from constants import *

class NeuralNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.stack = nn.Sequential(
			nn.Conv2d(3, 32, 7, 2, 3),
			nn.ReLU(),
			nn.Conv2d(32, 64, 7, 2, 3),
			nn.ReLU(),
			nn.Conv2d(64, 128, 7, 2, 3),
			nn.ReLU(),

			nn.ConvTranspose2d(128, 64, 6, 2, 2),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 32, 6, 2, 2),
			nn.ReLU(),
			nn.ConvTranspose2d(32, 3, 6, 2, 2),
			nn.Sigmoid()
		)
	
	def forward(self, x):
		return self.stack(x)
	
if __name__ == '__main__':
	example_input = torch.randn(size=(30, 3, 224, 224), device=device)
	model = NeuralNetwork()
	model.to(device)
	start = time.time()
	with torch.no_grad():
		predicted_output = model(example_input)
	print(example_input.shape)
	print(predicted_output.shape)
	print(f'Time: {time.time() - start}')