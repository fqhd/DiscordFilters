import torch
from torch import nn
import time
from constants import *

class NeuralNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.stack = nn.Sequential(
			nn.Conv2d(3, 32, 7, 1, 3),
			nn.MaxPool2d(2, 2),
			nn.ReLU(),

			nn.Conv2d(32, 64, 5, 1, 2),
			nn.MaxPool2d(2, 2),
			nn.ReLU(),

			nn.Conv2d(64, 128, 3, 1, 1),
			nn.MaxPool2d(2, 2),
			nn.ReLU(),

			nn.Conv2d(128, 256, 3, 1, 1),
			nn.MaxPool2d(2, 2),
			nn.ReLU(),

			nn.Conv2d(256, 16, 3, 1, 1),
			nn.ReLU(),
			nn.Flatten(),
			nn.Linear(800, 3)
		)
	
	def forward(self, x):
		return self.stack(x)
	
if __name__ == '__main__':
	example_input = torch.randn(size=(30, 3, 160, 90), device=device)
	model = NeuralNetwork()
	model.to(device)
	start = time.time()
	with torch.no_grad():
		predicted_output = model(example_input)
	print(example_input.shape)
	print(predicted_output.shape)
	print(f'Time: {time.time() - start}')