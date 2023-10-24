import sys

from dataset import train_dataloader, val_dataloader, test_dataloader
from neuralnetwork import NeuralNetwork
from constants import *
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image, ImageDraw

model = NeuralNetwork()
model = model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.L1Loss()

def train():
	avg_loss = 0
	for image_batch, selection in tqdm(train_dataloader):
		image_batch = image_batch.to(device)
		selection = selection.to(device)
		
		predicted_output= model(image_batch)
		loss = loss_fn(predicted_output, selection)

		optim.zero_grad()
		loss.backward()
		optim.step()

		avg_loss += loss.item()
	avg_loss /= len(train_dataloader)
	return avg_loss

def test(ds):
	avg_loss = 0
	for image_batch, selection in ds:
		image_batch = image_batch.to(device)
		selection = selection.to(device)
		with torch.no_grad():
			predicted_output = model(image_batch)
			loss = loss_fn(predicted_output, selection)
		avg_loss += loss.item()
	avg_loss /= len(ds)
	return avg_loss

for epoch in range(N_EPOCHS):
	start = time.time()
	print(f'Beginning Training For Epoch {epoch+1}')
	train_loss = train()
	val_loss = test(val_dataloader)
	print(f'Time: {time.time() - start}')
	print(f'Validation Loss: {val_loss}')
	print(f'Training Loss: {train_loss}')

model = model.to('cpu')
image_batch, selection = next(iter(test_dataloader))
with torch.no_grad():
	predicted_output = model(image_batch)

plt.figure(figsize=(8, 8))
plt.title('Final Model Test')
plt.axis('off')
for i in range(12):
	pil_img = T.ToPILImage()(image_batch[i])
	draw = ImageDraw.Draw(pil_img)

	w, h = pil_img.size
	x = predicted_output[i][0] * w
	y = predicted_output[i][1] * h
	s_w = predicted_output[i][2] * (98 - 20) + 20
	draw.rectangle((x, y, x+s_w, y+s_w))

	plt.subplot(4, 3, i+1)
	plt.axis('off')
	plt.imshow(pil_img.resize((224, 224)))

plt.show()

model = model.to('cpu')
torch.save(model, f'model.pkl')
