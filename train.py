import sys

if len(sys.argv) == 1:
	print('Warning: please specify the name under which you want to save the model')
	print('Example: python train.py my_model')
	exit()

model_name = sys.argv[1]

from dataset import train_dataloader, val_dataloader, test_dataloader
from neuralnetwork import NeuralNetwork
from constants import *
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import torchvision.transforms as T


model = NeuralNetwork()
model = model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.L1Loss()

def train():
	avg_loss = 0
	for image_batch, target_image in tqdm(train_dataloader):
		image_batch = image_batch.to(device)
		target_image = target_image.to(device)
		
		predicted_output= model(image_batch)
		loss = loss_fn(predicted_output, target_image)

		optim.zero_grad()
		loss.backward()
		optim.step()

		avg_loss += loss.item()
	avg_loss /= len(train_dataloader)
	return avg_loss

def test(ds):
	avg_loss = 0
	for image_batch, target_image in ds:
		image_batch = image_batch.to(device)
		target_image = target_image.to(device)
		with torch.no_grad():
			predicted_output = model(image_batch)
			loss = loss_fn(predicted_output, target_image)
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
image_batch, target_image = next(iter(test_dataloader))
with torch.no_grad():
	predicted_output = model(image_batch)

plt.figure(figsize=(8, 6))
plt.title('Final Model Test')
plt.axis('off')
for i in range(0, 12, 3):
	plt.subplot(4, 3, i+1)
	orig_img_pil = T.ToPILImage()(image_batch[i])
	plt.xticks([])
	plt.yticks([])
	plt.imshow(orig_img_pil.resize((224, 224)))
	plt.xlabel('Original')

	plt.subplot(4, 3, i+2)
	enh_img_pil = T.ToPILImage()(predicted_output[i])
	plt.xticks([])
	plt.yticks([])
	plt.imshow(enh_img_pil.resize((224, 224)))
	plt.xlabel('Predicted')

	plt.subplot(4, 3, i+3)
	tgt_img_pil = T.ToPILImage()(target_image[i])
	plt.xticks([])
	plt.yticks([])
	plt.imshow(tgt_img_pil.resize((224, 224)))
	plt.xlabel('Target')
plt.show()

model = model.to('cpu')
torch.save(model, f'models/{model_name}.pkl')
