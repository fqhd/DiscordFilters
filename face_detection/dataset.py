import torch
import torch.utils.data as dutils
import pandas as pd
from torchvision.io.image import read_image
import torchvision.transforms.functional as F
import torchvision.transforms as T
import random
from constants import *
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

class Dataset(dutils.Dataset):
	def __init__(self, data_augmentation=False, subset=None):
		super().__init__()
		self.df = pd.read_csv('dataset/selections.csv', names=['FilePath', 'XCoord', 'YCoord', 'Size'])
		self.subset = subset
		self.data_augmentation = data_augmentation
		self.transform = T.Compose([
			T.ColorJitter((0.5, 1.1), (0.2), (0.5, 1.5), hue=(-0.1, 0.1))
		])
		self.start_idx = 0
		if subset == 'validation':
			self.start_idx = 400
		elif subset == 'testing':
			self.start_idx = 450

	def __len__(self):
		if self.subset == 'training':
			return 400
		elif self.subset == 'validation':
			return 50
		elif self.subset == 'testing':
			return 50
		else:
			return 500
	
	def __getitem__(self, idx):
		idx += self.start_idx
		row = self.df.iloc[idx]
		img_path = row['FilePath']
		x_coord = row['XCoord']
		y_coord = row['YCoord']
		size = row['Size']
		image = read_image(f'dataset/{img_path}')
		image = image / 255.0
		if self.data_augmentation:
			image += torch.randn(image.shape) * 0.05

			image = self.transform(image)

			x_coord += random.random() * 0.005
			y_coord += random.random() * 0.005
			size += random.random() * 0.01

			if random.randint(0, 1):
				image = F.hflip(image)
				x_coord = 160 * x_coord
				size_pxl = size * (98 - 20) + 20
				x_coord = 80 - (x_coord - 80) - size_pxl
				x_coord /= 160
			return image, torch.tensor([x_coord, y_coord, size])
		else:
			return image, torch.tensor([x_coord, y_coord, size])

train_dataset = Dataset(data_augmentation=True, subset='training')
val_dataset = Dataset(data_augmentation=False, subset='validation')
test_dataset = Dataset(data_augmentation=False, subset='testing')

train_dataloader = dutils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = dutils.DataLoader(val_dataset, batch_size=50, shuffle=True)
test_dataloader = dutils.DataLoader(test_dataset, batch_size=50, shuffle=True)

if __name__ == '__main__':
	image_batch, selection_batch = next(iter(test_dataloader))

	plt.figure(figsize=(8, 8))
	plt.axis('off')
	for i in range(9):
		pil_img = T.ToPILImage()(image_batch[i])
		draw = ImageDraw.Draw(pil_img)
		w, h = pil_img.size
		x = selection_batch[i][0] * w
		y = selection_batch[i][1] * h
		s_w = selection_batch[i][2] * (98 - 20) + 20
		draw.rectangle((x, y, x+s_w, y+s_w))
		
		plt.subplot(3, 3, i+1)
		plt.axis('off')
		plt.imshow(pil_img)
	plt.show()