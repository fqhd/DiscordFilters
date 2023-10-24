import torch
import torch.utils.data as dutils
from torchvision import datasets
import torchvision.transforms as T
from torchvision.io import read_image
import os
from constants import *
import matplotlib.pyplot as plt

transform = T.Compose([
	T.RandomHorizontalFlip()
])

class Dataset(dutils.Dataset):
	def __init__(self, split, root='root', transform=None):
		self.root = root
		self.orig_img_names = os.listdir(f'{root}/original')
		self.enh_img_names = os.listdir(f'{root}/enhanced')
		num_orig_imgs = len(self.orig_img_names)
		num_enh_imgs = len(self.enh_img_names)
		if num_orig_imgs != num_enh_imgs:
			print('Warning: theres a different number of original and enhanced images, this may mean you are using the library incorrectly')
		self.num_imgs = min(num_orig_imgs, num_enh_imgs)
		self.start_idx = 0
		if split == 'validation':
			self.start_idx = int(.8 * self.num_imgs)
			self.num_imgs = int(.9 * self.num_imgs) - self.start_idx
		elif split == 'training':
			self.num_imgs = int(.8 * self.num_imgs)
		elif split == 'testing':
			self.start_idx = int(.9 * self.num_imgs)
			self.num_imgs = self.num_imgs - self.start_idx
		else:
			print('Warning: no split specified, using entire dataset')
		print(f'Using {self.num_imgs} images for {split}')
		self.transform = transform

	def __len__(self):
		return self.num_imgs

	def __getitem__(self, idx):
		idx += self.start_idx
		orig_img_name = self.orig_img_names[idx]
		enh_img_name = self.enh_img_names[idx]
		orig_img = read_image(f'{self.root}/original/{orig_img_name}')
		enh_img = read_image(f'{self.root}/enhanced/{enh_img_name}')
		orig_img = torch.unsqueeze(orig_img, 0)
		enh_img = torch.unsqueeze(enh_img, 0)
		img_pair = torch.concat([orig_img, enh_img], dim=0)
		transformed_images = self.transform(img_pair)
		return transformed_images[0] / 255.0, transformed_images[1] / 255.0

train_dataset = Dataset('training', root='dataset', transform=transform)
val_dataset = Dataset('validation', root='dataset', transform=transform)
test_dataset = Dataset('testing', root='dataset', transform=transform)

train_dataloader = dutils.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = dutils.DataLoader(val_dataset, batch_size=128, shuffle=True)
test_dataloader = dutils.DataLoader(test_dataset, batch_size=128, shuffle=True)

if __name__ == '__main__':
	test_itr = iter(test_dataset)
	orig, enh = next(test_itr)
	print('Original:', orig.shape, orig.dtype)
	print('Enhanced:', enh.shape, orig.dtype)
	plt.figure(figsize=(10, 8))
	plt.title('Debug Images')
	plt.axis('off')

	for i in range(0, 8, 2):
		orig, enh = next(test_itr)
		plt.subplot(4, 2, i+1)
		plt.xticks([])
		plt.yticks([])
		plt.xlabel('Original')
		plt.imshow(T.ToPILImage()(orig))

		plt.subplot(4, 2, i+2)
		plt.xticks([])
		plt.yticks([])
		plt.xlabel('Enhanced')
		plt.imshow(T.ToPILImage()(enh))

	plt.show()


