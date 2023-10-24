import torch
from dataset import test_dataloader
import matplotlib.pyplot as plt
import torchvision.transforms as T
from neuralnetwork import NeuralNetwork

model = torch.load('models/cartoon.pkl')
model.eval()

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