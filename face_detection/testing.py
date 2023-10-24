import torch
from dataset import test_dataloader
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import ImageDraw

model = torch.load('model.pkl')

image_batch, selection = next(iter(test_dataloader))
print(image_batch.shape)
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