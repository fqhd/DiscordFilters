import cv2
from tqdm import tqdm
import os
import torch

model = torch.load('face_detection/model.pkl')

def parse_videos(orig_path, enh_path):
	original = cv2.VideoCapture(orig_path)
	enhanced = cv2.VideoCapture(enh_path)
	start_idx = len(os.listdir('dataset/original'))

	orig_num_frames = int(original.get(cv2.CAP_PROP_FRAME_COUNT))
	enh_num_frames = int(enhanced.get(cv2.CAP_PROP_FRAME_COUNT))

	if orig_num_frames != enh_num_frames:
		print('Warning: Videos dont have the same number of frames...')

	num_frames = min(orig_num_frames, enh_num_frames)

	idx = 0

	for i in tqdm(range(120, num_frames-1)):
		original.set(cv2.CAP_PROP_POS_FRAMES, i)
		enhanced.set(cv2.CAP_PROP_POS_FRAMES, i)
		_, original_frame = original.read()
		_, enhanced_frame = enhanced.read()

		original_frame = cv2.resize(original_frame, (640, 360))
		enhanced_frame = cv2.resize(enhanced_frame, (640, 360))

		original_frame_lr = cv2.resize(original_frame, (160, 90))
		original_frame_lr = torch.tensor(original_frame_lr)
		original_frame_lr = original_frame_lr / 255.0
		original_frame_lr = original_frame_lr.permute(2, 0, 1)
		original_frame_lr = torch.unsqueeze(original_frame_lr, 0)
		with torch.no_grad():
			selection = model(original_frame_lr).numpy()[0]
			
		x = int(selection[0] * 640)
		y = int(selection[1] * 360)
		w = int(selection[2] * (392 - 80) + 80)
		if x < 0:
			x = 0
		if y < 0:
			y = 0
		if x+w >= 640:
			w = 640 - x
		if y+w >= 360:
			w = 360 - y

		original_frame = original_frame[y:y+w, x:x+w]
		enhanced_frame = enhanced_frame[y:y+w, x:x+w]
		original_frame = cv2.resize(original_frame, (224, 224))
		enhanced_frame = cv2.resize(enhanced_frame, (224, 224))
		cv2.imwrite(f'dataset/original/{idx + start_idx}.jpg', original_frame)
		cv2.imwrite(f'dataset/enhanced/{idx + start_idx}.jpg', enhanced_frame)
		idx += 1
		

original_videos = os.listdir('original_videos')
enhanced_videos = os.listdir('enhanced_videos')

if len(original_videos) != len(enhanced_videos):
	print('Warning: Number of original videos does not match number of enhanced videos')

for i in range(len(original_videos)):
	orig_video_path = f'original_videos/{original_videos[i]}'
	enh_video_path = f'enhanced_videos/{enhanced_videos[i]}'
	parse_videos(orig_video_path, enh_video_path)

