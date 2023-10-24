from tqdm import tqdm
import cv2
import os
import numpy as np

NUM_IMAGES_PER_VIDEO = 500
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480

def parse_video(path):
	original = cv2.VideoCapture(path)
	num_frames = int(original.get(cv2.CAP_PROP_FRAME_COUNT))
	total_frames_in_video = int(original.get(cv2.CAP_PROP_FRAME_COUNT))

	if num_frames < NUM_IMAGES_PER_VIDEO:
		print('Warning: There arent enough frames in the video to make enough training data')
		print('Found frames:', total_frames_in_video)
		print('Needed frames:', NUM_IMAGES_PER_VIDEO)
	
	num_frames = min(num_frames, NUM_IMAGES_PER_VIDEO)

	frame_ids = np.array(range(total_frames_in_video))
	np.random.shuffle(frame_ids)
	frame_ids = frame_ids[:NUM_IMAGES_PER_VIDEO]

	idx = len(os.listdir('dataset/frames'))

	for frame_id in tqdm(frame_ids):
		original.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
		_, frame = original.read()
		frame = cv2.resize(frame, (320, 180))
		cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		cv2.imwrite(f'dataset/frames/{idx}.jpg', frame)
		idx += 1

videos = os.listdir('../original_videos')
for video_name in videos:
	parse_video(f'../original_videos/{video_name}')