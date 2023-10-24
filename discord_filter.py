import pyvirtualcam
import cv2
import torch
import torchvision.transforms as T
import numpy as np

model = torch.load('models/old.pkl')
model.eval()
face_detection = torch.load('face_detection/model.pkl')
face_detection.eval()

def draw_image_np(main_image, small_image, x, y):
    # Ensure the images have the correct dimensions
    assert main_image.shape == (360, 640, 3), "Main image should be 640x360 RGB"
    
    # Get the height and width of the small image
    small_height, small_width = small_image.shape[:2]
    
    # Make sure the specified location is valid
    assert x >= 0 and x + small_width <= 640, "Invalid X coordinate"
    assert y >= 0 and y + small_height <= 360, "Invalid Y coordinate"
    
    # Copy the main image to avoid modifying the original array
    result_image = np.copy(main_image)
    
    # Get the region where the small image will be placed
    region = result_image[y:y+small_height, x:x+small_width]
    
    # Replace the region with the small image
    result_image[y:y+small_height, x:x+small_width] = small_image
    
    return result_image

def main():
	cv_cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
	cv_cam.set(cv2.CAP_PROP_GAIN, 45)
	cv_cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640);
	cv_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360);

	cam = pyvirtualcam.Camera(width=640, height=360, fps=20)

	while True:
		_, frame = cv_cam.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# Find face in frame
		original_frame_lr = cv2.resize(frame, (160, 90))
		original_frame_lr = torch.tensor(original_frame_lr)
		original_frame_lr = original_frame_lr / 255.0
		original_frame_lr = original_frame_lr.permute(2, 0, 1)
		original_frame_lr = torch.unsqueeze(original_frame_lr, 0)
		with torch.no_grad():
			selection = face_detection(original_frame_lr).numpy()[0]
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
		face_frame = frame[y:y+w, x:x+w]
		face_frame = cv2.resize(face_frame, (640, 360))

		#finance.income@ntu.ac.uk
		
		# Morph face with AI
		face_frame = cv2.resize(face_frame, (224, 224))
		face_frame = torch.tensor(face_frame)
		face_frame = torch.permute(face_frame, (2, 0, 1))
		face_frame = (face_frame / 255.0)
		with torch.no_grad():
			face_frame = model(face_frame)
		face_frame = torch.permute(face_frame, (1, 2, 0))
		face_frame = face_frame.numpy()
		face_frame = (face_frame * 255).astype('uint8')
		face_frame = cv2.resize(face_frame, (w, w))

		# Replace face in frame with morphed face
		frame = draw_image_np(frame, face_frame, x, y)

		# Send morphed frame back to virtual camera
		cam.send(frame)
		cam.sleep_until_next_frame()

main()