# DiscordFilters
DiscordFilters is an app that makes deepfakes of your face from images that are streamed from your camera. It uploads the streamed results onto a virtual camera that you can then use in whichever app you want(not limited to discord, works on whatsapp, obs, streamlabs etc... as well)

## Examples

## Dependencies
First, make sure you have the following dependencies installed:
- Python
- PyTorch(GPU version is preferable if you can)
- Numpy
- Matplotlib
- Pillow
- OpenCV
- PyVirtualCam

## Training
You will now need to provide some images of your face in order to train the model on your face for optimal results. If you wish to use the pretrained model, you can skip this step. However for optimal results it is recommended to finetune the model to your face. To do this, place videos of your face inside the `original_videos` folder(I recommend at least 4 minutes of content in total, but no more than 10). Now place the enhanced videos inside the `enhanced_videos` folder. Now run the following commands:
1. `python parse_dataset.py` This prepares the dataset for training.
1. `python train.py` This actually trains the AI, the more videos you have the longer this will take.

## Usage:
Now that you have finetuned the AI to your face, you can run it with this command:
- `python discord_filter.py`
Just make sure you pick `OBS Camera` in the app in which you want the changes to take effect.

Enjoy!
