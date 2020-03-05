# %%

from mmdet.apis import init_detector, inference_detector, save_result_pyplot
import mmcv
from os import listdir
from os.path import isfile, join

config_file = '/home/benbarka/mmdetection/configs/fouriernet/fourier_768_1x_r50_36_60.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '/home/benbarka/mmdetection/configs/fouriernet/fourier_768_1x_r50_36_60.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

images_path = '/home/benbarka/PolarMask/samples'
results_path = '/home/benbarka/PolarMask/results'
images_folder = [f for f in listdir(images_path) if isfile(join(images_path, f))]

for image in images_folder:
    result = inference_detector(model, join(images_path, image))
    save_result_pyplot(image, result, model.CLASSES, images_path, results_path)
    print(image)
