import os
import cv2
from train_val_sets import sets

_, val_x, _, _ = sets((256, 256), False)

folder = 'testing_images'
os.makedirs(folder, exist_ok=True)

for i, image in enumerate(val_x):
    result_path = os.path.join(folder, f'image_{i}.jpg')
    cv2.imwrite(result_path, image)