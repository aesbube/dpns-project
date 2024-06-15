import cv2
import numpy as np
import torch
from unet_model import unet_model
import random

input_shape = (256, 256)
model = unet_model(n_channels=3, n_classes=1)
model.load_state_dict(torch.load('models/unet/unet_model_50.pth'))
model.cuda()
model.eval()


def preprocess_image(image, input_shape):
    image = cv2.resize(image, input_shape)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = torch.tensor(image, dtype=torch.float32).cuda()
    return image


def predict_masks(image):
    image = preprocess_image(image, input_shape)
    with torch.no_grad():
        mask = model(image)[0].cpu().numpy()
    mask = np.transpose(mask, (1, 2, 0))
    mask = mask.squeeze()
    return mask


def remove_background(image, mask):
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = np.expand_dims(mask, axis=-1)
    mask = (mask > 0.5).astype(np.uint8)
    result = image * mask
    return result


number = random.randint(0, 690)

test_image_path = f'testing_images/image_{number}.jpg'
test_image = cv2.imread(test_image_path)

predicted_mask = predict_masks(test_image)

result_image = remove_background(test_image, predicted_mask)

cv2.imshow('Original Image', test_image)
cv2.imshow('Predicted Mask', predicted_mask)
cv2.imshow('Result Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
