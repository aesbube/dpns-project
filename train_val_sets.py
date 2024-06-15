from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import torch


def prepare_dataset_from_txt(txt_file, input_shape, flag):
    images = []
    masks = []
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            image_path, mask_path = line.strip().split()
            image_path = '.' + image_path
            mask_path = '.' + mask_path
            image = cv2.imread(image_path)
            if flag is True:
                image = cv2.resize(image, (input_shape[1], input_shape[0]))
                image = image / 255.0
            images.append(image)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (input_shape[1], input_shape[0]))
            if flag is True:
                mask = np.expand_dims(mask, axis=-1)
                mask = mask / 255.0
            masks.append(mask)
    if flag is True:
        return np.array(images), np.array(masks)
    else:
        return images, masks


def sets(input_shape, flag):
    txt_file = 'trainval.txt'

    images, masks = prepare_dataset_from_txt(txt_file, input_shape, flag)
    if flag is True:
        images_tensor = torch.tensor(images).permute(0, 3, 1, 2).float().cuda()
        masks_tensor = torch.tensor(masks).permute(0, 3, 1, 2).float().cuda()
        train_x, val_x, train_y, val_y = train_test_split(
            images_tensor, masks_tensor, test_size=0.2, random_state=42)
    else:
        train_x, val_x, train_y, val_y = train_test_split(
            images, masks, test_size=0.2, random_state=42)
    return train_x, val_x, train_y, val_y
