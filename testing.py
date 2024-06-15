import os
import torch
import cv2
import numpy as np
from sklearn.metrics import jaccard_score
from train_val_sets import sets


def calculate_jaccard_score(true_masks, predicted_masks):
    jaccard_scores = []
    for true_mask, predicted_mask in zip(true_masks, predicted_masks):
        true_mask_binary = (true_mask > 0.5).astype(np.uint8)
        predicted_mask_binary = (predicted_mask > 0.5).astype(np.uint8)
        jaccard_score_val = jaccard_score(
            true_mask_binary.flatten(), predicted_mask_binary.flatten(), zero_division=0)
        jaccard_scores.append(jaccard_score_val)
    avg_jaccard_score = np.mean(jaccard_scores)
    return avg_jaccard_score


def preprocess_image(image, input_shape, device):
    image = cv2.resize(image, input_shape)
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    image = torch.tensor(image, dtype=torch.float32).to(device)
    return image


def predict_masks(images, model, input_shape, device, flag):
    masks = []
    for image in images:
        image = preprocess_image(image, input_shape, device)
        with torch.no_grad():
            if flag is True:
                output = model(image)['out']  
                mask = torch.sigmoid(output).cpu().numpy()
            else:
                mask = model(image)[0].cpu().numpy()
                mask = np.transpose(mask, (1, 2, 0))
        mask = mask.squeeze()
        masks.append(mask)
    return masks


def test(model_path, results_path, model, flag=False):
    input_shape = (256, 256)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    _, val_x, _, val_y = sets(input_shape, False)

    predicted_masks = predict_masks(val_x, model, input_shape, device, flag)

    os.makedirs(results_path, exist_ok=True)

    for i, mask in enumerate(predicted_masks):
        result_path = os.path.join(results_path, f'result_mask_{i}.png')
        cv2.imwrite(result_path, (mask * 255).astype(np.uint8))

    accuracy = calculate_jaccard_score(val_y, predicted_masks)
    print(f'Accuracy: {accuracy}')
