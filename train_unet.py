from unet_model import unet_model
from training import train

epochs = 150
model_save_path = 'models/unet/unet_model'

model = unet_model(n_channels=3, n_classes=1)

train(epochs, model_save_path, model, False)