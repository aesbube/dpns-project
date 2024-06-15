# Backgound Subtraction 
This project uses the [DAVIS-dataset](https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip). After downloading the dataset, unzip it, and put the 'Annotations' and 'JPEGImages' folders in project and only use the 480p folders. Because the models are trained with a small database, the results may vary, so if you want better results you can use a larger database, but make sure to make a new trainval.txt file with keys and values according to your database.

## Install dependencies
To install all dependencies, simply run the commands:
1. 'pip install poetry'
2. 'poetry install --no-root'
3. run the command 'pip install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia' so that if you want to train the model, you can use your GPU instead of your CPU

## Running 
- To start training the model, run the script 'train_unet.py'.
- To test the model, run the script 'test_unet.py'
- To evaluate the model on a random image from the validation set, first run the script 'get_testing_images.py' and after that run the script 'evaluate.py'. 

