# FONT-DETECTION

## Overview
This project trains a cnn model to identify and predict a phrases " Hello, World!".

1. Source Code & Instructions:
   - The source code is structured into several files, each handling different aspects of the project:
     - `data.py`: Script responsible for creating the dataset using data augmentation techniques.
     - `assign.ipynb`: Jupyter notebook containing the data preprocessing, model training, and prediction steps.
     - Ensure all dependencies are installed before running the code. You can install the necessary packages using the provided `requirements.txt` file.
   - To run the project:
     1. Start with the data augmentation process by running `data.py`.
     2. Then proceed with the Jupyter notebook for training and testing the model.

2. Data Creation Script & Data Used for Training:
   - The dataset was created using a data augmentation process to generate additional samples. For each input image, 10 augmented images were generated to improve model generalization.
   - Techniques such as sharpening, contrast enhancement, and denoising were applied to enhance image quality.
   - Data was labeled and split into training and testing sets for model training.

3. Training Script (Deep Learning):
   - The `assign.ipynb` file contains the code for training the deep learning model.
   - The training process includes image preprocessing steps, model architecture setup, and evaluation metrics like accuracy and loss.

4. Model Architecture & Trained Weights:
   - The model is built using the following layers:
     - 3 Convolutional layers with 32, 64, and 128 filters respectively, used to capture spatial hierarchies in the images.
     - Each convolution layer is followed by a Max Pooling layer to downsample the feature maps and reduce computational cost.
     - 2 Fully connected (Dense) layers with 128 units and an output layer with 10 units representing the font classes.
     - Softmax is used at the output to predict the probability for each of the 10 font types.
   - The final trained model is saved as an `.h5` file, which can be used for further predictions.
   - The trained weights ensure the model is capable of identifying fonts accurately after preprocessing.
