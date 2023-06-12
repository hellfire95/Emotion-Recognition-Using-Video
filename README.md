# Emotion Recognition

## Introduction
Emotion recognition is the process of identifying human emotions using artificial intelligence. This project focuses on recognizing emotions from speech clips, which include both audio and video data. The goal is to implement a two-stream model that combines Speech Emotion Recognition (SER) and Facial Emotion Recognition (FER) to achieve more accurate results.

## Dataset
The models in this project are trained on the RAVDESS dataset, which contains speech clips with eight emotion classes: neutral, calm, happy, sad, angry, fearful, surprise, and disgust. For this project, only seven of these classes are used.

## Video Model
For the video part of the emotion recognition, a convolutional neural network (CNN) architecture is used. The model consists of the following layers:

- Input layer: The input shape is (width_targ, height_targ, 1), representing the dimensions of the video frames and the number of channels (1 for grayscale).
- Data augmentation: The model applies data augmentation techniques such as random horizontal flipping to enhance the training data.
- Convolutional layers: The model includes several convolutional layers with different filter sizes and depths. Each convolutional layer uses the ELU activation function for non-linearity.
- Pooling layers: Max pooling is applied to reduce the spatial dimensions of the feature maps.
- Dropout layers: Dropout is used to regularize the model and prevent overfitting.
- Fully connected layers: The model includes dense layers with activation functions to further process the extracted features.
- Output layer: The output layer uses the softmax activation function to predict the probabilities of the different emotion classes.

## Audio Model
For the audio part of the emotion recognition, a convolutional neural network (CNN) architecture is used. The model consists of the following layers:

- Input layer: The input shape is (128, 282, 1), representing the spectrogram dimensions and the number of channels (1 for grayscale).
- Convolutional layers: The model includes several convolutional layers with different filter sizes and depths. Each convolutional layer uses the ReLU activation function to introduce non-linearity.
- Pooling layers: Max pooling is applied to reduce the spatial dimensions of the feature maps.
- Dropout layers: Dropout is used to regularize the model and prevent overfitting.
- Fully connected layers: The model includes dense layers with activation functions to further process the extracted features.
- Output layer: The output layer uses the softmax activation function to predict the probabilities of the different emotion classes.



## Credits
This project utilizes open-source libraries such as TensorFlow, Keras, and scikit-learn. The RAVDESS dataset is credited for providing the labeled speech clips used for training and evaluation.

## Disclaimer
Please note that the accuracy of the emotion recognition system depends on various factors, including data quality, model architecture, and training techniques. Consider the limitations and biases of AI-based approaches and perform thorough evaluation and validation before deploying the system in critical applications. Respect privacy and data protection guidelines while using this project.

