MNIST Digit Recognizer with Neural Network and Streamlit Visualization
This document provides a comprehensive guide to the Python code for a MNIST digit recognizer implemented with a neural network and visualized using Streamlit.

1. Introduction

This project implements a neural network model to classify handwritten digits from the MNIST dataset. It then utilizes Streamlit to create a user-friendly web application for digit recognition. Users can upload an image of a handwritten digit, and the model predicts the corresponding digit.

2. Code Breakdown

2.1 Dependencies

The code requires the following Python libraries:

torch: Deep learning framework
torchvision: Library for computer vision tasks with PyTorch
streamlit: Framework for creating web apps in Python

2.2 Part 1: Neural Network Implementation

2.2.1 Imports and Setup

Imports necessary libraries.
Defines the device to be used for computations (CPU or GPU).

2.2.2 Data Loading and Preprocessing

Defines data transformations for image normalization.
Loads the MNIST training and testing datasets.
Creates data loaders for efficient training.

2.2.3 Neural Network Model (Multilayer Perceptron)

Defines a class MNISTNet for the neural network architecture.
The network consists of three layers:
Input layer with 784 neurons (28x28 pixels)
Hidden layer with 128 neurons with ReLU activation
Output layer with 10 neurons (one for each digit)
Defines the forward pass for data propagation through the network.

2.2.4 Training and Evaluation

Creates an instance of the MNISTNet model and moves it to the chosen device.
Defines the loss function (Cross-Entropy) and optimizer (Adam).
Trains the model for a specified number of epochs.
Calculates the accuracy on the test dataset after training.
Saves the trained model for later use.
2.3 Part 2: Streamlit Visualization

2.3.1 Streamlit App Setup

Imports the streamlit library.
Loads the trained model from a saved file.
Defines a function predict_digit to predict the digit from an uploaded image.

2.3.2 User Interface

Displays a title "MNIST Digit Recognizer" for the app.
Creates a file uploader for users to upload an image of a handwritten digit (supports JPEG and PNG formats).
Displays the uploaded image on the web app.
Provides a button "Predict Digit" that triggers prediction when clicked.
Upon clicking the button, the predict_digit function is called, predicting the digit and displaying the result.
3. How to Use

3.1 Running the App

Save the code in a Python file (e.g., mnist_app.py).
Install Streamlit using pip install streamlit (if not already installed).
Open a terminal and navigate to the directory where you saved the file.
Run the command: streamlit run mnist_app.py
This will launch the Streamlit app in your web browser, allowing you to upload images and get digit predictions.
3.2 Make Sure You Have:

A trained model saved as mnist_model.pt. You can train it by running the code without the Streamlit part.
All required libraries installed (torch, torchvision, streamlit).
4. Conclusion

This code demonstrates the combined power of neural networks and Streamlit for creating an interactive digit recognition app. Users can easily interact with the model and visualize its performance. You can further enhance the app by adding features like displaying the predicted probability distribution for each digit category.
