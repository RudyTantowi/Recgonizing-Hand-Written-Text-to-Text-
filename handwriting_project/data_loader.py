from scipy.io import loadmat
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_emnist():
    # Load the EMNIST dataset
    data = loadmat('emnist-byclass.mat')

    # Extract data
    x_train = data['dataset'][0][0][0][0][0][0]
    y_train = data['dataset'][0][0][0][0][0][1]
    x_test = data['dataset'][0][0][1][0][0][0]
    y_test = data['dataset'][0][0][1][0][0][1]

    # Normalize and reshape images
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, num_classes=62)
    y_test = to_categorical(y_test, num_classes=62)

    return (x_train, y_train), (x_test, y_test)
