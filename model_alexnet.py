from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Flatten, Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf

def AlexNet(input_shape = (64, 64, 3), num_classes = 5):
    model = tf.keras.Sequential([
        # Convolutional Layer 1

        Conv2D(96, kernel_size=11, strides=4, activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=2, strides=1),
        
        # Convolutional Layer 2
        Conv2D(256, kernel_size=5, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2, strides=2),
        
        # Convolutional Layer 3
        Conv2D(384, kernel_size=3, padding='same', activation='relu'),
        
        # Convolutional Layer 4
        Conv2D(384, kernel_size=3, padding='same', activation='relu'),
        
        # Convolutional Layer 5
        Conv2D(256, kernel_size=3, padding='same', activation='relu'),
        MaxPooling2D(pool_size=2, strides=1),
        
        # Flatten the feature maps
        Flatten(),
        
        # Fully connected Layer 1
        Dense(4096, activation='relu'),
        
        # Fully connected Layer 2
        Dense(4096, activation='relu'),
        
        # Output Layer
        Dense(num_classes, activation='softmax')
    ])
    return model