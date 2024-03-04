from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier

from model_alexnet import AlexNet
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

import pandas as pd
import numpy as np
import datetime
import pathlib
import scipy
import cv2
import os


data_dir = pathlib.Path("./my_flower_photos")

CLASS_NAMES = sorted(item.name for item in data_dir.glob('*') if item.is_dir())
print(CLASS_NAMES)

# print length of class names
output_class_units = len(CLASS_NAMES)
print(output_class_units)

# Create an instance of the AlexNet model
model = AlexNet()
model.compile(optimizer = 'adagrad', loss = "categorical_crossentropy", metrics = ['accuracy'])

# Print model summary
model.summary()

# Get file paths for all images
all_image_paths = [str(path) for path in data_dir.glob('*/*')]
# Extract labels from the file paths
labels = [pathlib.Path(path).parent.name for path in all_image_paths]

# Create a DataFrame containing image paths and labels
df = pd.DataFrame({'image_paths': all_image_paths, 'labels': labels})

# Split the DataFrame into training and testing sets
train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 42)

train_img_count = len(train_df)
test_img_count = len(test_df)
BATCH_SIZE = 16             # Can be of size 2^n, but not restricted to. for the better utilization of memory
IMG_HEIGHT = 64             # input Shape required by the model
IMG_WIDTH = 64              # input Shape required by the model
train_steps_per_epoch = np.ceil(train_img_count/BATCH_SIZE) # equals len(train_data_gen)

# Create instances of ImageDataGenerator
train_data_gen = ImageDataGenerator()
test_data_gen = ImageDataGenerator()

# Configure data generators using flow_from_dataframe
train_generator = train_data_gen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_paths',
    y_col='labels',
    batch_size=BATCH_SIZE,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    classes=list(CLASS_NAMES),
    drop_remainder=False
)

test_generator = test_data_gen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image_paths',
    y_col='labels',
    batch_size=BATCH_SIZE,
    shuffle=False,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    classes=list(CLASS_NAMES),
    drop_remainder=False
)
# Defining callback to stop training early
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get("accuracy") >= 0.99 and logs.get("loss")<0.03): # logs.get("val_accuracy") is not None and logs.get("val_accuracy") >= 0.95
            print("\nReached 100% accuracy so stopping training")
            self.model.stop_training =True
            
callbacks = myCallback()

# To monitor training
log_dir="logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir, histogram_freq = 1)

# Training
history = model.fit(
      train_generator,
      steps_per_epoch = train_steps_per_epoch,
      epochs = 50,
      callbacks = [tensorboard_callback,callbacks])

#Saving the model
model.save('AlexNet_saved_model/')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

new_model = tf.keras.models.load_model("AlexNet_saved_model/")
new_model.summary()

loss, acc = model.evaluate(test_data_gen)
print("accuracy:{:.2f}%".format(acc*100))