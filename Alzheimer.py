import os 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, TimeDistributed 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from PIL import Image 
from flask import Flask, render_template, request, jsonify 
 
 
# Dataset paths 
main_dir = "E:\Capstone\AD\AugmentedAlzheimerDataset"
categories = { 
    "NonDemented": os.path.join(main_dir, "NonDemented"), 
    "MildDemented": os.path.join(main_dir, "MildDemented"), 
    "VeryMildDemented": os.path.join(main_dir, "VeryMildDemented"), 
    "ModerateDemented": os.path.join(main_dir, "ModerateDemented") 
} 
 
 
 
# Image data generator for augmenting and loading the data 
datagen = ImageDataGenerator( 
    rescale=1.0 / 255, 
    validation_split=0.2 
) 
 
# Training data generator 
train_data = datagen.flow_from_directory( 
    main_dir, 
    target_size=(128, 128), 
    batch_size=32, 
    class_mode='categorical', 
    subset='training' 
) 
 
# Validation data generator 
val_data = datagen.flow_from_directory( 
    main_dir, 
    target_size=(128, 128), 
    batch_size=32, 
    class_mode='categorical', 
    subset='validation' 
) 
 
 
# CNN Model 
cnn_model = Sequential([ 
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)), 
    MaxPooling2D(pool_size=(2, 2)), 
    Conv2D(64, (3, 3), activation='relu'), 
    MaxPooling2D(pool_size=(2, 2)), 
    Flatten(), 
    Dense(128, activation='relu'), 
    Dropout(0.5), 
    Dense(4, activation='softmax')  # 4 classes 
]) 
 
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', 
metrics=['accuracy']) 
cnn_model.summary() 
 
 
 
# Train CNN model 
cnn_history = cnn_model.fit( 
    train_data, 
    validation_data=val_data, 
    epochs=10, 
    verbose=1 
) 
2 
# Save the CNN model 
cnn_model.save('cnn_model.h5') 
 
 
 
# Generate CNN predictions 
train_cnn_features = cnn_model.predict(train_data, 
steps=len(train_data))  # Shape: (num_samples, num_classes) 
val_cnn_features = cnn_model.predict(val_data, 
steps=len(val_data))        # Shape: (num_samples, num_classes) 
 
# Reshape CNN predictions to work with the RNN 
train_rnn_input = train_cnn_features.reshape(-1, 1, 
train_cnn_features.shape[1])  # Shape: (num_samples, 1, num_classes) 
val_rnn_input = val_cnn_features.reshape(-1, 1, 
val_cnn_features.shape[1])        # Shape: (num_samples, 1, num_classes) 
 
# Define the RNN model 
rnn_model = Sequential([
    TimeDistributed(Dense(64), input_shape=(1, 4)),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])

 
# Compile the RNN model 
rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', 
metrics=['accuracy']) 
rnn_model.summary() 
 
# Convert labels to one-hot encoding 
train_labels = tf.keras.utils.to_categorical(train_data.labels, 
num_classes=4) 
val_labels = tf.keras.utils.to_categorical(val_data.labels, num_classes=4) 
 
# Train the RNN 
rnn_history = rnn_model.fit( 
    train_rnn_input, 
    train_labels, 
    validation_data=(val_rnn_input, val_labels), 
    epochs=10, 
    verbose=1 
) 
 
  
rnn_model.save('rnn_model.h5') 
 
 
 
