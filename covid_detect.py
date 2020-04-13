# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 09:15:58 2020

@author: senthil kumar
"""

import cv2
img=cv2.imread('C:/Users/senthil kumar/Desktop/COVID/TRAIN/COVID_19/2020.02.26.20026989-p34-114_1%0.png')
img1=cv2.imread('C:\Users\senthil kumar\Desktop\COVID\TRAIN\NON_COVID\31.jpg')
cv2.imshow('COVID19 Positive Sample case', img)
cv2.imshow('COVID 19 Negative Sample Case',img1)

# Building the CNN
import tensorflow as tf
# Importing the Keras libraries and packages

from tensorflow.keras import layers, models, optimizers, callbacks

classifier = models.Sequential()

# Step 1 - Convolution
classifier.add(layers.Conv2D(32, (3, 3), input_shape = (256, 256, 1), activation = 'relu'))

# Step 2 - Pooling
classifier.add(layers.MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(layers.MaxPooling2D(pool_size = (2, 2)))

## Adding a third convolutional layer
classifier.add(layers.Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(layers.MaxPooling2D(pool_size = (2, 2)))

classifier.add(layers.Conv2D(16, (3, 3), activation = 'relu'))
classifier.add(layers.MaxPooling2D(pool_size = (2, 2)))



# Flattening
classifier.add(layers.Flatten())

# Full connection
classifier.add(layers.Dense(units = 1024, activation = 'relu'))
classifier.add(layers.Dense(units = 512, activation = 'relu'))

classifier.add(layers.Dense(units = 2, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy','AUC'])

# Fitting the CNN to the images

#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('C:\\Users\\senthil kumar\\Desktop\\COVID\\TRAIN',
                                                 target_size = (256, 256),
                                                 batch_size = 8,
                                                 class_mode = 'categorical',
                                                 color_mode='grayscale',
                                                 shuffle=True)

test_set = test_datagen.flow_from_directory('C:\\Users\\senthil kumar\\Desktop\\COVID\\TEST',
                                            target_size = (256, 256),
                                            batch_size = 8,
                                            class_mode = 'categorical',
                                            color_mode='grayscale',
                                            shuffle=False)

nb_train_samples=539
nb_validation_samples=207
batch_size=8

# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

history=classifier.fit_generator(training_set,
                         steps_per_epoch = nb_train_samples//batch_size,
                         epochs = 30,
                         validation_data = test_set,
                         validation_steps = nb_validation_samples//batch_size,callbacks=callbacks_list)



from matplotlib import pyplot as plt
plt.figure()
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.figure()
plt.plot(history.history['accuracy'],label='Train Accuracy')
plt.plot(history.history['val_accuracy'],label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.figure()
plt.plot(history.history['AUC'],label='Train AUC')
plt.plot(history.history['val_AUC'],label='Validation AUC')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('AUC')


