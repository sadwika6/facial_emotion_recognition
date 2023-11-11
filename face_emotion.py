# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 09:39:17 2023

@author: sadwika sabbella
"""

import os
#import matplotlib.pyplot as plt


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop  
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model


#Directory with angry images
Angry_dir= os.path.join(r'D:\DL\ds\face\angry')

# Directory with Happy images
Happy_dir= os.path.join(r'D:\DL\ds\face\happy')

# Directory with Neutral images
Neutral_dir = os.path.join(r'D:\DL\ds\face\neutral')

# Directory with Sad images
Sad_dir = os.path.join(r'D:\DL\ds\face\sad')

# Directory with Surprise images
Surprise_dir = os.path.join(r'D:\DL\ds\face\surprise')

train_Angry_names = os.listdir(Angry_dir)
print(train_Angry_names[:5])

train_Happy_names = os.listdir(Happy_dir)
print(train_Happy_names[:5])
batch_size = 256

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(r'D:\DL\ds\face',  # This is the source directory for training images
        target_size=(48, 48),  # All images will be resized to 48 x 48
        batch_size=batch_size,
        color_mode='grayscale',
        
        
        # Specify the classes explicitly
        classes = ['Angry','Happy','Neutral','Sad','Surprise'],
        # Since we use categorical_crossentropy loss, we need categorical labels
        class_mode='categorical')
target_size=(48,48)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 48*48 with 3 bytes color

     # The first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    # Flatten the results to feed into a dense layer
    tf.keras.layers.Flatten(),
    # 64 neuron in the fully-connected layer
    tf.keras.layers.Dense(64, activation='relu'),
    
    
    # 5 output neurons for 5 classes with the softmax activation
    tf.keras.layers.Dense(5,activation='softmax')
])
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['acc'])#RMSprop(lr=0.001)
# Total sample count
total_sample=train_generator.n
# Training
num_epochs = 10
model.fit_generator(train_generator,steps_per_epoch=int(total_sample/batch_size),
                    epochs=num_epochs,verbose=1)

# serialize model to JSON
model_json = model.to_json()
with open("D:\DL\models\model_face1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("D:\DL\models\model1_face1.h5")
print("Saved model to disk")
