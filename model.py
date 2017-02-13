import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
from sklearn.utils import shuffle
import tensorflow as tf
import os
import matplotlib.image as mpimg
from scipy.misc import imresize
import pickle
import matplotlib.pyplot as plt

#  Keras imports
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Convolution2D, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
# I was gettng overflow errors. To fix it, added the below code
K.set_image_dim_ordering('tf')


### The below code is executed only once, to create a pickle file.Loading driving log 
##driving_log = pd.read_csv('driving_log.csv', sep=',', header = None)
##
### Creating list of center images and steering angles
##center_images = []
##angles = []
##
##for x in range(1, len(driving_log)):
##    center_image = driving_log.get_value(x, 0)
##    angle = driving_data.get_value(x, 3)
##    center_images.append(center_image)
##    angles.append(angle)
##
### 
##print('Loading images...This process took extremely long on my CPU. Windows i7 2.6 GHZ 16 GB Ram ')
##print(len(center_images))
##print('reading images to memory')
##images = []
##for i in center_images:
##    image = mpimg.imread(i)
##    images.append(image)
##
### Make into numpy arrays
### The float part helps for visualizing at this point, if desired
##images = np.array(images).astype(np.float)
##angles = np.array(angles).astype(np.float)
##
##print('All images loaded.')
##
### Re-sizing images to quarter the size
##def resize(img):
##    img = imresize(img, (40, 80, 3))
##    return img
##
##def resizing(images):
##    resized_images = []
##    for image in images:
##        r_img = resize(image)
##        resized_images.append(r_img)
##        
##    return np.array(resized_images)
##
### Run all images through re-sizing
##print('Resizing images takes a very long time on my CPU')
##images = resizing(images)
##
##print('All images resized.')
##
### Change to typical training naming conventions
##X_train, y_train = images, angles
##
### Shuffling first, then splitting into training and validation sets
##X_train, y_train = shuffle(X_train, y_train)
##train_data = {"features":X_train,"labels":y_train}
##validation_data = {"features":X_val,"labels":y_val}

## Pickling the driving data
##pickle.dump( train_data, open( "driving_train.p", "wb" ) )


# Loading the Pickled training data. I did not have a separate pickled validation data. I am creating validation data by
# splitting the training data. Please see below.
print('Reading Pickled Files')
with open('driving_train.p', 'rb') as f:
        train_data = pickle.load(f)
X_train = train_data['features']
y_train = train_data['labels']

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=23)

# Data Visualization
plt.hist(y_train)
plt.ylabel('Count')
plt.xlabel('Steering Angle')
#plt.show()

# Creating the Neural network
print('Neural network initializing.')

# Looked at various implementations, read online to see what's the best convolution architecute to use
# It seems like have BatchNormalization to reduce internal covariate shift. I read online that it is a good idea to do this.
batch_size = 100
epochs = 10
pool_size = (2, 2)
input_shape = X_train.shape[1:]
print(input_shape)

model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
# I tried using multiple convolution layers, but having 4 convolutions as 
# you can see below, was turning out to be extremely slow on my CPU (Windows i7, 2.6GHz and 16GB memory)
# I wasn't even sure if this problem can solved on a CPU. I took out additional convolutions and just stuck to one convolution layer of 32 filters

### Convolutional Layer 1 and Dropout
##model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample=(1,1)))
##model.add(Activation('relu'))
##model.add(Dropout(0.2))

# Conv Layer 2
model.add(Convolution2D(32, 3, 3, border_mode='valid', subsample=(1,1)))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=pool_size))
# Dropout to avoid overfitting
model.add(Dropout(0.2))

### Conv Layer 3
##model.add(Convolution2D(16, 3, 3, border_mode='valid', subsample=(1,1)))
##model.add(Activation('relu'))
##
### Conv Layer 4
##model.add(Convolution2D(8, 3, 3, border_mode='valid', subsample=(1,1)))
##model.add(Activation('relu'))



# Flattening and adding Dropout
model.add(Flatten())
model.add(Dropout(0.5))

# Fully Connected Layer 1 and Dropout
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Fully Connected Layer 2
model.add(Dense(64))
model.add(Activation('relu'))

# Fully Connected Layer 3
model.add(Dense(32))
model.add(Activation('relu'))

# Adding the last layer with just one output as this is a regression problem and we need to predict the steering angle
model.add(Dense(1))

# Model Compilation 
model.compile(metrics=['mean_squared_error'], optimizer='adam', loss='mean_squared_error')

#Model fitting
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=2, validation_data=(X_val, y_val))

# Save model architecture and weights
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

# Model summary
model.summary()

