from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img
import math
import os


data_path = os.path.dirname(os.path.realpath(__file__)) + "/data"

test_dir = data_path + "/test"
val_dir = data_path + "/val"
train_dir = data_path + "/train"

# Setup
batch_size = 10
num_epochs = 20
num_train_samples = 1341 + 3875 # Normal + Pneumonia
num_val_samples = 8 + 8 # Normal + Pneumonia

input_shape = (255, 255, 3)

print("Input Shape: ", input_shape)
print("Batch Size: ", batch_size)
print("Number of Epochs: ", num_epochs)
print("Number of training samples: ", num_train_samples)
print("Number of validation samples: ", num_val_samples)


model = Sequential()

# Convolutional Network --------
# Input layer
model.add(Conv2D(64, (3, 3), input_shape=input_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# First hidden layer
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second hidden layer
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Feed forward network ----------
model.add(Flatten())
model.add(Dense(64, activation='relu'))

# Output Layer
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()

train_gen = ImageDataGenerator( rescale=1./255,
                                shear_range= 0.2,
                                zoom_range=0.2,
                                horizontal_flip= True)

val_gen = ImageDataGenerator(rescale=1. / 255)

train_set = train_gen.flow_from_directory(  train_dir,
                                            target_size= (input_shape[0], input_shape[1]),
                                            batch_size=batch_size,
                                            class_mode='binary')

val_set = val_gen.flow_from_directory(  val_dir,
                                        target_size=(input_shape[0], input_shape[1]),
                                        batch_size=batch_size,
                                        class_mode='binary')

model.fit_generator(train_set,
                    steps_per_epoch=int(num_train_samples/batch_size) ,
                    nb_epoch=num_epochs,
                    validation_data=val_set,
                    validation_steps=math.ceil(num_val_samples / batch_size))

model.save('model_10-28-2019_1.h5')

test_gen = ImageDataGenerator(rescale= 1./255)
test_set = test_gen.flow_from_directory(test_dir,
                                        target_size=(input_shape[0], input_shape[1]),
                                        batch_size=batch_size,
                                        class_mode='binary')

accuracy = model.evaluate_generator(test_set)
print("Accuracy: ", accuracy)


