from keras.models import load_model
from keras.preprocessing.image import  load_img, image, ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
import numpy as np
import os

cwd =  os.path.dirname(os.path.realpath(__file__))

model = load_model(cwd + "/model_10-27-2019.h5")
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

test_gen = ImageDataGenerator(rescale= 1./255)
test_set = test_gen.flow_from_directory(cwd + "/data/test",
                                        target_size=(255, 255),
                                        batch_size=15,
                                        class_mode='binary')

scores = model.evaluate_generator(test_set)
print("Accuracy: ", scores[1])
