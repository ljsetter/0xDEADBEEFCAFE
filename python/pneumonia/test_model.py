from keras.models import load_model
from keras.preprocessing.image import  load_img, image, ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
import numpy as np

model = load_model("model.h5")
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

test_gen = ImageDataGenerator(rescale= 1./255)
test_set = test_gen.flow_from_directory("/home/mason/Documents/school/machine_learning/projects/applied_project/0xDEADBEEFCAFE/python/pneumonia/data/test",
                                        target_size=(255, 255),
                                        batch_size=15,
                                        class_mode='binary')

scores = model.evaluate_generator(test_set)
print("Accuracy: ", scores[1])
