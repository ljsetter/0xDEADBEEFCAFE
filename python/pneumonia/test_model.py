from keras.models import load_model
from keras.preprocessing.image import  load_img, image, ImageDataGenerator
import numpy as np
from glob import glob
import os

def test_from_directory(directory, model):
    predictions = []
    test_img_filepaths = glob(directory + "/*")


    for path in test_img_filepaths:
        test_image = load_img(path, target_size=(255, 255, 3))
        test_image = image.img_to_array(test_image)
        test_image = test_image * (1. / 255)
        test_image = np.expand_dims(test_image, axis=0)
        predictions.append(model.predict_classes(test_image)[0][0])
    
    return predictions

cwd =  os.path.dirname(os.path.realpath(__file__))

model = load_model(cwd + "/model_10-27-2019.h5")
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

normal_predictions = test_from_directory(cwd + "/data/test/NORMAL", model)
pneumonia_predictions = test_from_directory(cwd + "/data/test/PNEUMONIA", model)
    
print("Normal Predictions: ", len(normal_predictions) - sum(normal_predictions))
print("Pneumonia Predictions: ", sum(pneumonia_predictions))

