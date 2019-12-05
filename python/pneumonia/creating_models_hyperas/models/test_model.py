from keras.models import load_model
from keras.preprocessing.image import  load_img, image, ImageDataGenerator
import numpy as np
from glob import glob
import os
import sys

def test_from_directory(directory, model):
    predictions = []
    test_img_filepaths = glob(directory + "/*")

    for path in test_img_filepaths:
        test_image = load_img(path, target_size=(255, 255, 3))
        test_image = image.img_to_array(test_image)
        test_image = test_image * (1. / 255)
        test_image = np.expand_dims(test_image, axis=0)
        predictions.append(model.predict_classes(test_image)[0][0])
    
    return predictions, len(test_img_filepaths)


def main():
    current_file_directory =  os.path.dirname(os.path.realpath(__file__)) # Current file directory

    model_filename = ""
    if (len(sys.argv) != 2):
        print("No model filename specified")
        return

    model_filename = sys.argv[1]

    model_filename = current_file_directory + "/" +  model_filename
    test_directory_normal = current_file_directory + "/../../data/test/NORMAL"
    test_directory_pneumonia = current_file_directory + "/../../data/test/PNEUMONIA"

    model = load_model(model_filename)

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    normal_predictions, total_normal = test_from_directory(test_directory_normal, model)
    pneumonia_predictions, total_pneumonia = test_from_directory(test_directory_pneumonia, model)

    normal_predictions = len(normal_predictions) - sum(normal_predictions)
    pneumonia_predictions = sum(pneumonia_predictions)

    print("                 Actual Class")
    print("                Norm\tPneum")
    print("Predicted|Norm| {}\t{}\n  Class  |Pneu| {}\t{}\n".format(normal_predictions,
                                                                    total_pneumonia - pneumonia_predictions,
                                                                    total_normal - normal_predictions,
                                                                    pneumonia_predictions))

if __name__ == "__main__":
    main()