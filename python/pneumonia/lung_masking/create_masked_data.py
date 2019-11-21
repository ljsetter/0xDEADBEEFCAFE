import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import os

from utils.dice import dice_coef

def dice_error(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

model = keras.models.load_model('unet_model_16_filters_kernel2x2.h5',custom_objects={'dice_coef':dice_coef,'dice_error':dice_error})

train_gen = ImageDataGenerator()

BATCH_SIZE = 50

INPUT_SHAPE = (128, 128, 3)

train_data_filepath = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/data" + "/test"
print(train_data_filepath)
train_set = train_gen.flow_from_directory(  train_data_filepath,
                                                target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                                batch_size=BATCH_SIZE,
                                                class_mode='binary')

image = next(train_set)

print(np.shape(image))