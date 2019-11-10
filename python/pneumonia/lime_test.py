from keras.models import load_model
from keras.preprocessing.image import  load_img, image, ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
import numpy as np
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
import os

cwd =  os.path.dirname(os.path.realpath(__file__))

model = load_model(cwd + "/model_10-27-2019.h5")
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

explainer = LimeImageExplainer()

path = 'C:\\Users\\pstanfel_a\\Documents\\CSCI 575\\Applied Project\\code\\0xDEADBEEFCAFE\\python\\pneumonia/data/test/PNEUMONIA\\person119_bacteria_567.jpeg'

test_image = load_img(path, target_size=(255,255,3))
test_image = image.img_to_array(test_image)
test_image = test_image * (1. / 255)
test_image = np.expand_dims(test_image, axis=0)

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

index = model.predict_classes(test_image)[0][0]
explanation = explainer.explain_instance(test_image[0], model.predict_proba, labels=["NORMAL", "PNEUMONIA"], top_labels=2, num_samples=100, random_seed=0)
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)

plt.imshow(mark_boundaries(temp, mask))

plt.show()