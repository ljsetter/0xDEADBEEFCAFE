# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'python'))
	print(os.getcwd())
except:
	pass
# %%
"""
This script tests lime and can use both a UNet to segment the image or no UNet by setting the use_unet boolean. The UNet is from reference [9] from the presentation.
"""


# %%
use_unet = True


# %%
import keras
from keras.models import load_model
from keras.preprocessing.image import  save_img, load_img, image, ImageDataGenerator
import matplotlib.pyplot as plt
from keras.models import Sequential
from python.lime.utils.dice import dice_coef
import numpy as np
from lime.lime_image import LimeImageExplainer
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
import os


# %%
cwd =  os.path.abspath('')

model = load_model(cwd + "/python/model_creating/models/masked_model.h5")
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# %%
def dice_error(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


# %%
unet_model = load_model('python/lime/unet_model_16_filters_kernel2x2.h5',custom_objects={'dice_coef':dice_coef,'dice_error':dice_error})


# %%
random_seed = 0
segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)

# segmentation_fn = SegmentationAlgorithm('felzenszwalb', scale=1,
#                                                     sigma=0.8, min_size=20)


# %%
# test images saved in the python directory

#path = os.getcwd() + '\\data\\test\\PNEUMONIA\\person47_virus_99.jpeg'

path = os.getcwd() + '\\python\\Pneumonia-right-middle-lobe-4.jpg'

#path = os.getcwd() + '\\python\\Pneumonia-CXR.png'

#path = os.getcwd() + '\\76052f7902246ff862f52f5d3cd9cd_big_gallery.jpg'

#path = os.getcwd() + '\\xray-chest-pneumonia.jpg'

# load normal image
test_image = load_img(path, target_size=(255,255,3))
test_image = image.img_to_array(test_image)
test_image = test_image * (1. / 255)
test_image = np.expand_dims(test_image, axis=0)

if use_unet:
    # if UNet is being used, mask the image and resave it
    img = load_img(path, color_mode='grayscale', target_size=(128,128))
    img = image.img_to_array(img)
    img = img * (1. / 255)
    img = np.expand_dims(img, axis=0)

    sgm = unet_model.predict(img).argmax(axis=3)[0]
    img_mask = sgm > 0
    img_mask = np.where(img_mask, img[...,0][0], 0)
    img_mask = np.reshape(img_mask, (128,128,1))

    mask_path = os.getcwd() + '\\python\\masked_image.jpg'

    save_img(mask_path, img_mask)

    lime_image = load_img(mask_path, target_size=(255,255,3))
    lime_image = image.img_to_array(lime_image)
    lime_image = lime_image * (1. / 255)
    lime_image = np.expand_dims(lime_image, axis=0)
else:
    lime_image = test_image


explainer = LimeImageExplainer()


# %%
explanation = explainer.explain_instance(lime_image[0], model.predict_proba, labels=["NORMAL", "PNEUMONIA"], 
                                         top_labels=2, num_samples=100, random_seed=0, segmentation_fn = segmentation_fn)


# %%
# 0 is normal, 1 is pneumonia
print(model.predict_classes(test_image))

temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True,hide_rest=False, num_features=100, min_weight=6e-6)
plt.imshow(mark_boundaries(test_image[0], mask))
plt.show()


# %%
explanation.local_exp

