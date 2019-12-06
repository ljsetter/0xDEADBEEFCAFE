# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..\\lung_masking'))
	print(os.getcwd())
except:
	pass
# %%
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import os
from utils.dice import dice_coef
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# %%
def dice_error(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


# %%
model = keras.models.load_model('unet_model_16_filters_kernel2x2.h5',custom_objects={'dice_coef':dice_coef,'dice_error':dice_error})


# %%
import glob
from keras.preprocessing.image import image, load_img, save_img

def mask_image(model, input_glob, output_path, label_header):
    """
    Mask an image using the UNet model.
    """
    crx_images = input_glob
    
    i = 0
    
    for crx_image in crx_images:
        img = load_img(crx_image, color_mode='grayscale', target_size=(128,128))
        img = image.img_to_array(img)
        img = img * (1. / 255)
        img = np.expand_dims(img, axis=0)

        sgm = model.predict(img).argmax(axis=3)[0]
        img_mask = sgm > 0
        img_mask = np.where(img_mask, img[...,0][0], 0)
        img_mask = np.reshape(img_mask, (128,128,1))

        path = output_path
        img_path = label_header + str(i) + '.jpg'
        path = path + img_path
        save_img(path, img_mask)
        i += 1


# %%
parent_path = os.path.dirname( os.getcwd() )

file_paths = ['\\train\\NORMAL\\', '\\train\\PNEUMONIA\\',               '\\test\\NORMAL\\', '\\test\\PNEUMONIA\\',               '\\val\\NORMAL\\', '\\val\\PNEUMONIA\\']

for i,file_path in enumerate(file_paths):
    input_path = glob.glob(parent_path + '\\data' + file_path + '*')
    output_path = parent_path + '\\lung_mask_data' + file_path

    if i % 2 == 0:
        header = "normal_crx"
    else:
        header = "pneumonia_crx"
        
    mask_image(model, input_path, output_path, header)
    


# %%


