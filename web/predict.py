
"""Example run of program from command line

    $ python3 predict.py IM-0007-0001.jpeg masked.png pred.txt

    This would run the prediction script on the input image 'IM-0007-0001.jpeg',
    saving the masked image to 'masked.png', and saving the prediction to 'pred.txt'.
"""

from keras.models import load_model
from keras.preprocessing.image import  load_img, image
import numpy as np
from PIL import Image
import math
import os
import sys
from lime.lime_image import LimeImageExplainer
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries

def load_image(input_filename):
    """ Loads an rgb image and converts to correct format

    input_filename  -> filename of the image to load

    return          -> Image loaded
    """

    img = load_img(input_filename, target_size=(255,255,3))
    img = image.img_to_array(img)
    img = img * (1. / 255)
    img = np.expand_dims(img, axis=0)

    return img


def predict_class(img, model):
    """ Predict the class that the image belongs to

    img     -> Image to predict
    model   -> Keras model to use for prediction

    return  -> 'Normal' if precition is normal, 'Pneumonia' if prediction is pneumonia (string)
    """

    return 'Normal' if model.predict_classes(img)[0][0] == 0 else 'Pneumonia'


def max_N_weight(local_exp, N=4):
    """ Get the Nth biggest weight out of an a local_exp array
    
    local_exp   -> local_exp from the explainer
    N           -> Nth biggest wight to use (biggest weight is returned if N > len(local_exp[0]))

    return      -> Nth biggest weight (or biggest weight if N > len(local_exp[0]))
    """

    weights = []
    for tp in local_exp[0]:
        weights.append(tp[1])
    weights.sort()

    if len(weights) >= N:
        return weights[-N]
    elif len(weights) != 0:
        return weights[-1]
    else:
        return math.inf


def mark_boundary(img, mask):
    """ Mark the boundaries of the mask

    img     -> Image to mask
    mask    -> Mask to get the boundaries from

    return  -> Image with the boundaries marked

    """

    img_array = mark_boundaries(img, mask)
    return Image.fromarray((255.0 / img_array.max() * (img_array - img_array.min())).astype(np.uint8))


def lime(img, model):
    """ Apply Lime to an image, marking the boundaries of the areas most used to make the prediction

    img     -> Image to use in the model
    model   -> Keras model used to make the prediction

    return  -> Image with the boundaries of the areas most used in making the prediction marked 
    """

    segmentation_fn = SegmentationAlgorithm('quickshift',   kernel_size=4,
                                                            max_dist=200, ratio=0.2,
                                                            random_seed=0)

    explainer = LimeImageExplainer()

    explanation = explainer.explain_instance(   img[0],
                                                model.predict_proba,
                                                labels=["NORMAL", "PNEUMONIA"], 
                                                top_labels=2,
                                                num_samples=100,
                                                random_seed=0,
                                                segmentation_fn = segmentation_fn)

    mask_weight = max_N_weight(explanation.local_exp)

    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                positive_only=True,
                                                hide_rest=False,
                                                num_features=100,
                                                min_weight=mask_weight)

    masked_image = mark_boundary(temp, mask)

    return masked_image


def predict(input_filename, model):
    """ Predicts the class of an image and marks the boundaries of the areas most important in that prediction

    input_filename  -> Filename of the image to predict from
    model           -> Keras model used to predict

    return          -> Image with the areas most important in the prediction marked
                    -> Prediction ('Normal' or 'Pneumonia')

    """

    test_image = load_image(input_filename)
    prediction = predict_class(test_image, model)
    masked_image = lime(test_image, model)

    return masked_image, prediction


def main():
    assert(len(sys.argv) == 4)
    input_filename = sys.argv[1]
    output_image_filename = sys.argv[2]
    output_prediction_filename = sys.argv[3]

    cwd = os.path.dirname(os.path.realpath(__file__))
    model = load_model(cwd + "/model.h5")
    input_filename = cwd + '/' + input_filename
    output_image_filename = cwd + '/' + output_image_filename

    img, prediction = predict(input_filename, model)
    
    img.save(output_image_filename)

    with open(output_prediction_filename, 'w') as file:
        file.write(prediction)


if __name__ == "__main__":
    main()
