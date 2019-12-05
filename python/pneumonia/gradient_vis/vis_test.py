from keras.models import load_model
from keras import backend as K
from keras.preprocessing.image import  load_img, image
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.misc
import matplotlib.image as mpimage

cwd =  os.path.dirname(os.path.realpath(__file__))

model = load_model(cwd + "/../creating_models/models/model_11-9-2019.h5")
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

paths = []
paths.append('../data/test/NORMAL/NORMAL2-IM-0232-0001.jpeg')
paths.append('../data/test/NORMAL/NORMAL2-IM-0346-0001.jpeg')
paths.append('../data/test/NORMAL/NORMAL2-IM-0130-0001.jpeg')
paths.append('../data/test/NORMAL/NORMAL2-IM-0381-0001.jpeg')
paths.append('../data/test/PNEUMONIA/person135_bacteria_647.jpeg')
paths.append('../data/test/PNEUMONIA/person87_bacteria_433.jpeg')
paths.append('../data/test/PNEUMONIA/person136_bacteria_652.jpeg')
paths.append('../data/test/PNEUMONIA/person93_bacteria_453.jpeg')
paths.append('../data/test/PNEUMONIA/person1670_virus_2886.jpeg')
paths.append('../data/test/PNEUMONIA/person1680_virus_2897.jpeg')
paths.append('../data/test/PNEUMONIA/person1663_virus_2876.jpeg')
paths.append('../data/test/PNEUMONIA/person1656_virus_2862.jpeg')

for i, path in enumerate(paths):
    test_image = load_img(path, target_size=(255,255,3))
    test_image = image.img_to_array(test_image)
    test_image = test_image * (1. / 255)
    test_image = np.expand_dims(test_image, axis=0)

    y_prob = model.predict(test_image)
    print("y_prob: ", y_prob)

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_name = "dense_14"

    layer_output = layer_dict[layer_name].output
    loss = layer_output

    input_img = model.input

    grads = K.gradients(layer_output, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    func = K.function([input_img], [grads])


    test_grad = func(test_image)[0]

    test_grad = np.absolute(test_grad)
    test_grad = np.power(test_grad, 1/2)
    test_grad /= test_grad.max()

    x = (test_image*255).squeeze().astype(np.uint8)
    g = (test_grad*255).squeeze().astype(np.uint8)

    plt.figure(0)
    plt.subplot(3,4,i+1)
    plt.imshow(x)
    plt.figure(1)
    plt.subplot(3,4,i+1)
    plt.imshow(g)
    plt.figure(2)
    plt.subplot(3,4,i+1)
    plt.hist(test_grad.flatten(), bins=100)

plt.show()
