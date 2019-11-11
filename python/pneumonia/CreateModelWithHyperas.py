from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator, load_img
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import uniform
import math
import os

data_path = os.path.dirname(os.path.realpath(__file__)) + "/data"

test_data_filepath = data_path + "/test"
train_data_filepath = data_path + "/train"
val_data_filepath = data_path + "/val"

BATCH_SIZE = 10
NUM_EPOCHS = 20
NUM_TRAIN_SAPLES = 1341 + 3875 # Normal + Pneumonia
NUM_VAL_SAMPLES = 8 + 8 # Normal + Pneumonia
INPUT_SHAPE = (255, 255, 3)

def data():
    data_path = os.path.dirname(os.path.realpath(__file__)) + "/data"
    test_data_filepath = data_path + "/test"
    train_data_filepath = data_path + "/train"
    val_data_filepath = data_path + "/val"

    BATCH_SIZE = 10
    NUM_EPOCHS = 20
    NUM_TRAIN_SAPLES = 1341 + 3875 # Normal + Pneumonia
    NUM_VAL_SAMPLES = 8 + 8 # Normal + Pneumonia
    INPUT_SHAPE = (255, 255, 3)

    train_gen = ImageDataGenerator( rescale=1./255,
                                    shear_range= 0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

    val_gen = ImageDataGenerator(rescale=1. / 255)

    train_set = train_gen.flow_from_directory(  train_data_filepath,
                                                target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                                batch_size=BATCH_SIZE,
                                                class_mode='binary')

    val_set = val_gen.flow_from_directory(  val_data_filepath,
                                            target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                            batch_size=BATCH_SIZE,
                                            class_mode='binary')

    test_gen = ImageDataGenerator(rescale= 1./255)
    test_set = test_gen.flow_from_directory(  test_data_filepath,
                                                target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1]),
                                                batch_size=BATCH_SIZE,
                                                class_mode='binary')

    return train_set, test_set, val_set

def model(train_set, test_set, val_set):
    data_path = os.path.dirname(os.path.realpath(__file__)) + "/data"
    test_data_filepath = data_path + "/test"
    train_data_filepath = data_path + "/train"
    val_data_filepath = data_path + "/val"

    BATCH_SIZE = 10
    NUM_EPOCHS = 20
    NUM_TRAIN_SAPLES = 1341 + 3875 # Normal + Pneumonia
    NUM_VAL_SAMPLES = 8 + 8 # Normal + Pneumonia
    INPUT_SHAPE = (255, 255, 3)

    model = Sequential()
    
    model.add(Conv2D(16, (3, 3), input_shape=INPUT_SHAPE, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout({{uniform(0, 1)}}))

    # First hidden layer
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout({{uniform(0, 1)}}))

    # Second hidden layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout({{uniform(0, 1)}}))

    # Feed forward network ----------
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    model.fit_generator(train_set,
                        steps_per_epoch=int(NUM_TRAIN_SAPLES/BATCH_SIZE) ,
                        nb_epoch=NUM_EPOCHS,
                        validation_data=val_set,
                        validation_steps=math.ceil(NUM_VAL_SAMPLES / BATCH_SIZE))
  
    accuracy = model.evaluate_generator(test_set)[1]
    return {'loss':-accuracy, 'status': STATUS_OK, 'model':model}


def main():
    best_run, best_model = optim.minimize(  model=model,
                                            data=data,
                                            algo=tpe.suggest,
                                            max_evals=10,
                                            trials=Trials())
    best_model.save('model_11-10-2019_hyperas.h5')
    print(best_run)


if __name__ == "__main__":
    main()