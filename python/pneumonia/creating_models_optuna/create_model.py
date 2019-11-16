from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator, load_img
import optuna
import math
import os
import time

data_path = os.path.dirname(os.path.realpath(__file__)) + "/../data"

test_data_filepath = data_path + "/test"
train_data_filepath = data_path + "/train"
val_data_filepath = data_path + "/val"

BATCH_SIZE = 50
NUM_EPOCHS = 20
NUM_TRAIN_SAMPLES = 1341 + 3875 # Normal + Pneumonia
NUM_VAL_SAMPLES = 8 + 8 # Normal + Pneumonia


INPUT_SHAPE = (255, 255, 3)

def objective(trial):
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

    model = Sequential()
    
    model.add(
        Conv2D( filters=trial.suggest_categorical('filters_1', [16, 32, 64]),
                kernel_size=trial.suggest_categorical('kernel_size_1', [2, 3, 4]),
                input_shape=INPUT_SHAPE,
                activation=trial.suggest_categorical('activation_1', ['relu', 'linear'])))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(trial.suggest_uniform('dropout_1', 0.0, 1.0)))

    # First hidden layer
    model.add(
        Conv2D( filters=trial.suggest_categorical('filters_2', [16, 32, 64]),
                kernel_size=trial.suggest_categorical('kernel_size_2', [2, 3, 4]),
                activation=trial.suggest_categorical('activation_2', ['relu', 'linear'])))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(trial.suggest_uniform('dropout_2', 0.0, 1.0)))

    # Second hidden layer
    model.add(
        Conv2D( filters=trial.suggest_categorical('filters_3', [16, 32, 64]),
                kernel_size=trial.suggest_categorical('kernel_size_3', [2, 3, 4]),
                activation=trial.suggest_categorical('activation_3', ['relu', 'linear'])))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(trial.suggest_uniform('dropout_3', 0.0, 1.0)))

    # Feed forward network ----------
    model.add(Flatten())
    model.add(Dense(trial.suggest_categorical('dense_layer_nodes_1', [16, 32, 64]), activation='relu'))

    # Output Layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    model.fit_generator(train_set,
                        steps_per_epoch=int(NUM_TRAIN_SAMPLES/BATCH_SIZE),
                        nb_epoch=NUM_EPOCHS,
                        validation_data=val_set,
                        validation_steps=math.ceil(NUM_VAL_SAMPLES / BATCH_SIZE))
  
    accuracy = model.evaluate_generator(test_set)[1]
    
    model.save(os.path.dirname(os.path.realpath(__file__)) + '/' + 'models/model_' + str(int(accuracy * 100 * 100)) + '_' + str(int(math.ceil(time.time()))) + '.h5')


    return -accuracy

def main():
    # study_name = "study_11-15"
    # study = optuna.create_study(study_name=study_name, storage='sqlite:///study_11-15.db')
    study = optuna.create_study()
    study.optimize(objective, n_trials = 2)
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_contour(study)
    optuna.visualization.plot_intermediate_values(study)
    optuna.visualization.plot_slice(study)
    optuna.visualization.plot_parallel_coordinate(study)


if __name__ == "__main__":
    main()