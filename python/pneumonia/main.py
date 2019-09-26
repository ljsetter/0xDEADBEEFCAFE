from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()

model.add(Conv2D(32, 3, input_shape=(64, 64, 3), activation='relu') )
model.add(MaxPooling2D(pool_size=(2, 2)) )
model.add(Conv2D(32, 3) )
model.add(MaxPooling2D(pool_size=(2, 2)) )
model.add(Flatten() )
model.add(Dense(128, activation='relu') )
model.add(Dense(1, activation='sigmoid') )

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_gen = ImageDataGenerator( rescale=1./255,
                                shear_range= 0.2,
                                zoom_range=0.2,
                                horizontal_flip= True)

test_gen = ImageDataGenerator(rescale= 1./255)

train_set = train_gen.flow_from_directory(  'python/pneumonia/data/train/',
                                            target_size= (64, 64),
                                            batch_size=32,
                                            class_mode='binary')

test_set = test_gen.flow_from_directory(    'python/pneumonia/data/test/',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

model.fit_generator(train_set,
                    samples_per_epoch=8000,
                    nb_epoch=25,
                    validation_data=test_set,
                    nb_val_samples=2000)