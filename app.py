from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

img_width, img_height = 255, 255

pasta_treinamento = 'Treinamento'
pasta_validacao = 'Teste'

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
        pasta_treinamento,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')
validation_generator = datagen.flow_from_directory(
        pasta_validacao,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3, input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3, input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossntropy', optimizer='rmsprop', metrics=['accuracy'])


model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=25)

model.save_weights('treinamento.h5')
