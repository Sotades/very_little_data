from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import regularizers
from keras import layers

# dimensions of our images.
img_width, img_height = 210, 1280
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 9728
nb_validation_samples = 2428

epochs = 2

batch_size = 16



if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)

else:
    input_shape = (img_width, img_height, 3)

model = Sequential()

# Add a dropout layer for input layer
model.add(layers.Dropout(0.2, input_shape=input_shape))

model.add(Conv2D(32, (3, 3), input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

# Add fully connected layer with a ReLU activation function
model.add(layers.Dense(units=64, activation='relu'))

# Add a dropout layer for previous hidden layer
model.add(layers.Dropout(0.5))

# Add fully connected layer with a sigmoid activation function
model.add(layers.Dense(units=1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])



# this is the augmentation configuration we will use for training

train_datagen = ImageDataGenerator(rescale=1. / 255)

# this is the augmentation configuration we will use for testing:

# only rescaling

test_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

    validation_data_dir,

    target_size=(img_width, img_height),

    batch_size=batch_size,

    class_mode='binary')



model.fit_generator(

    train_generator,

    steps_per_epoch=nb_train_samples // batch_size,

    epochs=epochs,

    validation_data=validation_generator,

    validation_steps=nb_validation_samples // batch_size)

# serialize model to JSON
model_json = model.to_json()
with open("conv3_relu1_maxpool1.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights('conv3_relu1_maxpool1.h5')