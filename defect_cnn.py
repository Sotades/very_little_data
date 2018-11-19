from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard
from keras import backend as K
from keras import regularizers
from keras import layers


# dimensions of our images.
img_width, img_height = 210, 461
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 5480
nb_validation_samples = 520

epochs = 2
lamda = 5E-5
batch_size = 16

# Set up TensorBoard
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)

else:
    input_shape = (img_width, img_height, 3)

model = Sequential()

#####DEFECT MODEL START#####

# Add a dropout layer for input layer
model.add(layers.Dropout(0.2, input_shape=input_shape))

# Convolution layer: 32 filters, kernal size 3 x 3, L2 regularization
model.add(Conv2D(32, (3, 3), input_shape=input_shape, kernel_regularizer=regularizers.l2(lamda)))
model.add(Activation('relu'))

# Pooling layer: subsampling 2 x 2, stride 2
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# Convolution layer: 64 filters, kernal size 3 x 3, L2 regularization
model.add(Conv2D(64, (3, 3), input_shape=input_shape, kernel_regularizer=regularizers.l2(lamda)))
model.add(Activation('relu'))

# Convolution layer: 64 filters, kernal size 3 x 3, L2 regularization
model.add(Conv2D(64, (3, 3), input_shape=input_shape, kernel_regularizer=regularizers.l2(lamda)))
model.add(Activation('relu'))

# Pooling layer: subsampling 2 x 2, stride 2
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# Convolution layer: 128 filters, kernal size 3 x 3, L2 regularization
model.add(Conv2D(128, (3, 3), input_shape=input_shape, kernel_regularizer=regularizers.l2(lamda)))
model.add(Activation('relu'))

# Convolution layer: 128 filters, kernal size 3 x 3, L2 regularization
model.add(Conv2D(128, (3, 3), input_shape=input_shape, kernel_regularizer=regularizers.l2(lamda)))
model.add(Activation('relu'))

# Pooling layer: subsampling 2 x 2, stride 2
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# Fully connected layer: 1024 Activation Units
model.add(layers.Dense(units=1024, activation='relu'))

# Dropout layer probability 0.5
model.add(layers.Dropout(0.5))

# Fully connected layer: 1024 Activation Units
model.add(layers.Dense(units=1024, activation='relu'))

# Dropout layer probability 0.5
model.add(layers.Dropout(0.5))

# Add fully connected layer with a sigmoid activation function
model.add(layers.Dense(units=1, activation='sigmoid'))


######DEFECT MODEL END######


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

    validation_steps=nb_validation_samples // batch_size,

    verbose=1,

    callbacks=[tensorboard])

# serialize model to JSON
model_json = model.to_json()
with open("defect_cnn.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights('defect_cnn.h5')