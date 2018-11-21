from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential
from keras.applications import vgg16
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard
from keras import backend as K
from keras import regularizers
from keras import layers

from time import time


# dimensions of our images.
img_width, img_height = 210, 461
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 5480
nb_validation_samples = 520

epochs = 50
lamda = 5E-5
batch_size = 16

# Set up TensorBoard
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)

else:
    input_shape = (img_width, img_height, 3)

model = Sequential()

#####VGG16 MODEL START#####

model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
model.summary(line_length=150)

flatten = Flatten()

# Add fully connected layer with a sigmoid activation function
new_layer2 = Dense(units=1, activation='sigmoid', name='my_dense_2')

inp2 = model.input
out2 = new_layer2(flatten(model.output))

model2 = Model(inp2, out2)
model2.summary(line_length=150)

#####VGG16 MODEL END######

model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])


model2.compile(loss='binary_crossentropy',

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