from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import models
from keras.models import Sequential
from keras.applications import vgg16
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard
from keras import backend as K
from keras import regularizers
from keras import layers
from keras import optimizers

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

# Create the model
from keras.applications import VGG16
#Load the VGG model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=input_shape )

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)

# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
model.summary()



#####VGG16 MODEL END######

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

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# Train the model
history = model.fit_generator(
      train_generator,
      steps_per_epoch=nb_train_samples // batch_size,
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps=nb_validation_samples // batch_size,
      verbose=1,
      callbacks=[tensorboard])

# serialize model to JSON
model_json = model.to_json()
with open("vgg16.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights('vgg16.h5')
