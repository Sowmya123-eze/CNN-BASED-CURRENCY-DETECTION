# import math
# import matplotlib.pyplot as plt
# import keras
# from keras import regularizers
# from keras.models import Model
# from keras.layers import Dense, Flatten, BatchNormalization, Dropout
# from keras import optimizers
# from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing import image
# from keras.applications.vgg16 import VGG16
# from keras.callbacks import ModelCheckpoint, EarlyStopping

# batch_size = 32

# train_gen = ImageDataGenerator()
# training_set = train_gen.flow_from_directory(
#     directory="dataset/train/",
#     target_size=(224,224),
#     batch_size=batch_size
# )

# test_gen = ImageDataGenerator()
# test_set = test_gen.flow_from_directory(
#     directory="dataset/test/",
#     target_size=(224,224),
#     batch_size=batch_size
# )


# vggmodel = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# print(vggmodel.summary())

# for layers in (vggmodel.layers):
#     layers.trainable = False

# x = Flatten()(vggmodel.output)

# x = Dense(128, activation='relu')(x)
# x = BatchNormalization()(x)
# x = Dropout(0.7)(x)

# x = Dense(128, activation='relu')(x)
# x = BatchNormalization()(x)
# x = Dropout(0.7)(x)

# predictions = Dense(2, activation='softmax', kernel_initializer='random_uniform',
#                     bias_initializer='random_uniform', bias_regularizer=regularizers.l2(0.01), name='predictions')(x)

# model_final = Model(input = vggmodel.input, output = predictions)

# training_size = 100
# validation_size = 100
# steps_per_epoch = math.ceil(training_size / batch_size)
# validation_steps = math.ceil(validation_size / batch_size)

# # compilation 1
# rms = optimizers.RMSprop(lr=0.0001, decay=1e-4)
# model_final.compile(loss="categorical_crossentropy", optimizer = rms, metrics=["accuracy"])
# print(model_final.summary())
# earlystop1 = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
# hist1 = model_final.fit_generator(
#     training_set,
#     steps_per_epoch=steps_per_epoch,
#     epochs=20,
#     validation_data = test_set,
#     validation_steps=validation_steps,
#     callbacks=[earlystop1],
#     workers=10,
#     shuffle=True
# )

# plt.plot(hist1.history["accuracy"])
# plt.plot(hist1.history['val_accuracy'])
# plt.plot(hist1.history['loss'])
# plt.plot(hist1.history['val_loss'])
# plt.title("model accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
# plt.savefig('vgg16' + '_initialModel_plot.png')
# plt.show()

# # make last 8 layers trainable

# for layer in model_final.layers[:15]:
#     layer.trainable = False

# for layer in model_final.layers[15:]:
#     layer.trainable = True

# # compilation 2
# sgd = optimizers.SGD(lr=0.0001, decay=1e-4, momentum=0.9, nesterov=True)
# model_final.compile(loss = "categorical_crossentropy", optimizer = sgd, metrics=["accuracy"])

# earlystop2 = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
# hist2 = model_final.fit_generator(
#     training_set,
#     epochs= 20,
#     steps_per_epoch=steps_per_epoch,
#     validation_steps=validation_steps,
#     validation_data= test_set,
#     callbacks=[earlystop2],
#     workers=10,
#     shuffle=True
# )

# model_final.save("currency_vgg16model.h5")

# print("vgg16_currency_class_indices", training_set.class_indices)
# f = open("vgg16_currency_class_indices.txt", "w")
# f.write(str(training_set.class_indices))
# f.close()

# plt.plot(hist2.history["accuracy"])
# plt.plot(hist2.history['val_accuracy'])
# plt.plot(hist2.history['loss'])
# plt.plot(hist2.history['val_loss'])
# plt.title("model accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
# plt.savefig('vgg16' + '_finalModel_plot.png')
# plt.show()

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'dataset/train'
valid_path = 'dataset/test'

# add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
    layer.trainable = False

    # useful for getting number of classes
folders = glob('dataset/train/*')

# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

# tell the model what cost and optimization method to use
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset/test',
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='categorical')


labels = (training_set.class_indices)
# print(labels)

labels = dict((v,k) for k,v in labels.items())
# {v: k for k, v in myArray.items()}
print(labels)

for image_batch, label_batch in training_set:

    break
print(image_batch.shape, label_batch.shape)


print (training_set.class_indices)

# Labels = '\n'.join(sorted(training_set.class_indices.keys()))
f = open("labels.txt", "w")
f.write(str(training_set.class_indices))
f.close()
# with open('labels.txt', 'w') as f:
#     f.write(str(Labels))

filepath="fake_currency.h5"
checkpoint1 = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint1]  

# fit the model
r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=20,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set),callbacks=callbacks_list
)

plt.plot(r.history['accuracy'])
plt.plot(r.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("plot_t.png")
plt.show()
