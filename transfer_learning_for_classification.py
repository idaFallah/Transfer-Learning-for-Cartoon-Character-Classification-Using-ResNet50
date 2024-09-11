
# import libs

import tensorflow as tf
import seaborn as sns
import zipfile
import numpy as np
import cv2
from google.colab.patches import cv2_imshow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model # to connect models together
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout  # GlobalAveragePooling2D similar to flatten layer,
tf.__version__

# loading the images

from google.colab import drive
drive.mount('/content/drive')

path = '/content/drive/MyDrive/Computer Vision/Datasets/homer_bart_2.zip'
zip_object = zipfile.ZipFile(file=path, mode='r')
zip_object.extractall('./')
zip_object.close()

tf.keras.preprocessing.image.load_img('/content/homer_bart_2/training_set/bart/bart100.bmp')

tf.keras.preprocessing.image.load_img('/content/homer_bart_2/training_set/homer/homer100.bmp')

# train/ test set splitting

training_generator = ImageDataGenerator(rescale=1./255,
                                        rotation_range=7,
                                        horizontal_flip=True,
                                        zoom_range=0.2)
train_dataset = training_generator.flow_from_directory('/content/homer_bart_2/training_set',
                                                        target_size = (256, 256),
                                                        batch_size = 8,
                                                        class_mode = 'categorical',
                                                        shuffle = True)

train_dataset.classes

train_dataset.class_indices

test_generator = ImageDataGenerator(rescale=1./255)
test_dataset = test_generator.flow_from_directory('/content/homer_bart_2/test_set',
                                                     target_size = (256, 256),
                                                     batch_size = 1,
                                                     class_mode = 'categorical',
                                                     shuffle = False)

# importing the pretrained network

base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False,
                                             input_tensor=Input(shape=(256, 256, 3)))     # the part that controls if we wanna use the last layer (dense layer) of NN

base_model.summary()

len(base_model.layers)

# freezing the weights in hidden layers

for layer in base_model.layers:
  layer.trainable = False   # freezing the parameters

for layer in base_model.layers:
  print(layer, layer.trainable)   # when layer.trainable is True the parameters for each layer will be adjusted

# ctreating and connecting our custom dense layer

base_model.output

head_model = base_model.output
head_model = GlobalAveragePooling2D()(head_model)   # flattening the basemodel output to a vector then concatenating it to the head model
head_model = Dense(units=1025, activation='relu')(head_model)
head_model = Dropout(rate=0.2)(head_model)  # to avoid overfitting
head_model = Dense(units=1025, activation='relu')(head_model)
head_model = Dropout(rate=0.2)(head_model)
head_model = Dense(2, activation='softmax')(head_model)  # the output layer of NN

# building & training the NN

base_model.input

network = Model(inputs=base_model.input, outputs=head_model)

network.summary()

8 * 8 * 2048, (2048+2)/2

network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = network.fit(train_dataset, epochs=50)  # takes 1 hour to run

# evaluating the NN

test_dataset.class_indices

predictions = network.predict(test_dataset)
predictions

predictions = np.argmax(predictions, axis=1)
predictions

test_dataset.classes

from sklearn.metrics import accuracy_score
accuracy_score(test_dataset.classes, predictions)

# to get best results from transfer learning the images better be similar

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_dataset.classes, predictions)
sns.heatmap(cm, annot=True);

from sklearn.metrics import classification_report
print(classification_report(test_dataset.classes, predictions))

# fine tuning to get better results -> some of the layers in conv layers

# unfreezing the base model
base_model.trainable = True

for layer in base_model.layers:
  print(layer, layer.trainable)

len(base_model.layers)

fine_tuning_at = 140   # layers behind this parameter will be freezed and the rest will train
# it's better to unfreeze only the last layers
# if the number of trainable layers are low, we need to train the model for more epochs

for layer in base_model.layers[:fine_tuning_at]:
  layer.trainable = False

network.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = network.fit(train_dataset, epochs=50)

# evaluating the model

predictions = network.predict(test_dataset)
predictions

predictions = np.argmax(predictions, axis=1)
predictions

test_dataset.classes

from sklearn.metrics import accuracy_score
accuracy_score(test_dataset.classes, predictions)

# saving & loading the model

model_json = network.to_json()
with open('network.json','w') as json_file:
  json_file.write(model_json)

from keras.models import save_model
network_saved = save_model(network, '/content/weights.hdf5')

with open('network.json', 'r') as json_file:
  json_saved_model = json_file.read()
json_saved_model

network_loaded = tf.keras.models.model_from_json(json_saved_model)
network_loaded.load_weights('weights.hdf5')
network_loaded.compile(loss = 'categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

network_loaded.summary()

# classifying one single image

image = cv2.imread('/content/homer_bart_2/test_set/homer/homer15.bmp')

cv2_imshow(image)

image = cv2.resize(image, (256, 256))
cv2_imshow(image)

image

image = image / 255   # normalizing
image

image.shape

image = image.reshape(-1, 256, 256, 3)
image.shape

result = network_loaded(image)
result

result = np.argmax(result)
result

test_dataset.class_indices

if result == 0:
  print('Bart')
else:
  print('Homer')

