import os
import cv2
import numpy as np
import random
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical
from model import MiniVGG
from imutils import paths
from keras_preprocessing.image import img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as ts

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

img_height=128
img_width=128
EPOCHS = 10
num_classes=2
INIT_LR = 1e-3
BS = 32

data = []
labels = []
imagePaths = sorted(list(paths.list_images("./train")))
random.seed(42)
random.shuffle(imagePaths)

print(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (img_height, img_width))
    image = img_to_array(image)
    data.append(image)
    label = imagePath.split(os.path.sep)[-2]
    label = 1 if label == "fake" else 0
    labels.append(label)

print(len(data))

data = np.array(data, dtype="float") / 255.0
np.save('data.npy',data)
labels = np.array(labels)
np.save('labels.npy',labels)
data=np.load('data.npy')
labels=np.load('labels.npy')
(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.25, random_state=42)
channels=trainX.shape[3]
trainY = to_categorical(trainY, num_classes)
testY = to_categorical(testY, num_classes)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")

print("Compiling model...")
model = MiniVGG(width=img_width, height=img_height, depth=channels, classes=num_classes)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS) #Optimise uisng Adam
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

print("Training network")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY), steps_per_epoch=len(trainX),
    epochs=EPOCHS, verbose=1)
label_name=["real","fake"]
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1)))

cm = confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1))
total = sum(sum(cm))
print(cm)
print("End")
