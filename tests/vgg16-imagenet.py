import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

import numpy as np
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
import itertools, copy

import time, sys, math

model = tf.keras.applications.VGG16(
    include_top=True, weights=None, input_tensor=None,
    input_shape=None, pooling=None, classes=1000)

model.compile(optimizer='sgd', loss='categorical_crossentropy')

#model.save_weights('h5/vgg16-trained.h5')

numImages = 1

#https://gist.github.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57
imagesets = ['n02442845', 'n15075141', 'n02457408', 'n03642806', 'n03100240', 'n03792782', 'n03131574', 'n13133613', 'n12144580', 'n02992211']
labels = ['mink', 'toilet_tissue', 'three-toed_sloth', 'laptop', 'convertible', 'mountain_bike', 'crib', 'ear', 'corn', 'cello']
#classes = [23, 889, 38, 228, 268, 255, 298, 329, 331, 342]

trdata = ImageDataGenerator()

traindata = trdata.flow_from_directory(directory="/home/nniranjhana/datasets/imagenet18/train", batch_size=32, target_size=(224, 224))

#traindata=itertools.islice(traindata, 40036)

tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="/home/nniranjhana/datasets/imagenet18/validation", target_size=(224, 224))

start=time.time()
model.fit_generator(steps_per_epoch=100, generator=traindata, validation_data=testdata, validation_steps=10, epochs=100, verbose=0)
print("Time to train:", time.time()-start)
images = []
img_labels = []

for i, l in zip(imagesets, labels):
    abspath = '/home/nniranjhana/datasets/imagenet18/validation/'
    abspathi = os.path.join(abspath, i)
    for j in range(numImages):
        rand_file = random.choice(os.listdir(abspathi))
        path = os.path.join(abspathi, rand_file)
        image = load_img(path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        out = model.predict(image)
        label = decode_predictions(out)
        label = label[0][0]
        print("Prediction:", label[1],"Correct label:", l)