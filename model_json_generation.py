
import tensorflow as tf

import numpy as np

model = tf.keras.applications.ResNet50()
jsonString = model.to_json()

file = open("resnet50.json",'w')
file.write(jsonString)

model = tf.keras.applications.VGG19()
jsonString = model.to_json()

file = open("vgg19.json",'w')
file.write(jsonString)

model = tf.keras.applications.MobileNet()
jsonString = model.to_json()

file = open("mobilenet.json",'w')
file.write(jsonString)

model = tf.keras.applications.InceptionV3()
jsonString = model.to_json()

file = open("inceptionv3.json",'w')
file.write(jsonString)