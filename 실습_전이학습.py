#!/usr/bin/env python
# coding: utf-8

# This Code is modified based on the code from
# 
#  https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial11-transfer-learning.py

# In[2]:


# 구글 드라이브 연동 #
from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


get_ipython().system("unzip '/content/drive/My Drive/cumvision/archive2.zip' -d '/content/drive/My Drive/cumvision'")


# In[6]:


import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow_hub as hub
import numpy as np
import pathlib
import keras.preprocessing.image
from keras.preprocessing.image import ImageDataGenerator


# In[20]:


# To Avoid GPU errors
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[21]:


# ================================================ #
#                  Pretrained-Model                #
# ================================================ #

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
# x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
img_size=299
model = keras.models.load_model('/content/drive/My Drive/cumvision/model.h5')
model


# In[22]:


# Freeze all model layer weights
model.trainable = False

# Can also set trainable for specific layers
for layer in model.layers:
    # assert should be true because of one-liner above
    assert layer.trainable == False
    layer.trainable = False

print(model.summary())  # for finding base input and output


base_inputs = model.layers[0].input
base_output = model.layers[-2].output
output = layers.Dense(10)(base_output)

new1_model = tf.keras.Model(base_inputs, output)

# This model is actually identical to model we
# loaded (this is just for demonstration and
# and not something you would do in practice).
print(new1_model.summary())

# As usual we do compile and fit, this time on new_model
new1_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

new1_model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=2)


# In[ ]:


import tensorflow_datasets as tfds
tfds.disable_progress_bar()

train_ds, validation_ds, test_ds = tfds.load(
    "cats_vs_dogs",
    # Reserve 10% for validation and 10% for test
    split=["train[:40%]", "train[40%:50%]", "train[50%:60%]"],
    as_supervised=True,  # Include labels
)

print("Number of training samples: %d" % tf.data.experimental.cardinality(train_ds))
print(
    "Number of validation samples: %d" % tf.data.experimental.cardinality(validation_ds)
)
print("Number of test samples: %d" % tf.data.experimental.cardinality(test_ds))


# In[13]:


batch_size = 32
img_height = 150
img_width = 150


# In[14]:


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  '/content/drive/My Drive/cumvision/chest_xray/train',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[15]:


test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  '/content/drive/My Drive/cumvision/chest_xray/test',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(int(label))
    plt.axis("off")


# In[ ]:


size = (299, 299)

train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))


# In[ ]:


train_ds.shape


# In[ ]:


batch_size = 3

train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)


# In[17]:


# =================================================== #
#                Pretrained Keras Model(1)            #
#             classifier부분만 학습                   #
# =================================================== #
from keras.applications.inception_v3 import InceptionV3
inception = InceptionV3(weights='imagenet',input_shape=(299,299,3) , include_top=True)

model = tf.keras.applications.InceptionV3(include_top=True)
print(model.summary())

# for input you can also do model.input,
# then for base_outputs you can obviously
# choose other than simply removing the last one :)
base_inputs = model.layers[0].input
base_outputs = model.layers[-2].output
classifier = tf.keras.layers.Dense(2)(base_outputs)
new_model = keras.Model(inputs=base_inputs, outputs=classifier)

new_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

print(new_model.summary())
new_model.fit(train_ds, epochs=15)


# In[16]:


# =================================================== #
#  Pretrained Keras Model (2) --전체 미세조정         #
# =================================================== #

from keras.applications.inception_v3 import InceptionV3
inception = InceptionV3(weights='imagenet',input_shape=(299,299,3) , include_top=True)
for layer in inception.layers[:]:
  layer.trainable = True
# for input you can also do model.input,
# then for base_outputs you can obviously
# choose other than simply removing the last one :)

base_inputs = inception.layers[0].input
base_outputs = inception.layers[-2].output
classifier = tf.keras.layers.Dense(2)(base_outputs)
new_model = keras.Model(inputs=base_inputs, outputs=classifier)


new_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

print(new_model.summary())
new_model.fit(train_ds, epochs=15)


# In[ ]:


print()


# In[10]:


# ================================================= #
#        Pretrained Hub Model(1)                    #
#       Classifier 부분만 학습                      #
# ================================================= #



url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4"
base_model = hub.KerasLayer(url, input_shape=(299, 299, 3))
model = tf.keras.Sequential(
    [
        base_model,
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(2),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

#model.fit(train_ds, epochs=15)
model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=2)


# # tensorflow HUB로 미세조정 전이학습
# 
# 다음 페이지에서 tensorflow HUB로 미세조정(fine tuning)에 대한  예제를 볼 수 있음 
# https://www.tensorflow.org/hub/common_saved_model_apis/images?hl=ko
# 
# 이 colab 코드에서 실행가능함
# https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/tf2_image_retraining.ipynb?hl=ko

# ## 1일차 실습: 
# 
# 실습1에 나왔던 폐렴 데이터셋에 대하여 전이학습으로 분류기를 학습시켜보기

# In[ ]:




