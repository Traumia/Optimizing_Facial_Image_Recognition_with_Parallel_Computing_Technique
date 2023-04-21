#!/usr/bin/env python
# coding: utf-8

# ## Paralleled Celebface Image Recognition Based On CNN Deep Learning
# ### Group 09 Yueheng LI

# **Dataset**
# 
# For this project I use the CelebA dataset (http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), which is available on Kaggle.
# 
# Description of the CelebA dataset from kaggle (https://www.kaggle.com/jessicali9530/celeba-dataset):

# **Introduction of Dataset**
# 
# A popular component of computer vision and deep learning revolves around identifying faces for various applications from logging into your phone with your face or searching through surveillance images for a particular suspect. This dataset is great for training and testing models for face detection, particularly for recognising facial attributes such as finding people with brown hair, are smiling, or wearing glasses. Images cover large pose variations, background clutter, diverse people, supported by a large quantity of images and rich annotations. This data was originally collected by researchers at MMLAB, The Chinese University of Hong Kong (specific reference in Acknowledgment section).

# **Data Files Overall**
# 
# **img_align_celeba.zip:** All the face images, cropped and aligned
# 
# 
# **list_eval_partition.csv:** Recommended partitioning of images into training, validation, testing sets. Images 1-162770 are training, 162771-182637 are validation, 182638-202599 are testing
# 
# 
# **list_bbox_celeba.csv:** Bounding box information for each image. "x_1" and "y_1" represent the upper left point coordinate of bounding box. "width" and "height" represent the width and height of bounding box
# 
# 
# **list_landmarks_align_celeba.csv:** Image landmarks and their respective coordinates. There are 5 landmarks: left eye, right eye, nose, left mouth, right mouth
#     
#     
# **list_attr_celeba.csv:** Attribute labels for each image. There are 40 attributes. "1" represents positive while "-1" represents negative
# 

# ## 1. Import modules

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
print(tf.__version__)


# ### 1.1. Parallel setup / device check

# In[ ]:


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[2]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[3]:


#get_ipython().system('nvidia-smi')


# In[3]:


#import tensorflow.compat.v1 as tf


# In[4]:


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)


# # ------------------------------------------

# ## 2. Load image label attributes

# In[5]:


MAIN_PATH = '../FinalProject/'
DATA_PATH = MAIN_PATH + 'img_align_celeba/img_align_celeba/'
ATTRIBUTE_PATH = MAIN_PATH + 'list_attr_celeba.csv'
#TEST_PIC_PATH = './test_pic/'

SMILING_PATH = '../FinalProject/New_data1/'


# In[7]:


time1=time.time()
dataset = pd.read_csv(ATTRIBUTE_PATH, index_col='image_id')
time2=time.time()
print("The time of the process:",time2-time1,"s")


# ### 2.1 With Dask Parallelism

# In[12]:


import dask
# from dask.distributed import Client

# client = Client(n_workers=4)
# client


# In[14]:


import dask.dataframe as dd
import time

time1=time.time()
dataset1 = dd.read_csv(ATTRIBUTE_PATH)
time2=time.time()
print("The time of the process with dask:",time2-time1,"s")
#dataset1.head()


# In[7]:


dataset


# ### 2.2 Exploratory Data Analysis

# In[41]:


#List of available attributes
# for i, j in enumerate(dataset.columns):
#     print(i, j)


# In[11]:


# plot picture and attributes
# img = load_img('C:/Users/Razer/Desktop/Data/img_align_celeba/img_align_celeba/000510.jpg')
# plt.grid(False)
# plt.imshow(img)
# dataset.loc['C:/Users/Razer/Desktop/Data/img_align_celeba/img_align_celeba/000510.jpg'.split('/')[-1]][['Smiling','Male','Young']] #some attributes


# In[42]:


# Female or Male?
# plt.title('Female or Male')
# sns.countplot(y='Male', data=dataset, color="c")
# plt.show()


# In[ ]:


# dataset1.replace(-1,0,inplace=True)
# arr=pd.DataFrame(dataset1.iloc[:,1:].sum(axis=0))
# arr.columns=['labels']
# arr.sort_values(by='labels',ascending=False)
# plt.figure(figsize=(16,8))
# plt.bar(arr.index,arr['labels'])
# plt.xticks(rotation=90)
# plt.show()


# In[ ]:


# import seaborn as sns
# plt.figure(figsize=(16,12))
# sns.heatmap(dataset1.iloc[:,1:].corr(), cmap="RdYlBu", vmin=-1, vmax=1)


# ## 3. Data pre-processing
# *Modified on local machine and we use pre-processed dataset after modification*

# In[22]:


# #split the training and testing set
# from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split(dataset, test_size = 0.2, random_state = 0)


# In[17]:


# train_set_not_smiling = train_set.query('Smiling == -1')
# train_set_smiling = train_set.query('Smiling == +1')

# test_set_not_smiling = test_set.query('Smiling == -1')
# test_set_smiling = test_set.query('Smiling == +1')


# In[21]:


#Copy the train and test data to the appropriate folder
# import shutil
# import time
# time1=time.time()


# for img in train_set_not_smiling.index.values:
#     shutil.copy(DATA_PATH + img, SMILING_PATH + "train_set/not_smiling")
# for img in train_set_smiling.index.values:
#     shutil.copy(DATA_PATH + img, SMILING_PATH + "train_set/smiling")
# for img in test_set_not_smiling.index.values:
#     shutil.copy(DATA_PATH + img, SMILING_PATH + "test_set/not_smiling")
# for img in test_set_smiling.index.values:
#     shutil.copy(DATA_PATH + img, SMILING_PATH + "test_set/smiling")
    
# time2=time.time()
# print("The time of the process:",time2-time1,"s")


# ## 3. Data Loader（Using keras DataGenerator）

# In[7]:

TARGET_SIZE = (64, 64)
BATCH_SIZE = 64
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


# In[8]:


time1=time.time()

train_generator = train_datagen.flow_from_directory(
    SMILING_PATH + 'train_set',
    target_size = TARGET_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'binary')

time2=time.time()
print("The time of the process:",time2-time1,"s")



# In[9]:


time1=time.time()

validation_generator = test_datagen.flow_from_directory(
    SMILING_PATH + 'test_set',
    target_size = TARGET_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'binary')

time2=time.time()
print("The time of the process:",time2-time1,"s")


# ## 4. Building CNN model

# ### 4.1 Model Definition ：Without Parallel Strategy

# In[16]:


# TARGET_SIZE = (64, 64)
# BATCH_SIZE = 128


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


classifier1 = Sequential()
classifier1.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=(64,64,3)))
classifier1.add(MaxPooling2D(pool_size=(2,2)))
classifier1.add(Conv2D(32,kernel_size=(3,3), activation='relu'))
classifier1.add(MaxPooling2D(pool_size=(2,2)))
classifier1.add(Flatten())
classifier1.add(Dense(units= 128, activation='relu'))
classifier1.add(Dense(units= 1, activation='sigmoid'))
classifier1.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# ### 4.2 Model Definition ：With MirroredStrategy

# In[ ]:


# TARGET_SIZE = (64, 64)
# BATCH_SIZE = 128
n_gpus=1

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

device_type = 'GPU'
devices = tf.config.experimental.list_physical_devices(device_type)
devices_names = [d.name.split("e:")[1] for d in devices]
print(devices_names)
strategy = tf.distribute.MirroredStrategy(devices= devices_names[:n_gpus])

with strategy.scope():
    classifier2 = Sequential()
    classifier2.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=(64,64,3)))
    classifier2.add(MaxPooling2D(pool_size=(2,2)))
    classifier2.add(Conv2D(32,kernel_size=(3,3), activation='relu'))
    classifier2.add(MaxPooling2D(pool_size=(2,2)))
    classifier2.add(Flatten())
    classifier2.add(Dense(units= 128, activation='relu'))
    classifier2.add(Dense(units= 1, activation='sigmoid'))
    classifier2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# ### 4.3 Model Definition ：With OneDeviceStrategy

# In[ ]:


# TARGET_SIZE = (64, 64)
# BATCH_SIZE = 128
# n_gpus=1

# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# device_type = 'GPU'
# devices = tf.config.experimental.list_physical_devices(device_type)
# devices_names = [d.name.split("e:")[1] for d in devices]
# print(devices_names)
# #strategy = tf.distribute.MirroredStrategy(devices= devices_names[:n_gpus])
# strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")


# with strategy.scope():
#     classifier3 = Sequential()
#     classifier3.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=(64,64,3)))
#     classifier3.add(MaxPooling2D(pool_size=(2,2)))
#     classifier3.add(Conv2D(32,kernel_size=(3,3), activation='relu'))
#     classifier3.add(MaxPooling2D(pool_size=(2,2)))
#     classifier3.add(Flatten())
#     classifier3.add(Dense(units= 128, activation='relu'))
#     classifier3.add(Dense(units= 1, activation='sigmoid'))
#     classifier3.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])



# In[48]:


#classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print(classifier1.summary())


# In[4]:


# # Allow memory growth for the GPU
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


# In[16]:


# import os
# os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"


# ## 5. Model Training

# ### 5.1 CPU computation
# *Modified on local machine (Training on Tensorflow CPU version)*

# In[32]:


#"This is CPU computation"
#print("This is CPU computation")
# time1=time.time()
# history = classifier3.fit(
#     train_generator,
#     steps_per_epoch = train_generator.samples // train_generator.batch_size,
#     epochs = 12,
#     validation_data = validation_generator,
#     validation_steps = validation_generator.samples // validation_generator.batch_size
# )
# time2=time.time()
# print("The time of the process:",time2-time1,"s")


# ### 5.2 GPU computation without parallel strategy

# In[ ]:



# time1_1=time.time()
# history = classifier1.fit(
#     train_generator,
#     steps_per_epoch = train_generator.samples // train_generator.batch_size,
#     epochs = 1,
#     validation_data = validation_generator,
#     validation_steps = validation_generator.samples // validation_generator.batch_size
# )
# time2_1=time.time()
# t1=time2_1-time1_1
# print("This is GPU computation-without parallal strategy:")
# print("The time of the process:",t1,"s")


# ### 5.3 GPU computation with parallel strategy(MirroredStrategy)

# In[2]:



time1_2=time.time()
history = classifier2.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // train_generator.batch_size/2,
    epochs = 1,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // validation_generator.batch_size
)
time2_2=time.time()
t2=time2_2-time1_2
print("This is GPU computation-with Mirroredparallal strategy")
print("The time of the process:",t2,"s")


# ### 5.4 GPU computation with parallel strategy(OneDeviceStrategy)
# In[ ]:



# time1_3=time.time()
# history = classifier2.fit(
#     train_generator,
#     steps_per_epoch = train_generator.samples // train_generator.batch_size,
#     epochs = 1,
#     validation_data = validation_generator,
#     validation_steps = validation_generator.samples // validation_generator.batch_size
# )
# time2_3=time.time()
# t3=time2_3-time1_3
# print("This is GPU computation-with OneDevice parallal strategy")
# print("The time of the process:",t3,"s")



TARGET_SIZE = (64, 64)
BATCH_SIZE = 64
time1=time.time()

train_generator1 = train_datagen.flow_from_directory(
    SMILING_PATH + 'train_set',
    target_size = TARGET_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'binary')

time2=time.time()
print("The time of the process:",time2-time1,"s")

time1=time.time()

validation_generator = test_datagen.flow_from_directory(
    SMILING_PATH + 'test_set',
    target_size = TARGET_SIZE,
    batch_size = BATCH_SIZE,
    class_mode = 'binary')

time2=time.time()
print("The time of the process:",time2-time1,"s")

n_gpus=2

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

device_type = 'GPU'
devices = tf.config.experimental.list_physical_devices(device_type)
devices_names = [d.name.split("e:")[1] for d in devices]
print(devices_names)
strategy = tf.distribute.MirroredStrategy(devices= devices_names[:2])

with strategy.scope():
    classifier4 = Sequential()
    classifier4.add(Conv2D(32,kernel_size=(3,3), activation='relu', input_shape=(64,64,3)))
    classifier4.add(MaxPooling2D(pool_size=(2,2)))
    classifier4.add(Conv2D(32,kernel_size=(3,3), activation='relu'))
    classifier4.add(MaxPooling2D(pool_size=(2,2)))
    classifier4.add(Flatten())
    classifier4.add(Dense(units= 128, activation='relu'))
    classifier4.add(Dense(units= 1, activation='sigmoid'))
    classifier4.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# # Disable AutoShard.
# options = tf.data.Options()
# options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
# train_generator = train_generator.with_options(options)
# validation_data = validation_data.with_options(options)

time1_4=time.time()
history = classifier4.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // train_generator.batch_size/2,
    epochs = 1,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // validation_generator.batch_size
)
time2_4=time.time()
t4=time2_4-time1_4
print("This is Multi-GPU computation(2 GPU Mirror Strategic)")
print("The time of the process:",t4,"s")

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ["1 GPU", "2 GPU"]
students = [t2,t4]
ax.bar(langs,students)
#plt.ylim((1800, 2200))
#ax.set_ylabel('Elapsed time/s')
ax.set_title('The trend curve of speedup of using different number of GPU')
plt.xlabel('Number of GPU')
plt.ylabel('Elapsed time/s')
plt.grid(color = 'grey', linestyle = 'dashed')
plt.savefig("CNN1.png")
plt.show()
# ## 6. Save the CNN model
# *Modified on local machine*

# In[63]:


#classifier.save('C:/Users/Razer/Desktop/Data/smiling_model.h5')


# In[57]:


# from keras.models import load_model
# classifier= load_model('./exported_model/smiling_model.h5')


# ## 7. Result analysis

# In[ ]:


# loss_values = history.history['loss']
# epochs = range(1, len(loss_values) + 1)

# plt.plot(epochs, loss_values, label='Training Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


# In[53]:


# plt.figure(figsize=(8, 4))
# plt.plot(history.history['accuracy'], label = 'train')
# plt.plot(history.history['val_accuracy'], label = 'valid')
# plt.legend()
# plt.title('Accuracy')
# plt.show()


# In[ ]:


# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# print(classification_report(y_val, predictions))


# In[34]:


# results = classifier.evaluate(validation_generator, verbose=0)
# results


# # In[35]:


# predictions = classifier.predict(validation_generator)
# predictions = (predictions > 0.5)


# # In[58]:


# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# cm = confusion_matrix(validation_generator.classes, predictions)
# plt.figure(figsize=(6,6))
# sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap="Blues_r")
# plt.xlabel("Predicted label")
# plt.ylabel("Actual label")
# plt.show()


# # -------------------------------------------------------------

# ## 8. Predict a single image

# In[39]:


# from keras_preprocessing import image

# def classify(imagePath):
#     test_image = image.load_img(imagePath, target_size = (64, 64))
#     test_image = image.img_to_array(test_image).astype('float32') / 255 
#     test_image = np.expand_dims(test_image, axis = 0) # add an extra dimension to fake batch

#     result = classifier.predict(test_image)
#     if result[0][0] >= 0.5:
#         prediction = 'smiling'
#         probability = result[0][0]
#     else:
#         prediction = 'not smiling'
#         probability = 1 - result[0][0]
#     print("This person is " + prediction + " (" + str(probability * 100) + "%).")


# In[40]:


# from keras.preprocessing.image import load_img

# def showImage(imagePath):
#     img = load_img(imagePath)
#     plt.imshow(img)
#     plt.show()


# In[47]:


# imageToClassify = DATA_PATH + "000012.jpg" # Can test from 0 to 3
# classify(imageToClassify)
# showImage(imageToClassify)


# ### With mutiprocessing method

# In[ ]:





# In[ ]:





# ## ----------------------- END -----------------------------
