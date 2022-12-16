#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install tensorflow


# In[2]:


pip install keras


# In[3]:


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


# In[4]:


tf.__version__


# In[10]:


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('C:\\Users\\Hp\\Downloads\\Projects (Download me But will only work if you have python 3.7_3.8) (1)\\Projects\\datasets\\dogs_cats\\training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


# In[11]:


test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('C:\\Users\\Hp\\Downloads\\Projects (Download me But will only work if you have python 3.7_3.8) (1)\\Projects\\datasets\\dogs_cats\\test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')


# In[12]:


cnn = tf.keras.models.Sequential()


# In[13]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))


# In[14]:


cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[15]:


cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))


# In[16]:


cnn.add(tf.keras.layers.Flatten())


# In[17]:


cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))


# In[18]:


cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# In[19]:


cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[20]:


cnn.fit(x = training_set, validation_data = test_set, epochs = 25)


# In[46]:


import numpy as np
from keras.preprocessing import image
test_image = tf.keras.utils.load_img('C:\\Users\\Hp\\Downloads\\Projects (Download me But will only work if you have python 3.7_3.8) (1)\\Projects\\datasets\\dogs_cats\\single_prediction\\cat_dog6.jpg', target_size = (64, 64))
test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices
if result[0][0] == 1:

  prediction = 'dog'
else:
  prediction = 'cat'


# In[47]:


print(prediction)


# In[48]:


print(result)


# In[ ]:




