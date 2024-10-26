#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install torch torchvision torchaudio diffusers transformers tensorflow_datasets')


# In[ ]:


#### NOTE: This would take significant amount of time when running for the first time
import tensorflow_datasets as tfds
dataset, info = tfds.load('kitti', with_info=True)


# In[1]:


# print(info)


# In[21]:


### Documentation on the dataset
# https://datasetninja.com/kitti-object-detection#object-distribution


# In[3]:


train_dataset = dataset['train']
test_dataset = dataset['test']
validation_dataset = dataset['validation']


# In[6]:


import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to display images and annotations
def show_images_with_annotations(dataset, num_images):
    plt.figure(figsize=(45, 45))
    for i, example in enumerate(dataset.take(num_images)):
        image = example['image'].numpy()
        bboxes = example['objects']['bbox'].numpy()
        plt.subplot(5, 5, i + 1)
        plt.imshow(image)
        for bbox in bboxes:
            ymin, xmin, ymax, xmax = bbox
            # Convert normalized coordinates to pixel values
            height, width, _ = image.shape
            xmin = int(xmin * width)
            xmax = int(xmax * width)
            ymin = int(ymin * height)
            ymax = int(ymax * height)
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='red', facecolor='none')
            plt.gca().add_patch(rect)
        plt.axis("off")
    plt.show()

# Display 5 images with bounding boxes
show_images_with_annotations(train_dataset, 5)


# In[ ]:





# In[9]:


import tensorflow_datasets as tfds
import tensorflow as tf

# Get the number of records
num_train = tf.data.experimental.cardinality(train_dataset).numpy()
num_test = tf.data.experimental.cardinality(test_dataset).numpy()
num_validation = tf.data.experimental.cardinality(validation_dataset).numpy()

print(f'Number of training records: {num_train}')
print(f'Number of validation records: {num_validation}')
print(f'Number of testing records: {num_test}')


# In[11]:


def preprocess(data):
    image = data['image']
    image = tf.image.resize_with_pad(image, 128, 128)  # Resize with padding to 128x128
    image = tf.cast(image, tf.float32) / 255.0
    
    # Create a binary vector for labels
    labels = tf.reduce_sum(tf.one_hot(data['objects']['type'], depth=8), axis=0)
    return image, labels


# In[12]:


batch_size = 128

train_dataset = train_dataset.map(preprocess).batch(batch_size)
validation_dataset = validation_dataset.map(preprocess).batch(batch_size)
test_dataset = test_dataset.map(preprocess).batch(batch_size)


# In[17]:


from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(8, activation='sigmoid')  # 8 classes for 'type'
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
              )


# In[18]:


from tensorflow.keras.callbacks import EarlyStopping

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='min',
    verbose=1
)


# In[19]:


model.fit(train_dataset, 
          validation_data=validation_dataset, 
          epochs=5,
         callbacks=[early_stopping_callback])


# In[20]:


test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test accuracy: {test_acc}')


# In[ ]:




