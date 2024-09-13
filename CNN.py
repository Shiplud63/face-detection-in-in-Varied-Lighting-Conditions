import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


import re
import gc
import glob
import keras
import pandas as pd
import numpy  as np

import tensorflow as tf
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc, multilabel_confusion_matrix


import keras.backend as K
from keras.models     import Sequential
from keras.layers     import Dense, Dropout, GlobalMaxPooling2D
from keras.optimizers import Adam, SGD
from keras.applications import MobileNetV2
from keras.callbacks    import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.model_selection   import train_test_split
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(451)
Using TensorFlow backend.
# General parameters
batch_size = 16
image_size = 224
epochs     = 50
# Read and prepare data
raw_data = pd.read_csv('../input/japanese-female-facial-expression-dataset-jaffe/data.csv')
raw_data['filepath'] = '../input/japanese-female-facial-expression-dataset-jaffe/' + raw_data['filepath']
raw_data.fillna('UNKNOWN', inplace=True)
raw_data.sample(3)
filepath	student	facial_expression
23	../input/japanese-female-facial-expression-dat...	KL	angry
96	../input/japanese-female-facial-expression-dat...	MK	happiness
38	../input/japanese-female-facial-expression-dat...	KL	neutral
label_count_df = pd.DataFrame(raw_data['facial_expression'].value_counts()).reset_index()
fig = px.bar(label_count_df,
             y='index',
             x='facial_expression',
             orientation='h',
             color='index',
             title='Label Distribution',
             opacity=0.8,
             color_discrete_sequence=px.colors.diverging.curl,
             template='plotly_dark'
            )
fig.update_xaxes(range=[0,35])
fig.show()
def plot_samples(df, label_list):
    for label in label_list:
        query_string = "facial_expression == '{}'".format(label)
        df_label = df.query(query_string).reset_index(drop=True)
        
        fig = plt.figure(figsize=(18,15))
        plt.subplot(1,4,1)
        plt.imshow(plt.imread(df_label.loc[0,'filepath']),cmap='gray')
        plt.title(label.capitalize())
        
        plt.subplot(1,4,2)
        plt.imshow(plt.imread(df_label.loc[1,'filepath']),cmap='gray')
        plt.title(label.capitalize())
        
        plt.subplot(1,4,3)
        plt.imshow(plt.imread(df_label.loc[2,'filepath']),cmap='gray')
        plt.title(label.capitalize())
        
        plt.subplot(1,4,4)
        plt.imshow(plt.imread(df_label.loc[3,'filepath']),cmap='gray')
        plt.title(label.capitalize())
        
        plt.show()
plot_samples(raw_data, ['happiness', 'surprise', 'neutral', 'disgust', 'angry', 'fear'])

# Create train and testing sets
train, test = train_test_split(raw_data,
                               test_size = 0.3,
                               stratify=raw_data['facial_expression'],
                               random_state=451
                              )
train_generator = ImageDataGenerator(
                    rescale     = 1./255,
                    shear_range = 0.1,
                    zoom_range  = 0.1,
                    width_shift_range  = 0.1,
                    height_shift_range = 0.1,
                    horizontal_flip    = True)

test_generator = ImageDataGenerator(rescale=1./255)
train_gen = train_generator.flow_from_dataframe(dataframe = train,
                                    class_mode  = 'categorical',
                                    x_col       = 'filepath',
                                    y_col       = 'facial_expression',
                                    shuffle     = True,
                                    batch_size  = batch_size,
                                    target_size = (image_size, image_size),
                                    seed=451)

test_gen  = test_generator.flow_from_dataframe(dataframe = test,
                                    class_mode='categorical',
                                    x_col='filepath',
                                    y_col='facial_expression',
                                    shuffle     = False,
                                    batch_size  = batch_size,
                                    target_size = (image_size, image_size),
                                    seed=451)
Found 149 validated image filenames belonging to 6 classes.
Found 64 validated image filenames belonging to 6 classes.
Model Architecture
# Create and compile model
model = Sequential()
model.add(MobileNetV2(input_shape=(image_size, image_size, 3), weights='imagenet', include_top=False))
model.add(GlobalMaxPooling2D())
model.add(Dropout(0.5))
model.add(Dense(7,activation='softmax'))
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# First Convolutional Layer
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
    
# Second Convolutional Layer
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
    
# Third Convolutional Layer
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
    
# Flattening the layer
model.add(Flatten())
    
# Fully connected layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
    
# Output layer for classification
model.add(Dense(4, activation='softmax'))  # Assuming 4 classes: crying, sad, angry, drowsy
 
# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
 
return model


Downloading data from https://github.com/JonathanCMitchell/mobilenet_v2_keras/releases/download/v1.1/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5
9412608/9406464 [==============================] - 2s 0us/step
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
mobilenetv2_1.00_224 (Model) (None, 7, 7, 1280)        2257984   
_________________________________________________________________
global_max_pooling2d_1 (Glob (None, 1280)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 1280)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 7)                 8967      
=================================================================
Total params: 2,266,951
Trainable params: 2,232,839
Non-trainable params: 34,112
_________________________________________________________________
Model Training
training_hist = model.fit(train_gen,
                          epochs = epochs,
                            )
Epoch 1/50
10/10 [==============================] - 34s 3s/step - loss: 1.8129 - accuracy: 0.7450
Epoch 2/50
10/10 [==============================] - 18s 2s/step - loss: 1.6585 - accuracy: 0.7641
Epoch 3/50
10/10 [==============================] - 17s 2s/step - loss: 1.4854 - accuracy: 0.7833
Epoch 4/50
10/10 [==============================] - 18s 2s/step - loss: 1.5347 - accuracy: 0.7718
Epoch 5/50
10/10 [==============================] - 17s 2s/step - loss: 1.3806 - accuracy: 0.7843
Epoch 6/50
10/10 [==============================] - 18s 2s/step - loss: 1.5467 - accuracy: 0.7747
Epoch 7/50
10/10 [==============================] - 18s 2s/step - loss: 1.4716 - accuracy: 0.7862
Epoch 8/50
10/10 [==============================] - 17s 2s/step - loss: 1.3613 - accuracy: 0.8035
Epoch 9/50
10/10 [==============================] - 18s 2s/step - loss: 1.1803 - accuracy: 0.8102
Epoch 10/50
10/10 [==============================] - 18s 2s/step - loss: 1.3296 - accuracy: 0.7939
Epoch 11/50
10/10 [==============================] - 18s 2s/step - loss: 1.1913 - accuracy: 0.8025
Epoch 12/50
10/10 [==============================] - 18s 2s/step - loss: 1.0942 - accuracy: 0.8207
Epoch 13/50
10/10 [==============================] - 17s 2s/step - loss: 1.0192 - accuracy: 0.8217
Epoch 14/50
10/10 [==============================] - 18s 2s/step - loss: 1.0313 - accuracy: 0.8226
Epoch 15/50
10/10 [==============================] - 18s 2s/step - loss: 0.9318 - accuracy: 0.8380
Epoch 16/50
10/10 [==============================] - 17s 2s/step - loss: 0.9598 - accuracy: 0.8255
Epoch 17/50
10/10 [==============================] - 19s 2s/step - loss: 0.9038 - accuracy: 0.8447
Epoch 18/50
10/10 [==============================] - 17s 2s/step - loss: 0.9393 - accuracy: 0.8274
Epoch 19/50
10/10 [==============================] - 17s 2s/step - loss: 0.6805 - accuracy: 0.8811
Epoch 20/50
10/10 [==============================] - 17s 2s/step - loss: 0.7236 - accuracy: 0.8600
Epoch 21/50
10/10 [==============================] - 19s 2s/step - loss: 0.8824 - accuracy: 0.8437
Epoch 22/50
10/10 [==============================] - 18s 2s/step - loss: 0.6652 - accuracy: 0.8686
Epoch 23/50
10/10 [==============================] - 17s 2s/step - loss: 0.6705 - accuracy: 0.8802
Epoch 24/50
10/10 [==============================] - 19s 2s/step - loss: 0.6452 - accuracy: 0.8734
Epoch 25/50
10/10 [==============================] - 18s 2s/step - loss: 0.6484 - accuracy: 0.8715
Epoch 26/50
10/10 [==============================] - 17s 2s/step - loss: 0.6001 - accuracy: 0.8792
Epoch 27/50
10/10 [==============================] - 18s 2s/step - loss: 0.4774 - accuracy: 0.8859
Epoch 28/50
10/10 [==============================] - 18s 2s/step - loss: 0.5023 - accuracy: 0.8945
Epoch 29/50
10/10 [==============================] - 18s 2s/step - loss: 0.4913 - accuracy: 0.9012
Epoch 30/50
10/10 [==============================] - 17s 2s/step - loss: 0.4177 - accuracy: 0.8945
Epoch 31/50
10/10 [==============================] - 19s 2s/step - loss: 0.3956 - accuracy: 0.9166
Epoch 32/50
10/10 [==============================] - 28s 3s/step - loss: 0.2346 - accuracy: 0.9329
Epoch 33/50
10/10 [==============================] - 18s 2s/step - loss: 0.3647 - accuracy: 0.9128
Epoch 34/50
10/10 [==============================] - 18s 2s/step - loss: 0.5022 - accuracy: 0.8936
Epoch 35/50
10/10 [==============================] - 17s 2s/step - loss: 0.4129 - accuracy: 0.9137
Epoch 36/50
10/10 [==============================] - 18s 2s/step - loss: 0.3757 - accuracy: 0.9175
Epoch 37/50
10/10 [==============================] - 18s 2s/step - loss: 0.3861 - accuracy: 0.9185
Epoch 38/50
10/10 [==============================] - 17s 2s/step - loss: 0.3607 - accuracy: 0.9233
Epoch 39/50
10/10 [==============================] - 18s 2s/step - loss: 0.3134 - accuracy: 0.9406
Epoch 40/50
10/10 [==============================] - 17s 2s/step - loss: 0.2461 - accuracy: 0.9358
Epoch 41/50
10/10 [==============================] - 18s 2s/step - loss: 0.2809 - accuracy: 0.9243
Epoch 42/50
10/10 [==============================] - 17s 2s/step - loss: 0.3098 - accuracy: 0.9310
Epoch 43/50
10/10 [==============================] - 18s 2s/step - loss: 0.2618 - accuracy: 0.9386
Epoch 44/50
10/10 [==============================] - 18s 2s/step - loss: 0.3988 - accuracy: 0.9233
Epoch 45/50
10/10 [==============================] - 17s 2s/step - loss: 0.3832 - accuracy: 0.9195
Epoch 46/50
10/10 [==============================] - 17s 2s/step - loss: 0.2439 - accuracy: 0.9425
Epoch 47/50
10/10 [==============================] - 18s 2s/step - loss: 0.1898 - accuracy: 0.9338
Epoch 48/50
10/10 [==============================] - 18s 2s/step - loss: 0.2078 - accuracy: 0.9549
Epoch 49/50
10/10 [==============================] - 17s 2s/step - loss: 0.1747 - accuracy: 0.9511
Epoch 50/50
10/10 [==============================] - 18s 2s/step - loss: 0.2632 - accuracy: 0.9789
print("Test Accuracy:", accuracy)
0.9789
train_df = pd.DataFrame(training_hist.history).reset_index()
fig = px.area(train_df,
            x='index',
            y='loss',
            template='plotly_dark',
            color_discrete_sequence=['rgb(18, 115, 117)'],
            title='Training Loss x Epoch Number',
           )

fig.update_yaxes(range=[0,2])
fig.show()
Evaluate Results
results = model.evaluate_generator(test_gen)
preds   = model.predict_generator(test_gen)
print('The current model achieved a categorical accuracy of {}%!'.format(round(results[1]*100,2)))
The current model achieved a categorical accuracy of 82.37%!
summarized_confusion_matrix = np.sum(multilabel_confusion_matrix(pd.get_dummies(test['facial_expression']), preds >= 0.5),axis=0)
fig = px.imshow(summarized_confusion_matrix,
                template ='plotly_dark',
                color_continuous_scale = px.colors.sequential.Blugrn
                )


fig.update_layout(title_text='Confusion Matrix', title_x=0.5)
fig.show()


# Data loading and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'train_data_directory',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')  # assuming binary classification (drowsy vs non-drowsy)

validation_generator = test_datagen.flow_from_directory(
    'validation_data_directory',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

# Creating the CNN model
def create_cnn_model(input_shape):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten layer
    model.add(layers.Flatten())

    # Dense layers
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))  # Output layer, binary classification

    return model

# Create the model
input_shape = (64, 64, 3)
model = create_cnn_model(input_shape)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples/train_generator.batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples/validation_generator.batch_size)

# Evaluate the model
test_loss, test_acc = model.evaluate(validation_generator, verbose=2)
print('\nTest accuracy:', test_acc)