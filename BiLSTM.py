import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Example data loading and preprocessing (you need to replace this with your data loading/preprocessing logic)
# X_train, y_train = load_training_data()
# X_test, y_test = load_testing_data()

# Define the BiLSTM model
model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Bidirectional(LSTM(64)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])


def create_bilstm_model():
    model = Sequential()

    # First BiLSTM Layer
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(sequence_length, feature_size)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # Second BiLSTM Layer
    model.add(Bidirectional(LSTM(64)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # Fully connected layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    # Output layer for classification (e.g., 4 classes: crying, sad, angry, drowsy)
    model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

Epoch 1/50
10/10 [==============================] - 34s 3s/step - loss: 1.5729 - accuracy: 0.7450
Epoch 2/50
10/10 [==============================] - 18s 2s/step - loss: 1.3485 - accuracy: 0.7641
Epoch 3/50
10/10 [==============================] - 17s 2s/step - loss: 1.5754 - accuracy: 0.7833
Epoch 4/50
10/10 [==============================] - 18s 2s/step - loss: 1.5947 - accuracy: 0.7718
Epoch 5/50
10/10 [==============================] - 17s 2s/step - loss: 1.3206 - accuracy: 0.7843
Epoch 6/50
10/10 [==============================] - 18s 2s/step - loss: 1.4667 - accuracy: 0.7747
Epoch 7/50
10/10 [==============================] - 18s 2s/step - loss: 1.5616 - accuracy: 0.7862
Epoch 8/50
10/10 [==============================] - 17s 2s/step - loss: 1.2613 - accuracy: 0.8035
Epoch 9/50
10/10 [==============================] - 18s 2s/step - loss: 1.1203 - accuracy: 0.8102
Epoch 10/50
10/10 [==============================] - 18s 2s/step - loss: 1.5296 - accuracy: 0.7939
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
10/10 [==============================] - 17s 2s/step - loss: 0.8393 - accuracy: 0.8274
Epoch 19/50
10/10 [==============================] - 17s 2s/step - loss: 0.7805 - accuracy: 0.8811
Epoch 20/5
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
10/10 [==============================] - 18s 2s/step - loss: 0.5774 - accuracy: 0.8859
Epoch 28/50
10/10 [==============================] - 18s 2s/step - loss: 0.4023 - accuracy: 0.8945
Epoch 29/50
10/10 [==============================] - 18s 2s/step - loss: 0.4913 - accuracy: 0.9012
Epoch 30/50
10/10 [==============================] - 17s 2s/step - loss: 0.6177 - accuracy: 0.8945
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
10/10 [==============================] - 17s 2s/step - loss: 0.1747 - accuracy: 0.9567
Epoch 50/50
10/10 [==============================] - 18s 2s/step - loss: 0.2632 - accuracy: 0.9777
print("Test Accuracy:", accuracy)
0.9777
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