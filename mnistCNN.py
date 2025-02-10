import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

''' 60000 is batch size(number of images) in 28x28 pixel size. 1 is dimension for grayscale.
we are converting the integers (0-255) to float as cnn work better on float, and /255 is to normalise all to between (0,1)'''

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

''' to_categorical converts class labels to one-hot encoded vectors. i.e here from integres(0-9) to binary vectors
of length 10. example for 3 is [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]'''

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape = (28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D (64,(3,3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64,activation = 'relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation = 'softmax')

])

'''1st layer we use 3x3 filter on 28x28, as output we get 26x26 with 32 filters
2nd layer is max pooling of 2x2. so for every 2x2 block we select max value, which reduces size of input, here output we get is 13x13
1st conv layer extracts basic features, next layers extracts more complex features. rule is for extracting general features
(1st layer) we keep less filters. furthur layers we keep more filters(usually double)

next layer is flatten, which converts 3x3x64(after last conv) becomes 576.
'''

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=2, batch_size=64,validation_split=0.2)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'\nTest accuracy: {test_acc:.4f}')
predictions = model.predict(test_images)

# Prediction from test labels
for i in range(7, 10):
    plt.figure(figsize=(3, 3))
    plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])  # Get the predicted class index
    true_label = np.argmax(test_labels[i])  # Convert one-hot encoded label to integer
    color = 'green' if predicted_label == true_label else 'red'
    plt.xlabel(f'Predicted = {predicted_label} Actual label = {true_label}', color=color)
    plt.show()
