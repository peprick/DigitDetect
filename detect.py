

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers



(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Train:","x=",x_train.shape,"y=",y_train.shape)
print("Test:","x=",x_test.shape,"y=",y_test.shape)

x_train = x_train.astype('float')/255.
x_test = x_test.astype('float')/255.

plt.figure(figsize=(20,2))
for i in range(10):
        plt.subplot(1,10,i+1)
        plt.imshow(x_train[i], cmap='binary')
        plt.xticks([])
        plt.yticks([])
       
        plt.xlabel(y_train[i])

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))


model.save('my_model_digits.h5')
print("Saved model to disk")


from tensorflow.keras.models import load_model

model=load_model('my_model_digits.h5')

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
 
def load_image(filename):
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	img = img_to_array(img)
	img = img.reshape(1, 28, 28, 1)
	img = img.astype('float32')
	img = img / 255.0
	return img
 
def predict_sample():
	img = load_image('sample2.jpg')
	model = load_model('my_model_digits.h5')
	digit = model.predict_classes(img)
	print(digit[0])
 
predict_sample()



plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

print(test_acc)

