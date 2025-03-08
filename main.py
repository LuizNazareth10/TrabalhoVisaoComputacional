from models.alexNet.alexNet_model import AlexNet
from models.vgg13.vgg13 import VGG13

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

len_train = len(x_train)
len_test = len(x_test)
print(f"Tamanho do conjunto de treino: {len_train}\nTamanho do conjunto de teste: {len_test}")

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    # Converter y_train[i] para escalar inteiro
    label_index = int(y_train[i])
    plt.xlabel(class_names[label_index])

    # Redimensionar imagens para 227x227x3
x_train = tf.image.resize(x_train[:1000], (227, 227))
x_test = tf.image.resize(x_test[:100], (227, 227))

# x_train = x_train[:1000]
# x_test = x_test[:100]

y_train = y_train[:1000]
y_test = y_test[:100]

# Normalizar os valores dos pixels para [0,1]
x_train = x_train / 255.0
x_test = x_test / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    # Converter y_train[i] para escalar inteiro
    label_index = int(y_train[i])
    plt.xlabel(class_names[label_index])

# Converter rótulos para one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

alexnet = AlexNet(numero_classes=10).model

# Compilar o modelo
alexnet.compile(optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Treinar o modelo
# alexnet.fit(datagen.flow(x_train, y_train, batch_size=32),
#             epochs=50,
#             validation_data=(x_test, y_test))

alexnet.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Salvar o modelo treinado
alexnet.save("alexnet_cifar10.h5")

# Avaliar no conjunto de teste
test_loss, test_acc = alexnet.evaluate(x_test, y_test)
print(f"Acurácia no conjunto de teste: {test_acc * 100:.2f}%")