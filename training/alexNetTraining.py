import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
import os
import sys

# Adiciona o diretório raiz ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.alexNet.alexNet_model import AlexNet  # Agora a importação deve funcionar

# Criando diretório para salvar o modelo treinado
SAVE_DIR = "models/alexNet/saved_model"

# Carregar o dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Redimensionar as imagens para 227x227
x_train = tf.image.resize(x_train, (227, 227)).numpy()
x_test = tf.image.resize(x_test, (227, 227)).numpy()

# Normalizar os dados para a faixa [0,1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Converter rótulos para one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Criar gerador de dados para aumentar a variedade do conjunto de treinamento
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(x_train)

# Criar e compilar o modelo AlexNet
model = AlexNet(numero_classes=10).model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
batch_size = 128
epochs = 50

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    validation_data=(x_test, y_test),
    epochs=epochs,
    verbose=1
)

# Salvar o modelo treinado
model.save(os.path.join(SAVE_DIR, "alexnet_cifar10.h5"))

print("Treinamento concluído e modelo salvo em", SAVE_DIR)
