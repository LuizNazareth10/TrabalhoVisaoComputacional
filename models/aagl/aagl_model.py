from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
import os

# arquitetura própria 

# CONV2D -> RELU -> MAXPOOL -> 
# CONV2D -> RELU -> MAXPOOL ->
# CONV2D -> RELU -> MAXPOOL -> 
# CONV2D -> RELU -> MAXPOOL -> 
# CONV2D -> RELU -> MAXPOOL -> 
# FLATTEN -> DENSE_1 -> 
# DENSE_2 -> DROPOUT -> 
# DENSE_3 -> DROPOUT ->  
# DENSE_4-> output

# estrutura da alexnet porém os parâmetros internos das camadas foram alterados
# para uma arquitetura própria, funcionando com imagens 32x32x3, 
# diferente da alexnet que funciona com 227x227x3
# a arquitetura foi alterada para funcionar com imagens de menor resolução

# O que foi ajustado?
# Kernels menores (3x3 em vez de 11x11 e 5x5)
# Redução do número de filtros (começa com 64 em vez de 96)
# Pooling ajustado para evitar redução excessiva
# Menos neurônios nas camadas densas (9216 → 512)

class AAGL(object):
    def __init__(self, numero_classes=10):
        self.numero_classes = numero_classes
        self.input_shape = (32, 32, 3)
        self.model = self.create_Alexnet(self.input_shape, self.numero_classes)  

    def conv2d(self, filters, kernel_size, strides=1, activation='relu'):
        return layers.Conv2D(filters, kernel_size, strides=strides, padding="same", activation=activation)

    def max_pooling(self, kernel_size=2, strides=2):
        return layers.MaxPooling2D(pool_size=kernel_size, strides=strides, padding="same")

    def create_Alexnet(self, input_shape, num_classes):
        inputs = layers.Input(shape=input_shape)
        x = self.conv2d(64, 3, 1, "relu")(inputs)  # 32x32x64
        x = self.max_pooling(2, 2)(x)              # 16x16x64
        x = self.conv2d(128, 3, 1, "relu")(x)      # 16x16x128
        x = self.max_pooling(2, 2)(x)              # 8x8x128
        x = self.conv2d(256, 3, 1, "relu")(x)      # 8x8x256
        x = self.conv2d(256, 3, 1, "relu")(x)      # 8x8x256
        x = self.conv2d(128, 3, 1, "relu")(x)      # 8x8x128
        x = self.max_pooling(2, 2)(x)              # 4x4x128
        x = layers.Flatten()(x)                    # 2048
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(num_classes, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=x)
        return model