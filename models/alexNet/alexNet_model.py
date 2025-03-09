from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
import os

# CONV2D -> RELU -> MAXPOOL -> 
# CONV2D -> RELU -> MAXPOOL ->
# CONV2D -> RELU -> MAXPOOL -> 
# CONV2D -> RELU -> MAXPOOL -> 
# CONV2D -> RELU -> MAXPOOL -> 
# FLATTEN -> DENSE_1 -> 
# DENSE_2 -> DROPOUT -> 
# DENSE_3 -> DROPOUT ->  
# DENSE_4-> output

class AlexNet(object):
    def __init__(self, numero_classes=1000):
        self.numero_classes = numero_classes
        self.input_shape = (227, 227, 3)
        self.model = self.create_Alexnet(self.input_shape, self.numero_classes)  
    
    def conv2d(self,filters, kernel_size, strides=1, activation='relu'):
        return layers.Conv2D(filters, kernel_size, strides=strides, padding="same", activation=activation)
        
    def max_pooling(self,kernel_size = 3, strides = 2):
        return layers.MaxPooling2D(pool_size=kernel_size, strides=strides, padding="same")
    
    
    def create_Alexnet(self, input_shape, num_classes):
        inputs = layers.Input(shape=input_shape)
        x = self.conv2d(96, 11, 4, "relu")(inputs)
        x = self.max_pooling(3, 2)(x)
        x = self.conv2d(256, 5, 1, "relu")(x)
        x = self.max_pooling(3, 2)(x)
        x = self.conv2d(384, 3, 1, "relu")(x)
        x = self.conv2d(384, 3, 1, "relu")(x)
        x = self.conv2d(256, 3, 1, "relu")(x)
        x = self.max_pooling(3, 2)(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(9216, activation="relu")(x)
        x = layers.Dense(4096, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(4096, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        # x = layers.Dense(1000, activation = "softmax")(x)
        x = layers.Dense(num_classes, activation = "softmax")(x)
    
        model = Model(inputs = inputs, outputs = x)
        
        return model
        
    