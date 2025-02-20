from tensorflow.keras import layers
from tensorflow.keras.models import Model, load_model
import os

##2 x CONV2D -> RELU -> MAXPOOL -> 2 x CONV2D -> RELU ->
# MAXPOOL -> 2 x CONV2D -> RELU -> MAXPOOL -> 2 x CONV2D -> 
# RELU -> MAXPOOL -> 2 x CONV2D -> RELU -> MAXPOOL -> FLATTEN -> 
# FC1 -> FC2 -> FC3 -> output
class VGG13(object):
    def __init__(self):
        self.input_shape = (227, 227, 3)
        self.model = self.create_VGG13(self.input_shape)
        
    def conv2d(self,filters, kernel_size=3, strides=1, activation='relu'):
        return layers.Conv2D(filters, kernel_size, strides=strides, padding="same", activation=activation)
    
    def max_pooling(self,kernel_size = 2, strides = 2):
        return layers.MaxPooling2D(pool_size=kernel_size, strides=strides, padding="same")
    
    def create_VGG13(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        x = self.conv2d(64, 3, 1, "relu")(inputs)
        x = self.conv2d(64, 3, 1, "relu")(x)
        x = self.max_pooling(2, 2)(x)
        x = self.conv2d(128, 3, 1, "relu")(x)
        x = self.conv2d(128, 3, 1, "relu")(x)
        x = self.max_pooling(2, 2)(x)
        x = self.conv2d(256, 3, 1, "relu")(x)
        x = self.conv2d(256, 3, 1, "relu")(x)
        x = self.max_pooling(2, 2)(x)
        x = self.conv2d(512, 3, 1, "relu")(x)
        x = self.conv2d(512, 3, 1, "relu")(x)
        x = self.max_pooling(2, 2)(x)
        x = self.conv2d(512, 3, 1, "relu")(x)
        x = self.conv2d(512, 3, 1, "relu")(x)
        x = self.max_pooling(2, 2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(4096, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(4096, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1000, activation = "softmax")(x)
        
        model = Model(inputs = inputs, outputs = x)
        
        return model