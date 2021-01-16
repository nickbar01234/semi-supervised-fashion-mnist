import tensorflow as tf 
from tensorflow.keras.layers import Layer, Conv2D, ReLU, Dropout, MaxPooling2D, Conv2DTranspose, GlobalAveragePooling2D

class Encoder(Layer):
    def __init__(self, filters: int, kernel_size: int, strides: int, name: str, *args, **kwargs):
        super(Encoder, self).__init__(name = name, *args, **kwargs)
        self.filters = filters 
        self.kernel_size = kernel_size
        self.strides = strides
    
    def build(self, input_shape: tuple, *args, **kwargs):
        self.conv1 = Conv2D(filters = self.filters, kernel_size = self.kernel_size,
                        strides = self.strides)
        self.conv2 = Conv2D(filters = self.filters, kernel_size = self.kernel_size, strides = self.strides)
        self.relu = ReLU()
    
    def call(self, inputs: tf.python.framework.ops.EagerTensor, *args, **kwargs):
        downsample = self.conv1(inputs)
        downsample = self.conv2(downsample)
        downsample = self.relu(downsample)
        return downsample 

class Decoder(Layer):
    def __init__(self, filters: int, kernel_size: int, strides: int, name: str, activation: str = None, *args, **kwargs):
        super(Decoder, self).__init__(name = name, *args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        
    def build(self, input_shape: tuple, *args, **kwargs):
        self.upsample1 = Conv2DTranspose(filters = self.filters, 
                                    kernel_size = self.kernel_size, 
                                    strides = self.strides, 
                                    activation = self.activation)
        
        self.upsample2 = Conv2DTranspose(filters = self.filters, 
                                    kernel_size = self.kernel_size, 
                                    strides = self.strides, 
                                    activation = self.activation)
        self.relu = ReLU()
    
    def call(self, inputs: tf.python.framework.ops.EagerTensor, *args, **kwargs):
        upsample = self.upsample1(inputs)
        upsample = self.upsample2(upsample)
        upsample = self.relu(upsample)
        return upsample 
    
class Losses:
    def __init__(self):
        self.mse = tf.keras.losses.MeanSquaredError()
    
    @tf.function
    def computeLoss(self, ground_truth, prediction):
        loss = self.mse(ground_truth, prediction)
        return loss
    