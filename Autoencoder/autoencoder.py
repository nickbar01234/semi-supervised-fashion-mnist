from Autoencoder.layers import * 

import tensorflow as tf 
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense

class AutoEncoder(Model):
    def __init__(self, name = "AutoEncoder", *args, **kwargs):
        super(AutoEncoder, self).__init__(name = "AutoEncoder", *args, **kwargs)
        #(-1, 28, 28, 1)
        self.encoder1 = Encoder(3, 3, 1, name = "encoder1")
        #(-1, 24, 24, 3)
        self.encoder2 = Encoder(8, 3, 1, name = "encoder2")
        #(-1, 20, 20, 8)
        self.encoder3 = Encoder(16, 2, 2, name = "encoder3")
        #(-1, 5, 5, 16)
        self.encoder4 = Encoder(32, 2, 1, name = "encoder4")
        #(-1, 3, 3, 32)
        self.encoder5 = Encoder(64, 1, 1, name = "encoder5")
        #(-1, 3, 3, 64)
        self.flatten = Flatten()
        #(-1 * 576)
        self.decoder1 = Decoder(16, 2, 1, name = "decoder1")
        #(-1, 5, 5, 16)
        self.decoder2 = Decoder(8, 2, 2, name = "decoder2")
        #(-1, 20, 20, 8)
        self.decoder3 = Decoder(3, 3, 1, name = "decoder3")
        #(-1, 24, 24, 3)
        self.decoder4 = Decoder(1, 3, 1, name = "decoder4", activation = "sigmoid")
        #(-1, 28, 28, 1)
        
    def call(self, inputs: tf.Tensor, pipeline: str, training: bool = False):
        encoded = self.encoder1(inputs)
        encoded = self.encoder2(encoded)
        encoded = self.encoder3(encoded)
        encoded = self.encoder4(encoded)
        encoded = self.encoder5(encoded)
        
        if pipeline == "classifcation":
            output = self.flatten(encoded)
            return output
        
        decoded = self.decoder1(encoded)
        decoded = self.decoder2(decoded)
        decoded = self.decoder3(decoded)
        output = self.decoder4(decoded)
        return output 
        
   