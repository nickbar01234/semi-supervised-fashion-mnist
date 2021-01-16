from autoencoder import AutoEncoder
from layers import * 

import pathlib 

class Train:
    def __init__(self, model: tf.keras.models.Model, loss: , callback, checkpoint: str = None, *args, **kwargs):
        self.model = model
        self.loss = loss
        self.num_epoch = 1
        self.callback = callback 
        self.checkpoint = checkpoint
        
    def trainLoop(self):
        if not self.checkpoint is 
        