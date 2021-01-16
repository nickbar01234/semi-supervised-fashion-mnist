from Autoencoder import autoencoder, callbacks, layers 

import pathlib 
import tensorflow as tf 
import numpy as np 
import os

class Train:
    def __init__(self, model: tf.keras.Model, loss: layers.Losses, optimizer:
                 tf.keras.optimizers.Adam, model_checkpoint: str =
                 None, history_path: str = None, *args, **kwargs):
        self.model = model
        self.optimizer = optimizer 
        self.loss = loss
        self.num_epoch = 1
        self.model_checkpoint = model_checkpoint
        self.history_path = history_path 

    def trainLoop(self, train, validation, epochs = 100, debug = False):
        tf.keras.backend.clear_session()
        tf.config.run_functions_eagerly(True)
        #os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

        print(f"Tensorflow version: {tf.__version__}")
        print(f"Eager execution: {tf.executing_eagerly()}")

        self._ckpt()
        self.train = train 
        self.validation = validation 

        len_train = tf.data.experimental.cardinality(train).numpy()
        len_valid = tf.data.experimental.cardinality(validation).numpy()
        self.callback = callbacks.Callback(len_train, len_valid, self.model, 5, checkpoint = self.history_path)


        for epoch in range(self.num_epoch, epochs):
            if debug:
                print(f"Learning rate: {self.optimizer.learning_rate}")

            train_log = self.train_step()
            valid_log = self.valid_step(epoch)
            history = self.callback.on_epoch_end(epoch, train_log, valid_log, self.model)
            
            self.model = self.callback.earlyStopping(epoch)
            self.optimizer = self.callback.exponentialDecay(self.optimizer, -0.003, epoch)
            
            self.ckpt.step.assign_add(1)
            save_path = self.manager.save()

            if self.model.stop_training:
                break 

        return self.model, history 

    def _ckpt(self):
        if self.model_checkpoint != None:
            self.ckpt = tf.train.Checkpoint(step = tf.Variable(1), optimizer = self.optimizer, net = self.model)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.model_checkpoint, max_to_keep = 3) 
            if self.manager.latest_checkpoint:
                status = self.ckpt.restore(self.manager.latest_checkpoint).assert_nontrivial_match() 
                print(f"Resumed training from {self.manager.latest_checkpoint}")
                self.num_epoch = int(self.manager.latest_checkpoint.split("-")[-1]) + 1
            else:
                print(f"No checkpoint provided, begin training")
                
    @tf.function
    def train_step(self):
        avg_loss = []
        for step, image in enumerate(self.train):
            x, y = image 
            with tf.GradientTape() as tape:
                prediction = self.model(x, pipeline = "autoencoder", training = True)
                loss = self.loss.computeLoss(y, prediction)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.callback._progbar(step = step, train = loss)
            avg_loss.append(loss)
        return tf.reduce_mean(avg_loss).numpy()
    
    @tf.function
    def valid_step(self, epoch):
        avg_loss = []
        for step, image in enumerate(self.validation):
            x, y = image 
            prediction = self.model(x, pipeline = "autoencoder", training = False)
            loss = self.loss.computeLoss(y, prediction)
            self.callback._progbar(step = step, validation = loss)
            avg_loss.append(loss)
        
        if epoch % 10 == 0:
            self.callback._plot_image(prediction, y)

        return tf.reduce_mean(avg_loss).numpy() 

  
            
        