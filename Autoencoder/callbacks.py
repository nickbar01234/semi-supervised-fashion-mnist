import tensorflow as tf 
import json
import ast
import numpy as np 
import matplotlib.pyplot as plt 

class Callback:
    def __init__(self, len_train: np.int64, len_valid: np.int64, model: tf.keras.models.Model, patience: int, restore_weights: bool = True, checkpoint: str = None, *args, **kwargs):
        self.model = model 
        self.checkpoint = checkpoint 

        self.train_progbar = tf.keras.utils.Progbar(len_train) 
        self.valid_progbar = tf.keras.utils.Progbar(len_valid) 

        if self.checkpoint == None:
            try:
                with open(self.checkpoint + "/history.json", "r") as log:
                    file = log.read()
                self.history = ast.literal_eval(file)
                for key in history:
                    tmp = self.history[key]
                    self.history[key] = [float(log) for log in tmp]
                print("Sucessfully loaded a history log")
            except:
                print("Failed to open a history log with a checkpoint provided")
        else:
            self.history = {"train_loss": [], "valid_loss": []}

        self.patience = patience 
        self.wait = 0
        self.best_loss = None
        self.restore_weights = restore_weights
        if self.restore_weights:
            self.best_weights = self.model.get_weights()

        self.learning_rate = 0
    
    def _progbar(self, step = None, train = None, validation = None):
        if train != None:
            self.train_progbar.update(
                step + 1,
                [
                    ("Train", train)
                ]
            )
        elif validation != None:
            self.valid_progbar.update(
                step + 1,
                [
                    ("Valid", validation)
                ]
            )

    def on_epoch_end(self, epoch: int, train: float, validation: float, model: tf.keras.models.Model):
        self.model = model 
        tmp = [train, validation]
        for index, key in enumerate(self.history):
            self.history[key].append(tmp[index])
        
        if epoch % 10 == 0:
            fig, ax = plt.subplots(1, 1, figsize = (12, 12))
            ax.plot(np.arange(1, epoch + 1, 1), self.history["train_loss"], c = 'r', label = "train loss")
            ax.plot(np.arange(1, epoch + 1, 1), self.history["valid_loss"], c = 'b', label = "valid_loss")
            ax.legend(loc = "right")
            plt.gca().invert_yaxis()
            plt.show()
        
        if not self.checkpoint is None:
            with open(str(self.checkpoint) + "/history.json", "w") as log:
                tmp_history = self.history
                for key in tmp_history:
                    tmp_log = tmp_history[key]
                    tmp_history[key] = [str(log) for log in tmp_log]
                history = json.dumps(tmp_history)
                log.write(history)

        return self.history

    def earlyStopping(self, epoch: int):
        if self.best_loss == None:
            self.best_loss = self.history["valid_loss"][-1]
            return self.model
        
        current_loss = self.history["valid_loss"][-1]
        if current_loss < self.best_loss:
            print(f"For epoch {epoch}, loss {current_loss} improved from {self.best_loss}")
            self.best_loss = current_loss 
            self.wait = 0
            if self.restore_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            print(f"For epoch {epoch}, loss {current_loss} did not improve from {self.best_loss}")
            if self.wait > self.patience:
                self.model.stop_training = True 
                print(f"Stopp training at epoch {epoch}")
                if self.restore_weights:
                    self.model.set_weights(self.best_weights)
        return self.model 
    
    def exponentialDecay(self, optimizer: tf.keras.optimizers, k: float, epoch: int):
        self.learning_rate = optimizer.learning_rate 
        if self.learning_rate > 1e-5:
            optimizer.learning_rate = self.learning_rate * np.exp([k * epoch])[0]
            self.learning_rate = optimizer.learning_rate 
        return optimizer 

    @staticmethod
    def _plot_image(predictions: tf.Tensor, ground_truth: tf.Tensor):
        predictions = np.stack(predictions).reshape(-1, 28, 28)
        ground_truth = np.stack(ground_truth).reshape(-1, 28, 28)
        sample = np.random.randint(0, predictions.shape[0], size = 1)

        fig, ax = plt.subplots(1, 2, figsize = (10, 10))
        ax[0].imshow(predictions[sample].reshape(28, 28), cmap = "gray")
        ax[0].set_title("Prediction")

        ax[1].imshow(ground_truth[sample].reshape(28, 28), cmap = "gray")
        ax[1].set_title("Ground Truth")

        plt.show()
  