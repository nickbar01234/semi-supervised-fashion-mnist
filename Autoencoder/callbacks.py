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
            self.history = {"train_mse": [], "valid_mse": [], "train_entropy": [], "valid_entropy": []}

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
                    ("train_mse", train[0]),
                    ("train_entropy", train[1])
                ]
            )
        elif validation != None:
            self.valid_progbar.update(
                step + 1,
                [
                    ("valid_mse", validation[0]),
                    ("valid_entropy", validation[1])
                ]
            )

    def on_epoch_end(self, epoch: int, train: float, validation: float, model: tf.keras.models.Model):
        self.model = model 

        tmp = [train[0], validation[0], train[1], validation[1]]
        for index, key in enumerate(self.history):
            self.history[key].append(tmp[index])
        
        if epoch % 10 == 0:
            fig, ax = plt.subplots(1, 2, figsize = (12, 12))
            ax[0].plot(np.arange(1, epoch + 1, 1), self.history["train_mse"], c = 'r', label = "train mse")
            ax[0].plot(np.arange(1, epoch + 1, 1), self.history["valid_mse"], c = 'b', label = "valid_mse")
            
            
            ax[1].plot(np.arange(1, epoch + 1, 1), self.history["train_entropy"], c = 'r', label = "train_entropy")
            ax[1].plot(np.arange(1, epoch + 1, 1), self.history["valid_entropy"], c = 'b', label = "valid_entropy")

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
            self.best_loss = self.history["valid_mse"][-1]
            return self.model
        
        current_loss = self.history["valid_mse"][-1]
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
    def _plot_image(x_train: tf.Tensor, predicted_image: tf.Tensor, x_label: tf.Tensor, predicted_label: tf.Tensor):

        label = {
            0: "T-shirt",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
        }

        x_train = np.stack(x_train).reshape(-1, 28, 28)
        predicted_image = np.stack(predicted_image).reshape(-1, 28, 28)
        x_label = x_label.numpy()
        predicted_label = predicted_label.numpy()

        sample = np.random.randint(0, predicted_image.shape[0], size = 1)

        fig, ax = plt.subplots(1, 2, figsize = (10, 10))
        ax[0].imshow(predicted_image[sample].reshape(28, 28), cmap = "gray")
        ax[0].set_title(f"Created image predicted {label[np.argmax(predicted_label[sample])]} for {label[x_label]}")

        ax[1].imshow(ground_truth[sample].reshape(28, 28), cmap = "gray")
        ax[1].set_title("Ground Truth")

        plt.show()
  