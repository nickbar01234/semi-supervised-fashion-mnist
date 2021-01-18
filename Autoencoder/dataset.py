import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from sklearn.model_selection import train_test_split 

class Dataset:
    def __init__(self, X: np.array, validation: bool = True, y: np.array = None, *args, **kwargs):
        self.validation = validation 
        if self.validation:
            train, validation = train_test_split(X, test_size = 5000, random_state = 42)
            train_x, train_y = self._transform(train)
            self.train_x, self.train_y = train_x, train_y

            valid_x, valid_y = self._transform(validation)
            self.valid_x, self.valid_y = valid_x, valid_y
        else:
            train_x, train_y = self._transform(X)
            self.train_x, self.train_y = self.train_x, train_y

    def dataset(self, show_sample: bool = False, batch_size: int = 32, buffer_size: int = 1024, *args, **kwargs):
        if show_sample:
            self._showSample()

        autotune = tf.data.experimental.AUTOTUNE
        if self.validation:
            train = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y)).batch(batch_size, drop_remainder = True)
            train = train.shuffle(buffer_size).prefetch(autotune)

            validation = tf.data.Dataset.from_tensor_slices((self.valid_x, self.valid_y)).batch(batch_size, drop_remainder = True)
            validation = validation.prefetch(autotune)
            return train, validation 
        else:
            ds = tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y)).batch(batch_size, drop_remainder = True)
            ds = ds.shuffle(buffer_size).prefetch(autotune)
            return ds 
    
    def _transform(self, x: np.array):
        y = x.copy()

        for i in range(len(x)):
            image = x[i]
            blur = cv2.GaussianBlur(image, (3, 3), 0)
            _, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            x[i] = threshold

        for i in range(len(y)):
            image = y[i]
            blur = cv2.GaussianBlur(image, (1, 1), 0)
            _, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(threshold, (2, 2), iterations = 1)
            edge = cv2.Canny(dilated, 0, 255)
            y[i] = edge

        return x.reshape(-1, 28, 28, 1).astype("float32") / 255.0, y.reshape(-1, 28, 28, 1).astype("Float32") / 255.0 

    def _showSample(self):
        sampled = np.random.randint(0, self.train_y.shape[0], size = 30)
        augmented_image = self.train_y[sampled]
        
        plt.figure(figsize=(20,20))
        columns = 10
        for i, image in enumerate(augmented_image):
            image = image.reshape(28, 28)
            ax = plt.subplot(len(augmented_image) / columns + 1, columns, i + 1)
            plt.axis('off')
            plt.subplots_adjust(bottom=0.1)
            plt.imshow(image, cmap='gray')
