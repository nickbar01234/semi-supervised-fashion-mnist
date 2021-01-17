import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from sklearn.model_selection import train_test_split 

class Dataset:
    def __init__(self, X: np.array, y: np.array, validation: bool = True, *args, **kwargs):
        self.validation = validation 
        self.X = X.copy()
        self.y_train = y 

    def dataset(self, show_sample: bool = False, batch_size: int = 32, shuffle: int = 32, *args, **kwargs):
        self._pipeline()
        if show_sample:
            self._sample()
        
        autotune = tf.data.experimental.AUTOTUNE

        train = tf.data.Dataset.from_tensor_slices((self.x_train, self.x_prediction, self.y_train))
        train = train.batch(batch_size, drop_remainder = True).shuffle(shuffle).prefetch(autotune)
        if self.validation:
            validation = tf.data.Dataset.from_tensor_slices((self.valid_x, self.valid_prediction, self.valid_y))
            validation = validation.batch(batch_size, drop_remainder = True).prefetch(autotune)
            return train, validation 
        else:
            return train 

    def _pipeline(self):   
        if self.validation:
            x_train, x_test, y_train, y_test = train_test_split(self.X, self.y_train, test_size = 5000, random_state = 42)
            x_train, x_prediction = self._transform(x_train)
            self.x_train, self.x_prediction = x_train, x_prediction 
            self.y_train = y_train

            valid_x, valid_prediction = self._transform(x_test)
            self.valid_x, self.valid_prediction = valid_x, valid_prediction 
            self.valid_y = y_test
        else: 
            x_train, x_prediction = self._transform(self.X)
            self.x_train, self.x_prediction = x_train, x_prediction 

    @staticmethod 
    def _transform(x: np.array):
        y = x.copy()

        for i in range(len(x)):
            image = x[i]
            try:
                blur = cv2.GaussianBlur(image, (3, 3), 0)
            except:
                print(image)
            _, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            x[i] = threshold
        
        for i in range(len(y)):
            image = y[i]
            blur = cv2.GaussianBlur(image, (1, 1), 0)
            _, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(threshold, (2, 2), iterations = 1)
            edge = cv2.Canny(dilated, 0, 255)
            y[i] = edge
        
        return x.astype("float32").reshape(-1, 28, 28, 1) / 255.0, y.astype("float32").reshape(-1, 28, 28, 1) / 255.0

    def _sample(self):
        sampled = np.random.randint(0, self.x_prediction.shape[0], size = 30)
        augmented_image = self.x_prediction[sampled]
        
        plt.figure(figsize=(20,20))
        columns = 10
        for i, image in enumerate(augmented_image):
            image = image.reshape(28, 28)
            ax = plt.subplot(len(augmented_image) / columns + 1, columns, i + 1)
            plt.axis('off')
            plt.subplots_adjust(bottom=0.1)
            plt.imshow(image, cmap='gray')
