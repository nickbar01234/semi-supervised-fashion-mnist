class Dataset:
    def __init__(self, pipeline: str, X: np.array, y: np.array = None, *args, **kwargs):
        self.x = X.copy()
        self.pipeline = pipeline
        if pipeline == "autoencoder":
            self.y = self.x.copy()
        elif pipeline == "classifcation":
            self.y = y 
            pass
        else:
            raise NotImplementedError
        
    def dataset(self, show_sample: bool = False, batch_size: int = 32, buffer_size: int = 1024, *args, **kwargs):
        if self.pipeline == "autoencoder":
            self._autoEncoderPipeline()
            self.y = self.y / 255.0 
            if show_sample:
                self._showSample()
        elif self.pipeline == "classifcation":
            pass
        
        self.x = self.x / 255.0
        autotune = tf.data.experimental.AUTOTUNE
        ds = tf.data.Dataset.from_tensor_slices((self.x, self.y)).batch(batch_size, drop_remainder = True)
        ds = ds.shuffle(buffer_size).prefetch(autotune)
        return ds
        
    def _autoEncoderPipeline(self):
        for i in range(len(self.x)):
            image = self.x[i]
            blur = cv2.GaussianBlur(image, (3, 3), 0)
            self.x[i] = blur 
        for i in range(len(self.y)):
            image = self.y[i]
            blur = cv2.GaussianBlur(image, (1, 1), 0)
            _, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(threshold, (2, 2), iterations = 1)
            edge = cv2.Canny(dilated, 0, 255)
            self.y[i] = edge
    
    def _showSample(self):
        sampled = np.random.randint(0, self.y.shape[0], size = 30)
        augmented_image = self.y[sampled]
        
        plt.figure(figsize=(20,20))
        columns = 10
        for i, image in enumerate(augmented_image):
            image = image.reshape(28, 28)
            ax = plt.subplot(len(augmented_image) / columns + 1, columns, i + 1)
            plt.axis('off')
            plt.subplots_adjust(bottom=0.1)
            plt.imshow(image, cmap='gray')