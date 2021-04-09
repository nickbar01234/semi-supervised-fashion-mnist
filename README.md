# Classification using AutoEncoder

This is a toy project for practicing implementing tensorflow, image preprocessing, and sckit-learn on FashionMnist dataset. I also implemented some aspects of tensorflow train loop, such as customizing callbacks and train loop. To begin type on command line `clone https://github.com/nickbar01234/semi-supervised-fashion-mnist`

The general idea I had was that to predict a piece of gray-scale piece of clothing, a model only needs to know about the positional space on the Cartesian coordinates. To further reinforce this idea, I decided to train an auto encoder. 

Auto encoder is a powerful denoising deep learning concept to capture importance features. The general idea is to predict the train set on itself. I used the following steps:

1) **Apply a Gaussian blur** 

![](https://i.imgur.com/9Zkdp7r.png)

2) **Apply thresholding**

![](https://i.imgur.com/Qf83QUw.png)

3) **Dilation to fill noises**

![](https://i.imgur.com/6dnN1Ma.png)

4) **Apply Canny edge detection**

![](https://i.imgur.com/eRRIwKQ.png)

The *output* after step 4 will be the target output for the train set. For the input data, I only applied Gaussian Blur to filter out initial noises. This is a sample output after 89 epochs, 

![](https://i.imgur.com/VLAAmk0.png)

I then extracted out the encoder part of the trained encoder and flatten the weights. In theory, this should represent the most important features of a piece of clothing. I then use this to fit into random forest classifier. 

