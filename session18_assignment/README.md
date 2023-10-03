# ERA1_Session18
## Assignment
    First part of assignment is to train your own UNet from scratch, you can use the dataset and strategy provided in this [link](https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406)
    
    However, you need to train it 4 times:
    • MP+Tr+BCE
    • MP+Tr+Dice Loss
    • StrConv+Tr+BCE
    • StrConv+Ups+Dice Loss
    and report your results.    
    
    **Second part**
    Design a variation of a VAE that:
    • takes in two inputs:
        ◦ an MNIST image, and
        ◦ its label (one hot encoded vector sent through an embedding layer)
    • Training as you would train a VAE
    • Now randomly send an MNIST image, but with a wrong label. Do this 25 times, and share what the VAE makes (25 images stacked in 1 image)
    • Now do this for CIFAR10 and share 25 images (1 stacked image)

## Solution - Notebook [s18_vae_mnist](https://github.com/sdev2030/ERA1_Transformer_PLUS/blob/main/session18_assignment/s18_vae_training_mnist.ipynb)
In this we will use VAE architecture from the Pytorch-Lightning bolt library, trained for 30 epochs.

We have implemented:
 - sending image and its label (one hot encoded vector sent through an embedding layer) as encoder input of shape bx2x28x28 where b stands for batch size of 32  
 - Decoder will generate an image based on sampled input from mean and STD of the encoder model

Following are the parameters Used for training:

        "batch_size": 32,
        "num_epochs": 30,
        "optimizer" : Adam
        "lr": 1**-4,

Following are the model parameter and training loss achieved in training the model for 30 epochs.
- Model Parameters - 20.1M
- Training ELBO Loss - -952 (Epoch Mean)

Graph showing training loss during training.

![Mnist Training Graph](https://github.com/sdev2030/ERA1_Transformer_PLUS/blob/main/session18_assignment/images/mnist_training_graph.png)

Result of 25 random images with wrong label

![Mnist Image](https://github.com/sdev2030/ERA1_Transformer_PLUS/blob/main/session18_assignment/images/mnist_image_with_wrong_label_result.png)

## Solution - Notebook [s18_vae_cifar10](https://github.com/sdev2030/ERA1_Transformer_PLUS/blob/main/session18_assignment/s18_vae_training_cifar10.ipynb)
In this we will send CIFAR10 images along with its label (one hot encoded passed through embedding layer) to encoder to train the VAE.

Following are the model parameter and training loss achieved in training the model for 30 epochs.
- Model Parameters - 20.1M
- Training ELBO Loss - -2900 (Epoch Mean)

Graph showing training loss during training.

![CIFAR10 Training Graph](https://github.com/sdev2030/ERA1_Transformer_PLUS/blob/main/session18_assignment/images/cifar10_training_graph.png)

Result of 25 random images with wrong label

![CIFAR10 Image](https://github.com/sdev2030/ERA1_Transformer_PLUS/blob/main/session18_assignment/images/cifar10_image_with_wrong_label_result.png)

