# Image Deblurring Models Comparison
## MSDS 696 Project that compares different models on deblurring images and improving resolution

This capstone project is the final project of the Data Science Practicum II course at Regis University. This project compares the image processing results of three distinct models. 

### Methodology and Datasources


### Models and Networks
As mentioned in the introduction, there are three different models used, CNN, Diffusion, and GAN. Below are brief descriptions of each.
#### Convolutional Neural Network (CNN)
A CNN ( https://en.wikipedia.org/wiki/Convolutional_neural_network ) is a neural network, https://en.wikipedia.org/wiki/Neural_network_(machine_learning), that uses convolution kernals, https://en.wikipedia.org/wiki/Convolution, in the network. The architecture is made up of an input layer, hidden layers, and then the output layer. Here is a basic diagram of a classification CNN.

![image](https://github.com/user-attachments/assets/3bf9ac42-c92f-442e-9fa7-55cf143bd534)

#### Diffusion
Diffusion models, https://en.wikipedia.org/wiki/Diffusion_model, have a process that adds noise on a forward pass of the network. Essentially it adds enough noise to effective destroy the image. The reverse pass reconstructs the image by reversing the noising process. Below is visual representation of the process.

![image](https://github.com/user-attachments/assets/0cf31d2a-0bcc-4e0b-9db8-e6f0c80fc480)

#### Generative Adversarial Network (GAN)
A GAN, https://en.wikipedia.org/wiki/Generative_adversarial_network, is made up of two neural networks. The two networks compete against each other and are referred to as a Generator and a Discriminator. The Generator adds noise to an image to try to fool the the Discriminator. Through a feedback loop they compete with other to get to a result where the Disciminator cannot tell the difference between the fake image and a real image. As with the other models, below is a visual look at the process.

![image](https://github.com/user-attachments/assets/1ac9b98a-0786-49f2-89da-65416b7358d4)





