# Image Deblurring Models Comparison
## MSDS 696 Project that compares different models on deblurring images and improving resolution

This capstone project is the final project of the Data Science Practicum II course at Regis University. This project compares the image processing results of three distinct models. 

### Methodology and Datasources
The methodlogy of this project is fairly simple. Using several datasets, the models are trained separate times and tested each time. Each time training and tested, it is with either a completely different dataset or a different validation subsection (one of the datasets has different types of blurred images). The trained models are then tested with a blurred image to see if improvements can be gleaned from the model. The results of each model are then compared against the other models. The outcomes are evaluated through subjective and objective means. What we see is the subjective evaluation and the objective evaluation are three measurements, MSE, PSNR, and SSIM.
#### Quantative Metrics
The quantative measurements of the images and outputs are the Mean Squared Error (MSE), Peak Signal to Noise Ratio (PSNR), and the Structurally Similarity Index Measure (SSIM). The code uses the following python library, https://scikit-image.org/docs/stable/api/skimage.metrics.html
* MSE https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.mean_squared_error
* PSNR https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.peak_signal_noise_ratio
* SSIM https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity
These metrics are a compartitive metric. In that, they compare the original image to the input image and to the output image.
#### Datasources
For training and testing, this project uses two datasources with one of the datasources used twice as it has different type of blurred images for validation and testing.
* Blur Dataset https://www.kaggle.com/datasets/kwentar/blur-dataset
* Celebrity Images (part of this big list of image sets) https://www.kaggle.com/datasets/jishnuparayilshibu/a-curated-list-of-image-deblurring-datasets

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

### Results
The following images are the results of three different passes of each model with a different dataset. On the images you see below, the top row of each set is from the CNN, the middle row is from the Diffusion model, and the last row is from the GAN.

#### Deblur Dataset and Gaussian Blurred Images
![image](https://github.com/user-attachments/assets/bae28d1f-660a-4585-9335-110afb9436d6)

#### Deblur Dataset and Motion Blurred Images
![image](https://github.com/user-attachments/assets/d77d5bd2-2fb3-47dc-b2d1-4a0c5057a7a6)

#### Celebrity Image Dataset
![image](https://github.com/user-attachments/assets/39c63680-a188-4ce7-a66e-27be4f12a5d4)








