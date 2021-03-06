<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
$$
\newcommand{\X}{\mathcal{X}}
\newcommand{\Y}{\mathcal{Y}}
\newcommand{\R}{\mathbb{R}}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
$$

- [Overview](#overview)
- [Motivation](#motivation)
  * [Dictionary Learning](#dicitonary-learning)
- [Constrained Recurrent Sparse Auto-Encoder](#constrained-recurrent-sparse-auto-encoder)
  * [Model Architecture](#model-architecture)
- [Random Projection](#random-projection)
- [Simulated Data](#simulated-data)
  * [Uncompressed](#uncompressed)
  * [Compressed](#compressed)
  * [Multiple Compression Matricies](#multiple-compression-matricies)
- [Application: MNIST](#adversarial-training-with-data-augmentation)
  * [Architectures](#MNIST-architectures)
  * [No Decoder Performance](#MNIST-no-encoder-performace)
  * [Autoencoder with Classifier](#MNIST-autoencoder-performance)
- [Conclusion](#conclusion)

# Overview
With advising from [Demba Ba](https://www.seas.harvard.edu/directory/demba) and [Bahareh Tolooshams](https://crisp.seas.harvard.edu/people/bahareh-tolooshams), this project aims to use a carefully designed auto-encoder to learn a dictionary for a set of observations, while enforcing a sparsity constraint on the produced encodings. This project uses the constrained recurrent sparse auto-encoder (CRsAE) to find a sprase encoding approximation using the Fast Iterative Shrinking-Threshold Algorithm (FISTA) to generate a sparse encoding approximation and then applies the same dictionary in the decoding step. In addition to evaluating the performance of this model, this project evaluates the performance of this model when evaluated on data that has been compressed through random projeciton.

# Motivation
In signal processing, dictionary learning is one of the most prominent frameworks for representation learning. Using a method that is neural network based opens the door for efficient computation from GPU based paralellization. Additionally, learning one set of weights used in both encoding and decoding allows for interpretability of the dictionary.
## Dictionary Learning
Dictionary learning can be described as learning a linear combination of vectors to represent an input such that each of vector is a column of a matrix (the dictionary) and the coefficients are vector representations. We focus on the case where these representations are sparse. More formally stated dictionary learning can be described in the following way. Given a set of inputs $$\Y = [y_1, \dots, y_K], y_i \in \R^{D}$$ we attempt to find a dictionary $$A \in \R^{D \times N}$$ and a representation or encoding $$X = [x_1, \dots, x_K], x_i \in \R^N$$ such that the reconstruction error is minimized $$\min||\Y - AX||$$. Enforcing a sparsity penalty, $$\lambda$$, we can write this as the following optimization problem.

$$ 
\argmin_{A \in \R^{D \times N, x_i \in \R^N}} \sum_{i=1}^K \frac{1}{2}||y_i -Ax_i||_2 + \lambda||x_i||_0
$$

Note: While the problem is written with $l_0$ norm this makes solving this problem NP-hard, so practically we use $l_1$ norm.

# Constrained Recurrent Sparse Auto-Encoder (CRsAE)
Originally proposed in Tolooshams et al. [^1], CRsAE is a recurrent auto-encoder architecture used for dictionary learning. In dictionary learning, we start knowing neither the dictionary, $$A$$, nor the encodings, $$x_i$$ that would reproduce the data. To simply the previous dual minimization problem we break the problem into a two-step update and take turns updating the encodings and then the dictionary. First we guess our dictionary and are left with $$K$$ of the following convex optimization problems to find our encodings. 

$$
\argmin_{x_i} \frac{1}{2}||y_i - Ax_i||_2 + \lambda||x_i||_1
$$

In CRsAE we perform this step using the forward pass of the encoder. Unlike many current methods, the forward pass of our autoencoder is easily parallelizable using GPUs. Next we update the dictionary by using our previously estimated encodings and minimizing reconstruction error.

$$
\argmin_{A} \sum_{i=1}^K\frac{1}{2}||y_i - Ax_i||_2 \text{ s.t. } ||a_n|| \leq 1 \text{ for } n=1,\dots,N
$$

We also constrain the columns of $A$ to be norm of 1 to ensure that the the energy of each reconstruction is captured in the encodings instead of the dictionary. This step is performed in CRsAE through backpropogation where the only model parameter considered is $A$. In practice this is performed using autograd.
## Model Architecture
To solve these optimization problems we use the following architecture.
!['CRsAE block'](/imgs/CRsAE_diagram.PNG) 
Starting with our data we apply FISTA using our dictionary $$A$$ to yield sparse encodings $$\hat{x}$$. We then decompress this encoding using the same dictionary $$A$$ by matrix multiplying our encoding by the dictionary. FISTA was first proposed by Beck and Teboulle [^2] and is a fast iterative procedure to find a sparse solution to the linear system in the above encoding approximation step. FISTA generates a sequence of sparse approximations using an $$l_1$$ norm penalty. The details of the algorithm are below in pseudocode and further reading can be seen in Beck and Teboulle [^2].

```python
# Let all x's and y's be column vectors with dimensionality of the encoding size (N)
# A is the dictionary (DxN)
# L is a parameter of FISTA that partially controls the rate of convergence
x_old = [0] * N
x_new = [0] * N
z_old = [0] * N
t_old = 0
for t in range(T):
    t_new = (1 + sqrt(1 + 4 * t_old * t_old)) / 2
    z_new = x_new + (t_old - 1) / t_new * (x_new - x_old)
    x_new = y_old + A_transpose * (y - A * z_old) / L

    # Apply shrinkage based off of size of penalty lambda
    x_new = ReLU(|x_new| - lambda / L) * sign(x_new)
    
    x_old = x_new
    t_old = t_new
    z_old = z_new

return x_new
```

The decoding step is a matrix multiplication of the dictionary by the encoding. Because of the sparsity of the encodings the decoding step can be thought of as selecting a few columns of the dictionary that are used to reconstruct the image. One notable aspect of this auto-encoder is that the decoding dictionary is used in the encoding step. This connection is imporant because the weights of the encoder can be interpreted as components in the reconstruction or features back in the original input space. 
# Random Projection
The basic idea of random projection is to multiply an input which is sparse by a rectangular matrix to project the input space to a lower dimension. The idea is that the original input space is too big for the amount of information captured and the projection is condensing many somewhat useful features into a few very useful features. There are many benefits from performing computation in lower dimension such as saving memory and performing fewer computations. These projection matricies can have many initializations but it is common to use Gaussian initialized orthogonal vectors. Overall the random projection process can be stated as such $$\phi \in \R^{M \times D}$$ s.t. $$ M << D $$ and we create our compressed features $$z_i = \phi y_i$$.

One exmaple of using random projection in the literature is from Pourkamali-Anaraki et al.[^3]. In this paper, random projection is applied to a simulated data set and traditional methods for learning dictionary learning are applied to the projected data. In order to take advantage of efficient computation the projection matricies are intialized with each entry drawn from {-1,0,1} with probabilities $$\{\frac{1}{2s}, 1- \frac{1}{s}, \frac{1}{2s}\}$$ for $$ s \geq 1 $$. Additionally, this paper uses multiple projection matricies for their dataset so blocks of data are compressed using the same $$\phi$$.

We apply random projection to CRsAE with both Gaussian and Sparse-Bernoulli intializations. In this setting the auto-encoder will receive an image that has been projected into compressed space, decompress that image by multiplying $$\phi^\top$$, apply FISTA to the decompressed image, decode the encoding back to image space using $$\phi A$$. 

!['CRsAE RP block'](/imgs/CRsAE_RP_diagram.PNG) 

Some important modifications to the original CRsAE diagram are the dimensions that the architecture receives and the dictionary that FISTA uses. The model only receives projected data, so in FISTA a matrix must be used to convert from compressed space $$\R^M$$ to encoding space $$\R^N$$. For this reason the product $$\phi A$$ is used instead of just $$A$$. In this model $$\phi$$ is a constant, so although FISTA uses $$\phi$$ to generate an encoding $$\phi$$ is never updated and only $$A$$ is modified in the dictionary update step. While this auto-encoder receives and outputs in compressed space $$\R^M$$, the encodings $$x_i$$'s can be taken and multiplied by $$A$$ instead of $$\phi A$$ to create reconstructions in the original uncompressed space $$\R^D$$.

TODO: Add images/plots and explain results
# Simulated Data
We generate simulated data assuming that there is a true dictionary and sparse encoding to be learned. We initialize sparse vectors of dimension 100 with 4 entries and then multiply them by a dictionary of dimension $$(100 \times 100)$$ to produce images. We draw the value of the sparse encodings from a uniform distribution on the interval $$[-5,-4] \cup [4,5]$$. In the following training setting we initialize our auto-encoder with a perturbed version of the dictionary used to generate the dataset. The below plots show the input images compared to the images recovered by the autoencoder after training, the simulated sparse encodings against the learned sparse encodings, and an error plot over training epoch. Our error is the maximal cosine distance over all the columns of the learned dictionary and initial dictionaries.
## Uncompressed
From these plots we can see that the auto-encoder is able to reproduce the images essentially perfectly. Additionally the encodings are able to be recovered with almost the same amplitude. We attribute the difference to the regularization factor. The last plot shows that the dictionary is able to converge to the true dictionary.

!['plots Identity simulated'](/imgs/sim_I_plots.PNG) 

## Compressed
In this setting we add a $$\phi$$ with dimension $$(80 \times 100)$$, which means that our encoder is working with data thats is 20% smaller. The auto-encoder is still able to recover images and the true encodings but the dictionary does not converge as nicely to the real dictionary.

!['plots Identity simulated'](/imgs/sim_80_plots.PNG)

Below is a plot demonstrating how dictionary error increases as the compression dimension decreases. The blue demonstrates having no $$\phi$$ while the other lines show deceasing compression dimensions.

!['compressed error plots'](/imgs/compressed_error_plots.png)

# Application: MNIST
We apply the CRsAE arhitecture to learning MNIST the handwritten digit dataset. Instead of only testing the auto-encoder, we attempt to classify based off of the encodings the sparse encodings that we generate.
## No Decoder Performance
In this setting we do not explicity attempt to learn a good image reconstruction. This architecture only performs the encoding step and then performs logistic regression on the sparse encodings. Therefore, the loss function of this architecture is the Cross entropy loss when classifying. We can, however, manually perform the decoding step by multiplying our sparse encodings by our learned dictionary manually.


 Below are plots of the sparse encodings produced, our manually reconstructed images, and the real inputted image. Lastly there is also a plot of the classification accuracy on the training and test sets.

!['MNIST image plots'](/imgs/MNIST_I_plots.PNG)
!['MNIST acc plots'](/imgs/MNIST_acc_plot.PNG)

From these plots we can see that the encodings are quite sparse and the reconstructions are quite good despite not explicitly telling the model to learn a good image representation. We also see that the model achieves 98% test accuracy, which is about as good as state of the art dense neural nets. Additionally, we can see the features that the dicionary is learning by plotting he columns of our dictionary as images.
!['MNIST features'](/imgs/MNIST_featuers.png)
From these pictures we can see that the dictionary is learning details of the images. Some image show curves and shaped that resemble the digits such the image in the far right that looks like a 3. Many images, however, just look like the initialized gaussian noise.
# Conclusion
Overall, CRsAE has shown to be a robust dictioanry learning tool. Not only does it show success in simulated data, but it also is successful on real world data. The dictionaries learned show meaningful and interpretable features, while still maintaining sparse encodings. Additionally, random projection has been shown to work successfully in tandem with CRsAE. 