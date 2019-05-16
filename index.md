<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
$$
\newcommand{\X}{\mathcal{X}}
\newcommand{\Y}{\mathcal{Y}}
\newcommand{\R}{\mathbb{R}}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
\usepackage[options ]{algorithm2e}
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
Originally proposed in Tolooshams et al. [^1], CRsAE is a recurrent auto-encoder architecture used for dictionary learning. In dictionary learning, we start knowing neither the dictionary, $$A$$, nor the encodings, $$x_i$$ that would reproduce the data. To simply the previous dual minimization problem we break the problem into a two-step update and take turns updating the encodings and then the dictionary. First we guess our dictionary and are left with $K$ convex optimization problems to find our encodings. 
$$
\argmin_{x_i} \frac{1}{2}||y_i - Ax_i||_2 + \lambda||x_i||_1
$$
In CRsAE we perform this step using the forward pass of the encoder. Unlike many current methods, the forward pass of our autoencoder is easily parallelizable using GPUs. Next we update the dictionary by using our previously estimated encodings and minimizing reconstruction error.
$$
\argmin_{A} \sum_{i=1}^K\frac{1}{2}||y_i - Ax_i||_2 \text{ s.t. } ||a_n|| \leq 1 \text{for} n=1,\dots,N
$$
We also constrain the columns of $A$ to be norm of 1 to ensure that the the energy of each reconstruction is captured in the encodings instead of the dictionary. This step is performed in CRsAE through backpropogation where the only model parameter considered is $A$. In practice this is performed using autograd.
## Model Architecture
To solve these optimization problems we use the following architecture.
!['CRsAE block'](/imgs/CRsAE_diagram.PNG) 
Starting with our data we apply FISTA using our dictionary $$A$$ to yield sparse encodings $$\hat{x}$$. We then decompress this encoding using the same dictionary $$A$$ by matrix multiplying our encoding by the dictionary. FISTA was first proposed by Beck and Teboulle [^2] and is a fast iterative procedure to find a sparse solution to the linear system in the above encoding approximation step. FISTA generates a sequence of sparse approximations using an $$l_1$$ norm penalty. The details of the algorithm are below and further reading can be seen in Beck and Teboulle [^2].

$$
\begin{algorithm}[H]
\SetAlgoLined
\KwResult{Write here the result }
 initialization\;
 \While{While condition}{
  instructions\;
  \eIf{condition}{
   instructions1\;
   instructions2\;
   }{
   instructions3\;
  }
 }
 \caption{How to write algorithms}
\end{algorithm}
$$