<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
$$
\newcommand{\X}{\mathcal{X}}
\newcommand{\Y}{\mathcal{Y}}
\newcommand{\R}{\mathbb{R}}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
$$
AC299R
==========================

- [Overview](#overview)
- [Motivation](#motivation)
  * [Dictionary Learning](#dicitonary-learning)
- [Literature Review](#literature-review)
  * [Fast Iterative Shrinking-Thresholding Algorithm](#fast-iterative-shrinking-thresholding-algorithm)
  * [Constrained Recurrent Sparse Auto-Encoder](#constrained-recurrent-sparse-auto-encoder) 
  * [Random Projection](#random-projection)
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
Dictionary learning can be described as learning a linear combination of vectors to represent an input such that each of vector is a column of a matrix (the dictionary) and the coefficients are vector representations. We focus on the case where these representations are sparse. More formally stated dictionary learning can be described in the following way. Given a set of inputs $\Y = [y_1, ..., y_K], y_i \in \R^{d}$ we attempt to find a dictionary $A \in \R^{d \times n}$ and a representation or encoding $X = [x_1, ..., x_K], x_i \in \R^n$ such that the reconstruction error is minimized $||\Y - AX||$. Enforcing a sparsity constraint we can write this as the following optimization problem.

$$ \argmin_{A \in \R^{d \times n}, r_i \in \R^n} \sum_{i=1}^K ||y_i -Ax_i||_2^2 + \lambda||x_i||_0$$
Note: While the problem is written with $l_0$ norm this makes solving this problem NP-hard, so practically we use $l_1$ norm.

# Literature Review
This project primarily applies previous work from Tolooshams et al.[^1] to a setting without convolutions and with the addition of random projection as compression. This section will briefly review FISTA, the algorithm used by CRsAE to achieve sparse encodings, the previous work on CRsAE, and Random Projection.
## Fast Iterative Shrinking-Thresholding Algorithm
