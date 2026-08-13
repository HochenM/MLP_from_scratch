# MLP From Scratch

A Multilayer Perceptron (MLP) implemented from scratch using NumPy, with a focus on understanding the mathematical foundations and internal mechanics of neural networks without relying on deep learning frameworks such as TensorFlow or PyTorch.

## Overview

This project implements the core components of a feed-forward neural network manually, including:

* Forward propagation
* Loss functions
* Backpropagation
* Gradient computation
* Mini-batch training
* L2 regularization
* Adam optimizer
* Validation loss
* Early stopping

The main goal is to connect the **mathematics of neural networks** with their actual implementation in NumPy.

## Architecture

The basic architecture is:

```text
Input
  ↓
Linear Layer
  ↓
ReLU
  ↓
Linear Layer
  ↓
Output
```

The output layer depends on the task.

### Binary Classification

```text
Linear → Sigmoid → Binary Cross-Entropy
```

### Multiclass Classification

```text
Linear → Softmax → Categorical Cross-Entropy
```

### Regression

```text
Linear → Linear Output → Mean Squared Error
```

## Concepts Implemented

### Neural Network Fundamentals

* Forward propagation
* Layers and parameters
* Weights and biases
* Matrix multiplication
* ReLU activation
* Sigmoid activation
* Softmax activation
* Linear output

### Loss Functions

* Mean Squared Error (MSE)
* Binary Cross-Entropy
* Categorical Cross-Entropy

### Backpropagation

* Chain rule
* Partial derivatives
* Gradient computation
* Error propagation
* Weight gradients
* Bias gradients

### Optimization

* Gradient Descent
* Mini-batch Gradient Descent
* Data shuffling
* Adam optimizer
* First and second moment estimates
* Bias correction
* Learning-rate updates

### Regularization and Training

* L2 regularization
* Validation loss
* Early stopping
* Mini-batch training

## Mathematical Foundation

For a neural network layer:

$$
Z^{(l)} = A^{(l-1)}W^{(l)} + b^{(l)}
$$

followed by an activation function:

$$
A^{(l)} = f(Z^{(l)})
$$

Backpropagation uses the chain rule to calculate gradients and propagate the error backward through the network.

### Binary Classification

For binary classification using Sigmoid activation and Binary Cross-Entropy:

$$
dZ^{(L)} = A^{(L)} - Y
$$

### Multiclass Classification

For multiclass classification using Softmax and Categorical Cross-Entropy:

$$
dZ^{(L)} = A^{(L)} - Y
$$

### Regression

For regression using MSE and a linear output:

$$
dZ^{(L)} =
\frac{2}{m}(\hat{Y}-Y)
$$

where:

* $m$ is the number of training examples
* $\hat{Y}$ is the predicted output
* $Y$ is the true target

## Training Process

The network is trained using mini-batch gradient-based optimization:

```text
Dataset
   ↓
Shuffle
   ↓
Create Mini-batches
   ↓
Forward Propagation
   ↓
Calculate Loss
   ↓
Backpropagation
   ↓
Calculate Gradients
   ↓
Update Parameters
   ↓
Repeat
```

## Adam Optimizer

Adam is implemented from scratch using first and second moment estimates.

First moment:

$$
v_t = \beta_1 v_{t-1} + (1-\beta_1)g_t
$$

Second moment:

$$
s_t = \beta_2 s_{t-1} + (1-\beta_2)g_t^2
$$

Bias correction is then applied before updating the parameters.

The parameter update follows the general form:

$$

\theta_t =
\theta_{t-1}

------------

\alpha
\frac{\hat{v}_t}
{\sqrt{\hat{s}_t}+\epsilon}
$$

where:

* $\theta$ represents the model parameters
* $\alpha$ is the learning rate
* $\hat{v}_t$ is the bias-corrected first moment
* $\hat{s}_t$ is the bias-corrected second moment
* $\epsilon$ prevents division by zero

## Project Structure

```text
MLP-from-scratch/
│
├── classification/
│   └── mlp_classifier.py
│
├── regression/
│   └── mlp_regressor.py
│
├── notebooks/
│   └── experiments.ipynb
│
├── README.md
└── requirements.txt
```

## Requirements

Install the required dependencies with:

```bash
pip install numpy scikit-learn
```

The neural network itself is implemented using **NumPy**.

Scikit-learn is only used where needed for datasets, preprocessing, or evaluation.

## Why From Scratch?

Instead of relying on high-level APIs such as:

```python
model.fit(X, y)
```

this project manually implements the main steps of neural network training:

```text
Forward Pass
      ↓
Loss Calculation
      ↓
Backpropagation
      ↓
Gradient Calculation
      ↓
Parameter Update
```

Implementing these components from scratch helps demonstrate how the mathematical equations behind neural networks translate directly into working code.

## Learning Goals

This project was built to develop a deeper understanding of:

* How neural network layers perform matrix operations
* How activation functions transform data
* How loss functions measure prediction error
* How the chain rule enables backpropagation
* How gradients are calculated
* How optimizers update model parameters
* How regularization helps control overfitting
* How mini-batch training works
* How classification and regression networks differ

## Future Improvements

* [ ] Multiple hidden layers
* [ ] Different weight initialization methods
* [ ] Dropout
* [ ] Batch normalization
* [ ] Learning-rate schedules
* [ ] Additional optimizers
* [ ] More evaluation metrics
* [ ] Support for arbitrary network architectures
* [ ] Gradient checking
* [ ] Hyperparameter tuning
* [ ] Visualization of training curves

## Technologies

* Python
* NumPy
* Scikit-learn

## Author

**Hossein**

This project is part of my hands-on study of machine learning and deep learning fundamentals.
