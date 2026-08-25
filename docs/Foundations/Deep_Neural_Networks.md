# Deep Neural Networks

## 1. Introduction

A **Deep Neural Network (DNN)** is a neural network that contains multiple layers between the input and output.

The key idea is to transform the input progressively through a hierarchy of learned representations.

A simplified DNN can be represented as:

```
Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Layer 3
  ↓
...
  ↓
Output
```

Each layer contains neurons that perform mathematical operations using:

- Weights
- Biases
- Activation functions

The term deep refers primarily to the presence of multiple computational layers.

---

## 2. From Perceptron to Deep Neural Network

The Perceptron is one of the simplest neural models:

```
Input Features
      ↓
Weighted Sum
      ↓
Activation
      ↓
Output
```

A neural network connects many such computational units:

```
Input Layer
     ↓
Hidden Layer
     ↓
Output Layer
```

When multiple hidden layers are added:

```
Input
  ↓
Hidden Layer 1
  ↓
Hidden Layer 2
  ↓
Hidden Layer 3
  ↓
Hidden Layer 4
  ↓
Output
```

we obtain a Deep Neural Network.

Conceptually:

```
Perceptron
    ↓
Multi-Layer Perceptron
    ↓
Deep Neural Network
```

---

## 3. What Makes a Network "Deep"?

A neural network is called deep when it contains multiple layers of learned transformations.

For example:

```
Input
  ↓
Linear + Activation
  ↓
Linear + Activation
  ↓
Linear + Activation
  ↓
Output
```

Each layer transforms the representation produced by the previous layer.

The important idea is not simply the number of neurons.

Instead:

> Depth refers to the number of sequential transformations through which the data passes.

---

## 4. Basic Architecture

A simple fully connected DNN can be represented as:

```
Input Layer
     ↓
Hidden Layer 1
     ↓
Hidden Layer 2
     ↓
Hidden Layer 3
     ↓
Output Layer
```

For example:

```
Input Features
     ↓
[ Neurons ]
     ↓
[ Neurons ]
     ↓
[ Neurons ]
     ↓
[ Output ]
```

Every neuron in one fully connected layer is connected to neurons in the next layer.

---

## 5. Input Layer

The input layer receives the features provided to the network.

Suppose we have:

- x₁
- x₂
- x₃
- x₄

Then the input can be represented as:

\[
\mathbf{x} =
[x_1,x_2,x_3,x_4]
\]

The input layer does not usually perform a learned transformation by itself.

Instead, it provides the data to the first computational layer.

---

## 6. Hidden Layers

Layers between the input and output are called hidden layers.

For example:

```
Input
  ↓
Hidden Layer 1
  ↓
Hidden Layer 2
  ↓
Hidden Layer 3
  ↓
Output
```

Hidden layers learn intermediate representations.

Each layer receives the output of the previous layer and transforms it.

---

## 7. Output Layer

The output layer produces the final prediction.

Its structure depends on the task.

### Binary Classification

A single output may be used:

```
Output
  ↓
Probability of Class 1
```

### Multi-Class Classification

Multiple output neurons can represent different classes:

```
Class 1
Class 2
Class 3
Class 4
```

### Regression

The network may produce a continuous numerical value:

```
Output = 42.7
```

---

## 8. Neuron Computation

A neuron performs a weighted combination of its inputs.

For a neuron:

\[
z=\sum_{i=1}^{n}w_ix_i+b
\]

Then an activation function is applied:

\[
a=f(z)
\]

Therefore:

```
Inputs
  ↓
Weighted Sum
  ↓
Bias
  ↓
Activation
  ↓
Output
```

This is the fundamental computation performed throughout a neural network.

---

## 9. Matrix Representation

Instead of calculating each neuron separately, neural networks usually represent the computation using matrices.

For a layer:

\[
\mathbf{z}=\mathbf{W}\mathbf{x}+\mathbf{b}
\]

Then:

\[
\mathbf{a}=f(\mathbf{z})
\]

where:

- \(\mathbf{x}\) = input vector
- \(\mathbf{W}\) = weight matrix
- \(\mathbf{b}\) = bias vector
- \(\mathbf{z}\) = pre-activation
- \(\mathbf{a}\) = activation

---

## 10. Example of a Fully Connected Layer

Suppose the input contains 4 features:

- x₁
- x₂
- x₃
- x₄

and the next layer contains 3 neurons.

The weight matrix has dimensions:

\[
3\times4
\]

because:

```
4 input features
        ↓
3 neurons
```

The computation is:

\[
\mathbf{z}=\mathbf{W}\mathbf{x}+\mathbf{b}
\]

producing:

- z₁
- z₂
- z₃

After activation:

- a₁
- a₂
- a₃

---

## 11. Why Do We Need Activation Functions?

If every layer only performed a linear transformation:

\[
\mathbf{z}=\mathbf{W}\mathbf{x}+\mathbf{b}
\]

then stacking many layers would still result in an overall linear transformation.

For example:

\[
W_2(W_1x+b_1)+b_2
\]

can be simplified into another linear transformation.

Therefore, simply adding more linear layers would not provide the expressive power expected from a deep neural network.

Activation functions introduce nonlinearity.

---

## 12. Nonlinearity

A simplified DNN layer is:

\[
\mathbf{a}=f(\mathbf{W}\mathbf{x}+\mathbf{b})
\]

The function \(f\) introduces nonlinearity.

Common activation functions include:

- ReLU
- Sigmoid
- Tanh
- GELU

For example, ReLU is:

\[
ReLU(x)=\max(0,x)
\]

Therefore:

- Negative value → 0
- Positive value → unchanged

---

## 13. Hierarchical Representation Learning

One of the most important properties of deep networks is that different layers can learn different levels of representation.

Conceptually:

```
Raw Input
   ↓
Low-Level Features
   ↓
Intermediate Features
   ↓
High-Level Features
   ↓
Task-Specific Representation
   ↓
Prediction
```

For images, this might look like:

```
Pixels
  ↓
Edges
  ↓
Textures
  ↓
Shapes
  ↓
Objects
  ↓
Class
```

This idea is called hierarchical representation learning.

---

## 14. What Does "Representation" Mean?

A representation is the numerical form in which the network internally describes the input.

Suppose the original input is:

Image

After passing through several layers, the network transforms it into:

Feature Representation

For example:

```
Image
 ↓
Layer 1
 ↓
[0.2, 0.7, 0.1, ...]
 ↓
Layer 2
 ↓
[1.2, 0.0, 0.4, ...]
 ↓
Layer 3
 ↓
[0.8, 2.1, 0.3, ...]
```

Each layer creates a different representation.

---

## 15. Representation Learning

Traditional machine learning often uses:

```
Raw Data
 ↓
Manual Feature Engineering
 ↓
Feature Vector
 ↓
Machine Learning Model
 ↓
Prediction
```

Deep learning attempts to learn useful representations automatically:

```
Raw Data
 ↓
Neural Network
 ↓
Learned Features
 ↓
Prediction
```

This is one of the major differences between traditional feature engineering and deep representation learning.

---

## 16. Traditional Feature Engineering vs DNN

| Aspect                         | Traditional Approach                       | Deep Learning Approach                       |
|--------------------------------|-------------------------------------------|---------------------------------------------|
| Feature extraction             | Often manual                              | Learned                                      |
| Representation                 | Usually predefined                        | Learned from data                           |
| Depth                          | Usually limited                          | Multiple layers                              |
| Parameters                     | Often fewer                              | Potentially millions/billions               |
| Data requirements              | Often lower                              | Often higher                                |
| Computation                    | Usually lower                            | Often higher                                |
| Feature engineering            | Important                                 | Reduced, but not eliminated                 |
| Hierarchical features          | Limited                                   | Strong capability                           |

---

## 17. Forward Propagation

During forward propagation, data moves from the input toward the output.

For a simple network:

```
Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Layer 3
  ↓
Output
```

Each layer performs:

\[
\mathbf{a}^{(l)}
=
f(
\mathbf{W}^{(l)}
\mathbf{a}^{(l-1)}
+
\mathbf{b}^{(l)}
)
\]

where \(l\) represents the current layer.

---

## 18. Example of Forward Propagation

Suppose:

```
Input
 ↓
Layer 1
 ↓
Layer 2
 ↓
Output
```

The first layer computes:

\[
z^{(1)}=W^{(1)}x+b^{(1)}
\]

Then:

\[
a^{(1)}=f(z^{(1)})
\]

The second layer receives \(a^{(1)}\):

\[
z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}
\]

Then:

\[
a^{(2)}=f(z^{(2)})
\]

Finally:

\[
\hat{y}=g(z^{(3)})
\]

where \(\hat{y}\) is the prediction.

---

## 19. Prediction

The network's output is represented as:

\[
\hat{y}
\]

The symbol \(\hat{y}\) means the predicted output.

The model compares this prediction with the true label:

\[
y
\]

For example:

- True Label      = 1
- Prediction      = 0.82

The difference between prediction and target is measured using a loss function.

---

## 20. Loss Function

A neural network needs a way to measure how well it is performing.

This is the role of the loss function.

Conceptually:

```
Prediction
    ↓
Compare with True Label
    ↓
Loss
```

For classification, common losses include:

- Cross-Entropy Loss
- Binary Cross-Entropy

For regression:

- Mean Squared Error
- Mean Absolute Error

The goal of training is generally to minimize the loss.

---

## 21. Backpropagation

After calculating the loss, the network needs to determine how its parameters should change.

This is done using backpropagation.

Conceptually:

```
Forward Propagation
        ↓
Prediction
        ↓
Loss
        ↓
Backpropagation
        ↓
Gradients
        ↓
Parameter Updates
```

Backpropagation uses the chain rule to calculate how the loss changes with respect to the network's parameters.

---

## 22. Gradients

A gradient tells us how a small change in a parameter affects the loss.

For a weight \(w\):

\[
\frac{\partial L}{\partial w}
\]

means:

> How does the loss \(L\) change when the weight \(w\) changes?

The optimizer uses this information to update the weights.

---

## 23. Gradient Descent

A basic gradient descent update is:

\[
w \leftarrow w-\eta\frac{\partial L}{\partial w}
\]

where:

- \(w\) = parameter
- \(L\) = loss
- \(\eta\) = learning rate

The negative sign means that the parameter is updated in the direction that tends to reduce the loss.

---

## 24. Complete Training Process

A DNN training loop can be represented as:

```
Training Data
      ↓
Forward Propagation
      ↓
Prediction
      ↓
Loss Calculation
      ↓
Backpropagation
      ↓
Gradient Calculation
      ↓
Optimizer
      ↓
Weight Update
      ↓
Next Training Step
```

This process is repeated many times.

---

## 25. Epochs

An epoch represents one complete pass through the training dataset.

For example:

```
Epoch 1
 ↓
Entire Training Dataset

Epoch 2
 ↓
Entire Training Dataset

Epoch 3
 ↓
Entire Training Dataset
```

Training a model for 20 epochs means the training process has passed through the dataset approximately 20 times.

---

## 26. Batch Training

Large datasets are usually divided into smaller groups called batches.

For example:

```
Dataset = 10,000 samples

Batch 1 → 32 samples
Batch 2 → 32 samples
Batch 3 → 32 samples
...
```

The model performs an update after processing a batch.

The number of samples in each batch is called the:

> Batch Size

---

## 27. Parameters

The weights and biases of a neural network are called parameters.

For example:

```
Layer 1
 ↓
Weights + Biases

Layer 2
 ↓
Weights + Biases

Layer 3
 ↓
Weights + Biases
```

Training means learning appropriate values for these parameters.

---

## 28. Hyperparameters

Hyperparameters are configuration values chosen by the researcher or practitioner rather than learned directly from the data.

Examples include:

- Learning rate
- Batch size
- Number of layers
- Number of neurons
- Number of epochs
- Weight decay
- Dropout rate

For example:

- Learning Rate = 0.001
- Batch Size    = 32
- Epochs        = 20

---

## 29. Number of Parameters

The number of parameters in a fully connected layer depends on:

- Input Features × Output Neurons

plus the biases.

For example, if a layer has:

- 100 input features
- 50 neurons

then the number of weights is:

\[
100\times50=5000
\]

and there are 50 biases.

Therefore:

\[
5000+50=5050
\]

parameters.

---

## 30. Overfitting

A DNN can have a very large number of parameters.

If the model becomes too specialized to the training data, it may perform well on training samples but poorly on unseen data.

This is called overfitting.

Conceptually:

```
Training Performance
        ↑
        │      ●
        │    ●
        │  ●
        └────────────→

Validation Performance
        ↑
        │   ●
        │  ●
        │ ●
        └────────────→
```

The model needs to learn patterns that generalize beyond the training data.

---

## 31. Regularization

Several techniques can reduce overfitting.

Common approaches include:

- Dropout
- Weight decay
- Data augmentation
- Early stopping
- Regularization penalties

For example:

```
DNN
 ↓
Dropout
 ↓
Reduced dependence on individual neurons
```

---

## 32. Dropout

Dropout randomly disables some neurons during training.

Conceptually:

Before Dropout

```
● ─ ● ─ ● ─ ●
│   │   │   │
● ─ ● ─ ● ─ ●
```

After Dropout

```
● ─ ✕ ─ ● ─ ●
│       │
✕ ─ ● ─ ✕ ─ ●
```

This encourages the network to learn more robust representations.

---

## 33. Weight Decay

Weight decay penalizes excessively large weights.

A simplified objective can be written as:

\[
L_{total}
=
L_{data}
+
\lambda\|\mathbf{W}\|^2
\]

where:

- \(L_{data}\) = task loss
- \(\lambda\) = regularization strength
- \(\|\mathbf{W}\|^2\) = weight magnitude penalty

This encourages the model to use more controlled parameter values.

---

## 34. Vanishing Gradients

Deep networks can encounter a problem called the vanishing gradient problem.

During backpropagation, gradients are propagated through many layers.

If gradients become very small:

```
Gradient
 ↓
0.1
 ↓
0.01
 ↓
0.001
 ↓
0.0001
 ↓
...
```

early layers may receive extremely small updates.

As a result:

```
Early Layers
 ↓
Very Small Updates
 ↓
Slow Learning
```

This can make very deep networks difficult to train.

---

## 35. Exploding Gradients

The opposite problem is the exploding gradient problem.

Gradients can become very large:

```
0.1
 ↓
1
 ↓
10
 ↓
100
 ↓
1000
```

This can lead to:

- Unstable training
- Very large parameter updates
- Numerical instability

Gradient clipping and appropriate initialization/normalization methods can help in some settings.

---

## 36. Why Depth Can Become Difficult

Increasing depth provides the potential to learn more complex representations.

However, deeper networks can also introduce optimization difficulties:

```
More Layers
     ↓
Longer Gradient Path
     ↓
More Difficult Optimization
     ↓
Vanishing / Exploding Gradients
```

Another issue is the degradation problem, where simply adding layers can make optimization harder rather than automatically improving performance.

These problems motivated architectures such as ResNet.

---

## 37. Deep Neural Networks and CNNs

A standard DNN often uses fully connected layers:

```
Input
 ↓
Fully Connected
 ↓
Fully Connected
 ↓
Fully Connected
 ↓
Output
```

A CNN is designed specifically for structured spatial data such as images:

```
Image
 ↓
Convolution
 ↓
Feature Maps
 ↓
Pooling
 ↓
More Convolutions
 ↓
Feature Representation
 ↓
Classifier
```

Therefore, CNNs can be considered specialized deep neural network architectures.

---

## 38. CNNs Learn Spatial Representations

A fully connected network treats its inputs more generically.

CNNs exploit spatial structure.

For images:

```
Pixels
 ↓
Edges
 ↓
Textures
 ↓
Shapes
 ↓
Object Parts
 ↓
Objects
```

Convolutional layers make this hierarchical visual representation possible.

---

## 39. DNNs and Representation Learning

One of the most important ideas in modern deep learning is:

> The network can learn useful representations directly from data.

Traditional pipeline:

```
Raw Data
 ↓
Human-Designed Features
 ↓
Classifier
```

Deep learning pipeline:

```
Raw Data
 ↓
Learned Representations
 ↓
Classifier
```

This is known as representation learning.

---

## 40. Feature Extraction in Deep Networks

Consider an image entering a deep network.

Different layers may learn different types of information:

```
Input Image
     ↓
Layer 1
     ↓
Edges / Simple Patterns
     ↓
Layer 2
     ↓
Textures / Local Structures
     ↓
Layer 3
     ↓
Shapes / Parts
     ↓
Deeper Layers
     ↓
Semantic Representation
```

The exact interpretation of individual neurons is not always this clean, but the hierarchy provides a useful conceptual model.

---

## 41. Deep Representations

Suppose a network transforms an image into a vector:

\[
\mathbf{h}\in\mathbb{R}^{2048}
\]

This means the image is represented by 2048 numerical values.

Conceptually:

```
Image
 ↓
Deep Neural Network
 ↓
[ h₁, h₂, h₃, ..., h₂₀₄₈ ]
```

This vector can be viewed as the learned representation of the image at that layer.

---

## 42. Feature Space

If every input is represented by a vector, the representations can be viewed as points in a feature space.

For example:

```
Image A → [0.2, 0.8, 0.1, ...]
Image B → [0.3, 0.7, 0.2, ...]
Image C → [2.1, 0.1, 1.4, ...]
```

Each vector corresponds to a point in a high-dimensional space.

This concept becomes especially important when analyzing embeddings using techniques such as:

- PCA
- t-SNE
- UMAP

---

## 43. DNNs as Function Approximators

A neural network can be viewed mathematically as a parameterized function:

\[
\hat{y}=f(\mathbf{x};\theta)
\]

where:

- \(\mathbf{x}\) = input
- \(\hat{y}\) = prediction
- \(\theta\) = all learnable parameters

Training attempts to find parameters:

\[
\theta^*
\]

that minimize the loss.

Conceptually:

```
Input
 ↓
Parameterized Function
 ↓
Prediction
 ↓
Loss
 ↓
Optimization
```

---

## 44. Universal Approximation

Neural networks with appropriate architectures and nonlinear activation functions have powerful function approximation capabilities.

The Universal Approximation Theorem states, in simplified terms, that certain neural networks with at least one hidden layer can approximate a broad class of continuous functions to arbitrary accuracy under suitable conditions.

However:

> This theorem does not mean that one hidden layer is always practical or that training such a network is easy.

Deep architectures can provide more efficient hierarchical representations for many problems.

---

## 45. Deep Learning Workflow

A typical DNN project can be organized as:

```
Dataset
   ↓
Preprocessing
   ↓
Train / Validation / Test Split
   ↓
Model Definition
   ↓
Forward Propagation
   ↓
Loss Calculation
   ↓
Backpropagation
   ↓
Optimization
   ↓
Validation
   ↓
Testing
   ↓
Evaluation
```

Common evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

depending on the task.

---

## 46. DNN vs Traditional Machine Learning

| Aspect                           | Traditional ML                            | Deep Neural Network                       |
|----------------------------------|------------------------------------------|-------------------------------------------|
| Feature extraction               | Often manual                             | Learned                                    |
| Representation                   | Usually predefined                       | Learned from data                         |
| Depth                            | Usually limited                         | Multiple layers                           |
| Parameters                       | Often fewer                             | Potentially millions/billions             |
| Data requirements                | Often lower                             | Often higher                              |
| Computation                      | Usually lower                           | Often higher                              |
| Feature engineering              | Important                               | Reduced, but not eliminated               |
| Hierarchical features            | Limited                                 | Strong capability                         |

---

## 47. DNN vs Perceptron

| Aspect                         | Perceptron                              | Deep Neural Network                       |
|--------------------------------|----------------------------------------|-------------------------------------------|
| Layers                         | Single computational unit               | Multiple layers                          |
| Decision boundary              | Linear                                 | Can be nonlinear                          |
| Representation learning        | Very limited                          | Strong                                    |
| Activation                     | Classical step function                | ReLU, Sigmoid, GELU, etc.                |
| Training                       | Perceptron learning rule               | Backpropagation + optimizer              |
| Complexity                     | Very low                               | Potentially very high                     |

---

## 48. From DNN to ResNet

Increasing depth can improve representation capacity, but very deep networks can become difficult to optimize.

The conceptual progression is:

```
Deep Neural Network
        ↓
More Layers
        ↓
Optimization Difficulties
        ↓
Vanishing / Exploding Gradients
        ↓
Degradation Problem
        ↓
Residual Learning
        ↓
ResNet
```

ResNet addresses some of these difficulties using skip connections.

---

## 49. Residual Learning

Instead of forcing a block to directly learn a desired mapping:

\[
H(x)
\]

ResNet reformulates the problem as learning a residual:

\[
F(x)=H(x)-x
\]

Then:

\[
H(x)=F(x)+x
\]

This leads to the residual block:

```
┌───────────────┐
          │               │
Input ────┤   Residual    │
  │       │   Function    │
  │       └───────┬───────┘
  │               │
  └──────────────→ Add
                   ↓
                Output
```

This idea is the foundation of ResNet.

---

## 50. Why DNNs Are Important for ResNet

ResNet is not an isolated concept.

It addresses challenges that arise when building increasingly deep neural networks.

The progression is:

```
Perceptron
    ↓
Neural Network
    ↓
Deep Neural Network
    ↓
Very Deep Network
    ↓
Optimization Problems
    ↓
Residual Learning
    ↓
ResNet
```

Understanding DNNs therefore provides the foundation for understanding why ResNet was introduced.

---

## 51. Key Takeaways

The most important concepts are:

1. A DNN contains multiple layers of learned transformations.
2. Each neuron computes a weighted sum plus a bias.
3. Activation functions introduce nonlinearity.
4. Multiple layers allow hierarchical representations.
5. DNNs can learn features directly from data.
6. Forward propagation produces predictions.
7. A loss function measures prediction error.
8. Backpropagation computes gradients.
9. Optimizers update model parameters.
10. Deep networks can suffer from vanishing and exploding gradients.
11. Increasing depth can introduce optimization difficulties.
12. CNNs are specialized architectures for spatial data such as images.
13. ResNet uses residual learning and skip connections to make very deep networks easier to optimize.

---

## 52. Final Conceptual Diagram

### Deep Neural Network

```
                         Input
                           │
                           ↓
                  ┌─────────────────┐
                  │   Hidden Layer 1│
                  │ W₁x + b₁        │
                  │ Activation      │
                  └────────┬────────┘
                           │
                           ↓
                  ┌─────────────────┐
                  │   Hidden Layer 2│
                  │ W₂x + b₂        │
                  │ Activation      │
                  └────────┬────────┘
                           │
                           ↓
                  ┌─────────────────┐
                  │   Hidden Layer 3│
                  │ W₃x + b₃        │
                  │ Activation      │
                  └────────┬────────┘
                           │
                           ↓
                        Output
                           │
                           ↓
                       Prediction
```

The training process is:

```
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss
  ↓
Backpropagation
  ↓
Gradients
  ↓
Optimizer
  ↓
Updated Parameters
  ↓
Repeat
```

The central idea is:

> A Deep Neural Network learns a hierarchy of representations by passing data through multiple parameterized nonlinear transformations and adjusting its parameters using the error produced by its predictions.

