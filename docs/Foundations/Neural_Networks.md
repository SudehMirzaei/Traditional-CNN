# Neural Networks

## 1. Introduction

A **Neural Network** is a computational model inspired by the basic idea of biological neurons.

Neural networks learn a mapping between inputs and outputs by adjusting a set of learnable parameters called **weights** and **biases**.

A simplified neural network can be represented as:

```
Input
  ↓
Weighted Computation
  ↓
Activation
  ↓
Output
```

When many neurons are connected together in layers, they form a neural network.

---

## 2. From Perceptron to Neural Network

The Perceptron is one of the simplest neural computational units.

A single Perceptron can be represented as:

```
x₁ ──→ w₁ ──┐
x₂ ──→ w₂ ──┤
x₃ ──→ w₃ ──┤
             ↓
       Weighted Sum
             ↓
           + Bias
             ↓
        Activation
             ↓
           Output
```

Mathematically:

\[
z=\sum_{i=1}^{n}w_ix_i+b
\]

and:

\[
y=f(z)
\]

A neural network connects multiple such computational units together.

---

## 3. Why Connect Multiple Neurons?

A single neuron has limited representational power.

For example, a single Perceptron can learn a linear decision boundary.

However, many real-world problems are nonlinear.

By connecting multiple neurons and introducing nonlinear activation functions, a neural network can learn more complex relationships.

Conceptually:

```
Single Neuron

Input
  ↓
Output
```

becomes:

```
Multiple Neurons

Input
  ↓
Neurons
  ↓
Neurons
  ↓
Output
```

This allows the network to learn increasingly complex functions.

---

## 4. Basic Neural Network Structure

A simple neural network consists of:

1. Input Layer
2. Hidden Layer(s)
3. Output Layer

```
Input Layer
     ↓
Hidden Layer
     ↓
Output Layer
```

For a larger network:

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

---

## 5. Input Layer

The input layer receives the features of the input data.

Suppose an example contains three features:

- x₁
- x₂
- x₃

The input vector is:

\[
\mathbf{x}=
[x_1,x_2,x_3]
\]

For a tabular dataset, these could represent numerical features.

For an image, the inputs may represent pixel values.

For example, a grayscale image of size:

\[
28\times28
\]

contains:

\[
28\times28=784
\]

pixel values.

These values can be provided to a neural network as input features.

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
Output
```

Hidden layers transform the input representation.

Each layer receives the output of the previous layer and produces a new representation.

---

## 7. Output Layer

The output layer produces the final result of the network.

Its structure depends on the task.

### Binary Classification

A network may produce one output:

```
Input
 ↓
Neural Network
 ↓
Probability
```

For example:

Tumor / No Tumor

### Multi-Class Classification

The output layer may contain one neuron per class:

```
Class 1
Class 2
Class 3
Class 4
```

### Regression

The network may produce a continuous value:

```
Input
 ↓
Neural Network
 ↓
42.7
```

---

## 8. Neurons

A neuron is a basic computational unit of a neural network.

It receives inputs and calculates a weighted sum.

For example:

```
x₁ ──→ w₁ ──┐
x₂ ──→ w₂ ──┤
x₃ ──→ w₃ ──┤
             ↓
      Weighted Sum
             ↓
            + b
             ↓
        Activation
             ↓
          Output
```

The mathematical computation is:

\[
z=w_1x_1+w_2x_2+w_3x_3+b
\]

Then:

\[
a=f(z)
\]

where:

- \(x_i\) = input
- \(w_i\) = weight
- \(b\) = bias
- \(z\) = pre-activation
- \(f\) = activation function
- \(a\) = activation/output

---

## 9. Weights

Weights determine how strongly each input influences a neuron.

For example:

\[
z=2x_1+0.5x_2-1x_3+b
\]

The weights are:

- \(x₁ →  2\)
- \(x₂ →  0.5\)
- \(x₃ → -1\)

A large positive weight means that increasing the corresponding input tends to increase the neuron's weighted sum.

A negative weight means that increasing the input tends to decrease the weighted sum.

Weights are learnable parameters.

---

## 10. Bias

The bias is an additional learnable parameter.

The complete equation is:

\[
z=\mathbf{w}^T\mathbf{x}+b
\]

The bias allows the neuron to shift its activation.

Without a bias, the decision boundary would be more constrained.

With a bias, the model has greater flexibility.

---

## 11. Activation Functions

After calculating the weighted sum, the neuron applies an activation function.

The general form is:

\[
a=f(z)
\]

Activation functions introduce nonlinearity into the network.

Common activation functions include:

- ReLU
- Sigmoid
- Tanh
- GELU
- Softmax

---

## 12. ReLU

The Rectified Linear Unit (ReLU) is one of the most commonly used activation functions in neural networks.

It is defined as:

\[
ReLU(x)=\max(0,x)
\]

Therefore:

- Negative → 0
- Positive → unchanged

For example:

\[
ReLU(-3)=0
\]

\[
ReLU(5)=5
\]

ReLU is widely used in hidden layers of modern neural networks.

---

## 13. Sigmoid

The sigmoid function is:

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

Its output is between 0 and 1:

\[
0<\sigma(x)<1
\]

This makes it useful in some binary classification settings.

For example:

Output = 0.91

can be interpreted as a high predicted probability for the positive class when the setup uses sigmoid probabilities.

---

## 14. Tanh

The hyperbolic tangent function is:

\[
tanh(x)
\]

Its output range is:

\[
-1<tanh(x)<1
\]

It was historically common in neural networks and remains useful in certain architectures and applications.

---

## 15. Softmax

For multi-class classification, the output layer often uses Softmax.

For class (i):

\[
P(y=i)=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
\]

Softmax converts logits into values that sum to 1.

For example:

- Class A → 0.10
- Class B → 0.70
- Class C → 0.20

The predicted class is usually the class with the highest probability.

---

## 16. Why Nonlinearity Is Important

Suppose a neural network contains only linear transformations:

\[
z=W x+b
\]

Even if many such layers are stacked together, the entire network can still be represented as a single linear transformation.

For example:

\[
W_2(W_1x+b_1)+b_2
\]

can be rewritten as another linear function of \(x\).

Therefore, simply adding more linear layers does not provide the desired nonlinear modeling capability.

Activation functions solve this problem.

```
Linear Transformation
        ↓
Activation
        ↓
Linear Transformation
        ↓
Activation
```

This allows the network to model nonlinear relationships.

---

## 17. Fully Connected Layer

A Fully Connected (FC) layer connects every neuron in one layer to every neuron in the next layer.

For example:

```
Input Layer          Hidden Layer

● ────────────────→ ●
● ────────────────→ ●
● ────────────────→ ●
● ────────────────→ ●
```

Every input neuron is connected to every neuron in the next layer.

The computation can be written as:

\[
\mathbf{z}=\mathbf{W}\mathbf{x}+\mathbf{b}
\]

followed by:

\[
\mathbf{a}=f(\mathbf{z})
\]

---

## 18. Matrix Representation

Instead of computing every neuron separately, neural networks perform the calculations using matrix operations.

Suppose:

- Input Size = 4
- Output Neurons = 3

Then:

\[
\mathbf{x}\in\mathbb{R}^{4}
\]

and:

\[
\mathbf{W}\in\mathbb{R}^{3\times4}
\]

The output is:

\[
\mathbf{z}=\mathbf{W}\mathbf{x}+\mathbf{b}
\]

where:

\[
\mathbf{b}\in\mathbb{R}^{3}
\]

The result contains three values, one for each output neuron.

---

## 19. Layers as Transformations

A neural network can be viewed as a sequence of transformations.

For layer \(l\):

\[
\mathbf{z}^{(l)}
=
\mathbf{W}^{(l)}
\mathbf{a}^{(l-1)}
+
\mathbf{b}^{(l)}
\]

Then:

\[
\mathbf{a}^{(l)}
=
f(\mathbf{z}^{(l)})
\]

The output of one layer becomes the input of the next:

```
Input
  ↓
Transformation 1
  ↓
Representation 1
  ↓
Transformation 2
  ↓
Representation 2
  ↓
Transformation 3
  ↓
Output
```

---

## 20. Forward Propagation

Forward propagation is the process of passing data through the network from input to output.

For example:

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

At each layer:

\[
a^{(l)}
=
f(W^{(l)}a^{(l-1)}+b^{(l)})
\]

The final activation becomes the network's prediction.

---

## 21. Example of Forward Propagation

Consider:

```
Input
 ↓
Hidden Layer
 ↓
Output
```

First:

\[
z^{(1)}=W^{(1)}x+b^{(1)}
\]

Then:

\[
a^{(1)}=f(z^{(1)})
\]

The output layer receives \(a^{(1)}\):

\[
z^{(2)}=W^{(2)}a^{(1)}+b^{(2)}
\]

Finally:

\[
\hat{y}=g(z^{(2)})
\]

where \(\hat{y}\) is the prediction.

---

## 22. Prediction

The output produced by the network is called the prediction.

It is commonly represented as:

\[
\hat{y}
\]

The true target is:

\[
y
\]

For example:

- True Label = 1
- Prediction = 0.87

The network then uses a loss function to measure how well the prediction matches the target.

---

## 23. Loss Function

A loss function measures the error of the model.

Conceptually:

```
Prediction
    ↓
Compare with Target
    ↓
Loss
```

For classification, common loss functions include:

- Cross-Entropy Loss
- Binary Cross-Entropy

For regression:

- Mean Squared Error
- Mean Absolute Error

The training objective is generally to minimize the loss.

---

## 24. Backpropagation

After calculating the loss, the network needs to determine how its parameters contributed to that error.

This is done through backpropagation.

The process is:

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

Backpropagation applies the chain rule to compute gradients.

---

## 25. Gradients

A gradient measures how a change in a parameter affects the loss.

For a weight \(w\):

\[
\frac{\partial L}{\partial w}
\]

means:

> How much does the loss change when this weight changes?

The gradient provides the direction used by the optimizer to update the parameter.

---

## 26. Gradient Descent

A simple gradient descent update is:

\[
w\leftarrow
w-\eta\frac{\partial L}{\partial w}
\]

where:

- \(w\) = weight
- \(L\) = loss
- \(\eta\) = learning rate

The learning rate controls the size of the update.

---

## 27. Optimizers

An optimizer determines how the network parameters are updated using gradients.

Common optimizers include:

- SGD
- Adam
- AdamW
- RMSprop

A simplified training process is:

```
Loss
 ↓
Gradients
 ↓
Optimizer
 ↓
Updated Weights
```

---

## 28. Training a Neural Network

The complete training process can be summarized as:

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
Update Weights and Biases
      ↓
Repeat
```

This process is repeated over many training examples and epochs.

---

## 29. Epoch

An epoch is one complete pass through the training dataset.

For example:

```
Epoch 1
 ↓
All Training Samples

Epoch 2
 ↓
All Training Samples

Epoch 3
 ↓
All Training Samples
```

Training for 20 epochs means the model has gone through the training dataset approximately 20 times.

---

## 30. Batch Size

Instead of processing the entire dataset at once, neural networks usually process data in smaller groups called batches.

For example:

- Dataset = 10,000 samples
- Batch 1 → 32 samples
- Batch 2 → 32 samples
- Batch 3 → 32 samples
- ...

The number of samples in each batch is called the batch size.

---

## 31. Parameters vs Hyperparameters

### Parameters

Parameters are learned automatically during training.

Examples:

- Weights
- Biases

### Hyperparameters

Hyperparameters are chosen before or during training.

Examples:

- Learning rate
- Batch size
- Number of epochs
- Number of layers
- Number of neurons
- Dropout rate
- Weight decay

---

## 32. Representation Learning

One of the most important ideas in neural networks is representation learning.

Traditional machine learning often follows:

```
Raw Data
   ↓
Manual Feature Engineering
   ↓
Feature Vector
   ↓
Classifier
```

Neural networks can learn representations directly:

```
Raw Data
   ↓
Neural Network
   ↓
Learned Representation
   ↓
Prediction
```

This reduces the need to manually define every useful feature.

---

## 33. Hierarchical Representations

When multiple layers are used, the network can learn a hierarchy of representations.

For images, a simplified conceptual progression is:

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
High-Level Representation
  ↓
Classification
```

The exact features learned by individual neurons are more complex than this simplified diagram, but the hierarchy is useful for understanding deep learning.

---

## 34. Feature Engineering vs Representation Learning

### Traditional Machine Learning

```
Image
 ↓
Manual Feature Extraction
 ↓
HOG / SIFT / Texture Features
 ↓
Feature Vector
 ↓
SVM
 ↓
Prediction
```

### Neural Network

```
Image
 ↓
Neural Network
 ↓
Learned Features
 ↓
Representation
 ↓
Prediction
```

The major difference is who defines the representation.

In manual feature engineering:

> The human designs the features.

In representation learning:

> The model learns useful representations from data.

---

## 35. Neural Networks and Images

A basic fully connected neural network can process an image by flattening it into a vector.

For example:

```
28 × 28 Image
      ↓
Flatten
      ↓
784-dimensional Vector
      ↓
Fully Connected Layer
```

However, flattening removes explicit spatial structure.

CNNs were designed to address this limitation.

---

## 36. Neural Networks vs CNNs

A standard fully connected network:

```
Image
 ↓
Flatten
 ↓
Fully Connected Layers
 ↓
Output
```

A CNN:

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
Learned Representation
 ↓
Fully Connected / Classification Head
 ↓
Output
```

CNNs exploit local spatial relationships and parameter sharing.

---

## 37. Neural Networks and Deep Neural Networks

A neural network may contain one or several hidden layers.

A Deep Neural Network (DNN) contains multiple layers.

Conceptually:

```
Neural Network
      ↓
Multiple Hidden Layers
      ↓
Deep Neural Network
```

As depth increases, the network can potentially learn increasingly complex representations.

However, deeper networks also introduce optimization challenges.

---

## 38. Vanishing Gradient Problem

When a network becomes very deep, gradients may become extremely small as they propagate backward.

Conceptually:

```
Output
  ↓
Gradient = 1.0
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

Early layers may receive very small updates.

This can make them difficult to train.

This problem is called the:

> Vanishing Gradient Problem

---

## 39. Exploding Gradient Problem

The opposite problem is the Exploding Gradient Problem.

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

This can cause unstable parameter updates.

---

## 40. Degradation Problem

Adding more layers does not always automatically improve a network.

Very deep networks can become increasingly difficult to optimize.

This can lead to the degradation problem, where a deeper network may have higher training error than a shallower network.

This was one of the motivations behind the development of ResNet.

---

## 41. Residual Learning

ResNet introduced Residual Learning to make very deep networks easier to optimize.

Instead of directly learning:

\[
H(x)
\]

a residual block learns:

\[
F(x)=H(x)-x
\]

Therefore:

\[
H(x)=F(x)+x
\]

The input can bypass some layers through a skip connection.

```
┌───────────────┐
             │               │
Input ───────┤ Residual Path │
  │          │               │
  │          └───────┬───────┘
  │                  │
  └────────────────→ Add
                       ↓
                    Output
```

---

## 42. From Neural Networks to ResNet

The conceptual progression is:

```
Perceptron
    ↓
Neural Network
    ↓
Deep Neural Network
    ↓
CNN
    ↓
Very Deep CNN
    ↓
Optimization Problems
    ↓
Residual Learning
    ↓
ResNet
```

ResNet builds upon the fundamental concepts of neural networks while introducing residual connections to facilitate very deep architectures.

---

## 43. Key Components of a Neural Network

The fundamental components are:

- **Inputs**: The data provided to the model.
- **Weights**: Learnable parameters controlling feature importance.
- **Bias**: Learnable parameter that shifts the transformation.
- **Linear Transformation**: 

\[
z=Wx+b
\]

- **Activation Function**: Introduces nonlinearity.
- **Layers**: Organize neurons into sequential transformations.
- **Loss Function**: Measures prediction error.
- **Backpropagation**: Computes gradients.
- **Optimizer**: Updates parameters.

---

## 44. Advantages of Neural Networks

Neural networks provide several important advantages:

- Can model nonlinear relationships
- Can learn representations from data
- Can handle high-dimensional inputs
- Can approximate complex functions
- Can be adapted to many tasks
- Form the foundation of deep learning
- Can be extended into specialized architectures such as CNNs, RNNs, and Transformers

---

## 45. Limitations

Neural networks also have limitations:

- Often require substantial training data
- Can require significant computational resources
- Can overfit
- Training can be sensitive to hyperparameters
- Deep networks can suffer from optimization difficulties
- Their internal representations can be difficult to interpret
- Performance depends strongly on architecture and training strategy

---

## 46. Neural Networks in Computer Vision

For image-related tasks, neural networks have evolved through several stages:

```
Manual Features
      ↓
HOG / SIFT
      ↓
Traditional Classifiers
      ↓
CNNs
      ↓
Deep CNNs
      ↓
ResNet / DenseNet / EfficientNet
      ↓
Vision Transformers
```

This progression reflects the increasing ability of models to learn useful representations directly from raw images.

---

## 47. Example: Image Classification

Consider a brain MRI classification task.

A traditional pipeline might be:

```
MRI
 ↓
Manual Feature Extraction
 ↓
Texture / Shape Features
 ↓
Classifier
 ↓
Prediction
```

A neural network approach can instead learn representations automatically:

```
MRI
 ↓
Neural Network
 ↓
Learned Representation
 ↓
Classification Head
 ↓
Prediction
```

A CNN-based architecture such as ResNet can further exploit the spatial structure of the MRI image.

---

## 48. Neural Networks as Compositions of Functions

A neural network can mathematically be viewed as a composition of functions.

For example:

\[
f(x)=f_3(f_2(f_1(x)))
\]

where:

- \(f_1\) is the first layer,
- \(f_2\) is the second layer, and
- \(f_3\) is the third layer.

Conceptually:

```
Input
 ↓
f₁
 ↓
Representation 1
 ↓
f₂
 ↓
Representation 2
 ↓
f₃
 ↓
Output
```

This mathematical view becomes especially useful when studying deep learning architectures.

---

## 49. Final Conceptual Diagram

### Neural Network

```
Input Features
      │
      ↓
┌──────────────────┐
│   Hidden Layer 1 │
│   W₁x + b₁       │
│   Activation     │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│   Hidden Layer 2 │
│   W₂x + b₂       │
│   Activation     │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│   Hidden Layer 3 │
│   W₃x + b₃       │
│   Activation     │
└────────┬─────────┘
         │
         ↓
┌──────────────────┐
│   Output Layer   │
└────────┬─────────┘
         │
         ↓
     Prediction
```

---

## 50. Training Concept

The complete learning process can be summarized as:

```
Training

Input Data
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
Update Parameters
    ↓
Repeat
```

---

## 51. Summary

A Neural Network is a collection of interconnected computational units organized into layers.

Each neuron performs:

\[
z=Wx+b
\]

followed by an activation function:

\[
a=f(z)
\]

By connecting many neurons and layers, the network can learn increasingly complex relationships.

The overall learning process consists of:

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
Gradient
 ↓
Optimization
 ↓
Updated Parameters
```

The most important conceptual progression is:

```
Perceptron
    ↓
Neural Network
    ↓
Deep Neural Network
    ↓
Representation Learning
    ↓
CNN
    ↓
Residual Learning
    ↓
ResNet
```

The central idea is:

> A neural network learns a mapping from inputs to outputs by combining learnable parameters with nonlinear transformations, allowing increasingly complex representations to be learned from data.

