# Activation Functions in Neural Networks

## 1. Introduction

An **Activation Function** is a mathematical function applied to the output of a neuron or layer.

Activation functions determine how the output of a neural network layer is transformed before being passed to the next layer.

A simplified neural network operation is:

```text
Input
  ↓
Weighted Sum
  ↓
Activation Function
  ↓
Output
```

Mathematically:

\[
z = Wx + b
\]

and then:

\[
a = f(z)
\]

where:

- \(x\) = input
- \(W\) = learnable weights
- \(b\) = bias
- \(z\) = pre-activation value
- \(f\) = activation function
- \(a\) = activation

---

## 2. Why Do We Need Activation Functions?

The most important purpose of an activation function is to introduce non-linearity into a neural network.

Without activation functions, even a very deep neural network would behave like a single linear transformation.

Suppose we have:

Input
  ↓
Linear Layer
  ↓
Linear Layer
  ↓
Linear Layer
  ↓
Output

Each layer performs:

\[
y = Wx+b
\]

The composition of multiple linear transformations can still be represented as another linear transformation.

Therefore:

Linear
  ↓
Linear
  ↓
Linear

does not provide the network with enough expressive power to learn complex non-linear relationships.

Activation functions solve this problem:

Linear Transformation
        ↓
Activation Function
        ↓
Non-Linear Representation

---

## 3. Linear vs Non-Linear Networks

Consider a simple linear function:

\[
y = 2x
\]

This relationship can be represented by a straight line.

However, many real-world problems are not linear.

For example, an image classification problem may require the network to distinguish between highly complex patterns.

CNNs therefore need to learn non-linear relationships.

Activation functions provide this capability.

Input
  ↓
Convolution
  ↓
Activation Function
  ↓
Non-Linear Feature Representation

---

## 4. Activation Functions in CNNs

In a CNN, a typical sequence is:

Input Image
     ↓
Convolution
     ↓
Batch Normalization
     ↓
Activation Function
     ↓
Feature Map

For example, in ResNet:

7 × 7 Conv
     ↓
BatchNorm
     ↓
ReLU

The convolution extracts a learned feature representation.

The activation function then introduces non-linearity into that representation.

---

## 5. Pre-Activation and Activation

Suppose a convolution produces:

```
[-3, -1, 0, 2, 5]
```

These values are called pre-activations.

After applying an activation function, we obtain the activations.

For example, using ReLU:

Before ReLU:

```
[-3, -1, 0, 2, 5]
```

After ReLU:

```
[0, 0, 0, 2, 5]
```

Therefore:

Convolution Output
        ↓
Pre-Activation
        ↓
Activation Function
        ↓
Activation

---

## 6. ReLU

The most widely used activation function in CNNs is ReLU.

ReLU stands for:

> Rectified Linear Unit

Its mathematical definition is:

\[
ReLU(x)=\max(0,x)
\]

This means:

- If \(x < 0\): \(ReLU(x) = 0\)
- If \(x \geq 0\): \(ReLU(x) = x\)

---

## 7. ReLU Example

Consider:

```
[-5, -2, -1, 0, 2, 4, 7]
```

Applying ReLU:

```
[0, 0, 0, 0, 2, 4, 7]
```

The negative values become zero.

Positive values remain unchanged.

Therefore:

Negative Activation
        ↓
        0

Positive Activation
        ↓
    Unchanged

---

## 8. ReLU on Feature Maps

ReLU is applied element-by-element.

Suppose a convolution produces a feature map:

```
┌────┬────┬────┐
│ -2 │  4 │ -1 │
├────┼────┼────┤
│  3 │ -5 │  2 │
├────┼────┼────┤
│ -1 │  6 │  0 │
└────┴────┴────┘
```

After ReLU:

```
┌────┬────┬────┐
│  0 │  4 │  0 │
├────┼────┼────┤
│  3 │  0 │  2 │
├────┼────┼────┤
│  0 │  6 │  0 │
└────┴────┴────┘
```

ReLU does not change the spatial dimensions.

For example:

```
56 × 56 × 64
remains 56 × 56 × 64
```

Only the values are transformed.

---

## 9. ReLU Does Not Change the Number of Feature Maps

Suppose a convolution produces:

```
56 × 56 × 64
```

This means:

64 Feature Maps

Applying ReLU gives:

```
56 × 56 × 64
```

There are still:

64 Feature Maps

ReLU operates on the values inside the feature maps.

Therefore:

Before ReLU:
```
56 × 56 × 64
       ↓ ReLU
After ReLU:
56 × 56 × 64
```

The dimensions remain unchanged.

---

## 10. Why Is ReLU Useful?

ReLU has several important advantages.

1. **Introduces Non-Linearity**: It allows CNNs to learn complex relationships.
2. **Computationally Simple**: The operation is simply: \(\max(0,x)\)
3. **Efficient Gradient Computation**: For positive values, its derivative is \(1\), which helps gradient propagation.
4. **Sparse Activations**: Negative values become zero, producing sparse representations.

---

## 11. ReLU Derivative

The derivative of ReLU is:

\[
ReLU'(x)=
\begin{cases}
0 & x<0\\
1 & x>0
\end{cases}
\]

At \(x=0\), the derivative is not uniquely defined, but implementations use a practical convention.

The important idea is:

Positive Input
      ↓
Derivative ≈ 1

Negative Input
      ↓
Derivative = 0

This property is important during backpropagation.

---

## 12. ReLU and Vanishing Gradients

One reason ReLU became popular is that it avoids some of the severe gradient shrinking associated with sigmoid and tanh in their saturated regions.

For positive values:

\[
ReLU'(x)=1
\]

Therefore, the gradient does not automatically become very small simply because the activation is large and positive.

Conceptually:

Positive Activation
       ↓
Derivative ≈ 1
       ↓
Gradient can propagate

This helped make deep neural networks easier to train.

However, ReLU does not completely solve the vanishing-gradient problem.

Architectures such as ResNet further improve gradient flow using skip connections.

---

## 13. The Dying ReLU Problem

ReLU has an important limitation.

If a neuron consistently receives negative inputs:

\(x < 0\)

then:

\[
ReLU(x)=0
\]

and:

\[
ReLU'(x)=0
\]

Therefore, the gradient through that neuron can become zero.

If this happens persistently, the neuron may stop learning.

This is called the:

> Dying ReLU Problem

Conceptually:

Negative Inputs
      ↓
ReLU = 0
      ↓
Gradient = 0
      ↓
Neuron may stop updating

---

## 14. Leaky ReLU

Leaky ReLU was introduced to address the dying ReLU problem.

Its definition is:

\[
f(x)=
\begin{cases}
x & x>0\\
\alpha x & x\leq 0
\end{cases}
\]

where \(\alpha\) is a small positive value. For example:

\[
\alpha=0.01
\]

Then:

Input:
```
[-5, -2, 0, 3, 5]
```

becomes approximately:

```
[-0.05, -0.02, 0, 3, 5]
```

Unlike ReLU, negative inputs are not mapped exactly to zero.

---

## 15. ReLU vs Leaky ReLU

ReLU

\[
f(x)=\max(0,x)
\]

- Negative → 0
- Positive → unchanged

Leaky ReLU

Negative → Small negative value
Positive → unchanged

Conceptually:

ReLU:

```
       /
      /
-----/
```

Leaky ReLU:

```
       /
      /
-----/
    /
```

The small negative slope allows gradients to continue flowing for negative inputs.

---

## 16. Sigmoid

Another activation function is the Sigmoid function.

It is defined as:

\[
\sigma(x)=\frac{1}{1+e^{-x}}
\]

Its output is always between:

\[
0 \ < \sigma(x) \ < \ 1
\]

For example:

Input     Output

-5        ≈ 0.007  
-2        ≈ 0.119  
 0        = 0.5  
 2        ≈ 0.881  
 5        ≈ 0.993  

Therefore:

Large Negative
      ↓
Output ≈ 0

Zero
      ↓
Output = 0.5

Large Positive
      ↓
Output ≈ 1

---

## 17. Sigmoid Shape

The sigmoid function has an S-shaped curve.

Conceptually:

```
1 |             ______
  |          __/
  |       __/
0.5|------/
  |    __
  |  _/
0 |__/
  +-------------------
```

Its output is bounded between 0 and 1.

This makes it useful in some binary classification settings.

---

## 18. The Problem with Sigmoid in Deep Networks

Sigmoid can suffer from vanishing gradients.

Its derivative is:

\[
\sigma'(x)=\sigma(x)(1-\sigma(x))
\]

The maximum derivative is \(0.25\) and for very large positive or negative inputs, the derivative approaches zero.

For example:

```
Large Positive Input
       ↓
Sigmoid ≈ 1
       ↓
Derivative ≈ 0
```

and:

```
Large Negative Input
       ↓
Sigmoid ≈ 0
       ↓
Derivative ≈ 0
```

During backpropagation, repeated multiplication by small gradients can make the gradient extremely small.

This is one reason sigmoid is generally not used as the main activation function in deep CNN hidden layers.

---

## 19. Tanh

Another classical activation function is Tanh.

It is defined as:

\[
tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}
\]

Its output range is:

\[
-1 < tanh(x) < 1
\]

For example:

- Large Negative → approximately -1
- Zero           → 0
- Large Positive → approximately +1

Conceptually:

```
-1 |____
   |    \
  0|-----\
   |      \____
+1 |
```

Tanh is zero-centered, unlike sigmoid.

---

## 20. Tanh and Vanishing Gradients

Although tanh behaves better than sigmoid because it is zero-centered, it can still suffer from vanishing gradients.

For large positive or negative values:

\[
tanh'(x)\rightarrow0
\]

Therefore:

```
Large |x|
   ↓
Tanh Saturation
   ↓
Small Gradient
```

This is one reason ReLU-family functions became more common in deep CNNs.

---

## 21. Softmax

Softmax is commonly used in the final layer of a multi-class classification network.

It converts a vector of logits into a probability distribution.

For \(K\) classes:

\[
P(y=i)=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
\]

The outputs satisfy:

\[
\sum_i P(y=i)=1
\]

and:

\[
0<P(y=i)<1
\]

---

## 22. Softmax Example

Suppose a classifier produces:

Logits:

```
[2.0, 1.0, 0.1]
```

Softmax converts them into probabilities approximately like:

```
[0.66, 0.24, 0.10]
```

The probabilities sum to:

\[
0.66 + 0.24 + 0.10 = 1
\]

Therefore:

- Class 1 → 66%
- Class 2 → 24%
- Class 3 → 10%

The class with the highest probability becomes the predicted class.

---

## 23. Softmax vs ReLU

ReLU and Softmax serve completely different purposes.

### ReLU

Usually used inside the network:

```
Convolution
   ↓
ReLU
   ↓
Next Layer
```

Purpose:

> Introduce non-linearity.

### Softmax

Usually used at the final classification stage:

```
Final Logits
     ↓
Softmax
     ↓
Class Probabilities
```

Purpose:

> Convert class scores into a probability distribution.

---

## 24. Activation Functions in a CNN

A simplified CNN may look like:

```
Input Image
     ↓
Convolution
     ↓
ReLU
     ↓
Pooling
     ↓
Convolution
     ↓
ReLU
     ↓
Pooling
     ↓
Fully Connected Layer
     ↓
Softmax
     ↓
Class Probabilities
```

The activation functions provide non-linearity throughout the network.

---

## 25. Activation Functions in ResNet

ResNet primarily uses ReLU in its original architecture.

A simplified sequence is:

```
Input
  ↓
7 × 7 Conv
  ↓
BatchNorm
  ↓
ReLU
  ↓
Max Pooling
  ↓
Residual Blocks
  ↓
Global Average Pooling
  ↓
Fully Connected Layer
  ↓
Classification
```

Inside a bottleneck residual block, the simplified structure is:

```
Input
  │
  ├─────────────────────────┐
  │                         │
  ↓                         │
1 × 1 Conv                  │
  ↓                         │
BatchNorm                   │
  ↓                         │
ReLU                        │
  ↓                         │
3 × 3 Conv                  │
  ↓                         │
BatchNorm                   │
  ↓                         │
ReLU                        │
  ↓                         │
1 × 1 Conv                  │
  ↓                         │
BatchNorm                   │
  │                         │
  └────────── Add ←─────────┘
              ↓
             ReLU
```

The exact placement of activations is important when describing the original ResNet implementation.

---

## 26. ReLU Inside a Residual Block

Consider:

\[
y=F(x)+x
\]

where:

- \(x\) = shortcut input
- \(F(x)\) = residual branch

After addition, the original ResNet applies ReLU:

\[
y=ReLU(F(x)+x)
\]

Conceptually:

```
Residual Branch
      ↓
     F(x)
      │
      ├───────┐
      │       │
Input x ──────┤
              ↓
             Add
              ↓
             ReLU
              ↓
            Output
```

This activation introduces non-linearity after the residual addition.

---

## 27. Why Activation Functions Matter for Deep CNNs

A deep CNN contains many layers.

Without non-linear activation functions:

```
Conv
 ↓
Linear
 ↓
Conv
 ↓
Linear
 ↓
Conv
```

the overall network would still behave like a largely linear transformation.

With activation functions:

```
Conv
 ↓
ReLU
 ↓
Conv
 ↓
ReLU
 ↓
Conv
 ↓
ReLU
```

the network can learn complex non-linear representations.

This allows CNNs to model:

- Edges
- Textures
- Shapes
- Object parts
- Complex visual patterns
- High-level representations

---

## 28. Activation Functions and Representation Learning

Activation functions are a key component of representation learning.

A simplified hierarchy is:

```
Raw Pixels
    ↓
Linear Transformation
    ↓
Activation
    ↓
Low-Level Representation
    ↓
Linear Transformation
    ↓
Activation
    ↓
Mid-Level Representation
    ↓
Linear Transformation
    ↓
Activation
    ↓
High-Level Representation
```

The repeated combination of:

Linear Transformation
+
Non-Linear Activation

allows the network to progressively learn more complex representations.

---

## 29. Activation Functions and Backpropagation

Activation functions also affect gradient flow during training.

During backpropagation, the chain rule is used:

\[
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\frac{\partial y}{\partial x}
\]

The derivative of the activation function therefore becomes part of the gradient calculation.

For example, with ReLU:

\[
ReLU'(x)=
\begin{cases}
0 & x<0\\
1 & x>0
\end{cases}
\]

Therefore, the activation function directly affects how gradients propagate backward through the network.

---

## 30. Activation Functions and the Vanishing Gradient Problem

Activation functions such as sigmoid and tanh can saturate.

For example:

```
Large Positive Input
        ↓
Activation approaches a limit
        ↓
Derivative approaches zero
        ↓
Gradient becomes small
```

Repeated across many layers:

```
Small Gradient
      ↓
Small Gradient
      ↓
Smaller Gradient
      ↓
Very Small Gradient
```

This contributes to the vanishing gradient problem.

ReLU reduces this problem for positive activations because:

\[
ReLU'(x)=1
\]

for positive \(x\).

However, ReLU has its own problem for negative inputs.

---

## 31. Comparison of Common Activation Functions

| Activation Function | Output Range           | Main Advantage                          | Main Limitation                   |
|---------------------|-----------------------|-----------------------------------------|-----------------------------------|
| ReLU                | \([0, \infty)\)      | Simple and efficient                    | Dying ReLU                        |
| Leaky ReLU          | \((-\infty, \infty)\)| Keeps negative gradient                  | Requires slope parameter           |
| Sigmoid             | \((0, 1)\)           | Useful for probabilities                | Vanishing gradients               |
| Tanh                | \((-1, 1)\)          | Zero-centered                           | Vanishing gradients               |
| Softmax             | \((0, 1)\), sum = 1  | Multi-class probabilities               | Usually used only at output      |

---

## 32. ReLU vs Sigmoid vs Tanh

A simplified comparison:

### ReLU
```
────────────
Negative → 0
Positive → unchanged
```

### Sigmoid
```
────────────
Everything → 0 to 1
```

### Tanh
```
────────────
Everything → -1 to 1
```

In deep CNN hidden layers:

- **ReLU**: Common Choice  
- **Sigmoid** and **Tanh**: Used more selectively.

---

## 33. Activation Functions and Sparsity

ReLU maps all negative activations to zero.

For example:

Before:

```
[-2, 3, -1, 5, -4, 2]
```

After:

```
[0, 3, 0, 5, 0, 2]
```

Some activations become exactly zero.

This produces a sparse activation pattern.

Conceptually:

```
Many Activations
      ↓
ReLU
      ↓
Some Become Zero
      ↓
Sparse Representation
```

Sparse representations can be useful because not every learned feature needs to be active for every input.

---

## 34. Activation Function Is Not a Filter

It is important not to confuse an activation function with a convolutional filter.

### A convolutional filter:
- Contains Learnable Weights

### An activation function:
- Applies a Mathematical Transformation

For example:

```
Convolution
    ↓
Learned Filter
    ↓
Feature Map
    ↓
ReLU
    ↓
Activated Feature Map
```

The filter learns during training.

ReLU itself does not learn weights.

---

## 35. Activation Function Is Not Pooling

Activation and pooling also perform different operations.

### Activation
Transforms each value independently.

For ReLU:

\[
x \rightarrow \max(0,x)
\]

### Pooling
Combines values from a local spatial region.

For Max Pooling:

\[
[x_1,x_2,x_3,x_4] \rightarrow \max(x_1,x_2,x_3,x_4)
\]

Therefore:

- **ReLU** → Element-wise transformation
- **Pooling** → Spatial summarization

---

## 36. Complete CNN Example

Consider:

```
Input Image
224 × 224 × 3
       ↓
Convolution
       ↓
Batch Normalization
       ↓
ReLU
       ↓
Feature Maps
       ↓
Max Pooling
       ↓
Smaller Feature Maps
       ↓
Convolution
       ↓
Batch Normalization
       ↓
ReLU
       ↓
Deeper Feature Maps
       ↓
Global Average Pooling
       ↓
Feature Vector
       ↓
Fully Connected Layer
       ↓
Softmax
       ↓
Class Probabilities
```

Each component has a different responsibility:

- **Convolution** → Feature Extraction
- **Batch Normalization** → Activation Stabilization
- **ReLU** → Non-Linearity
- **Pooling** → Spatial Downsampling
- **Global Average Pooling** → Spatial Aggregation
- **Fully Connected Layer** → Classification Mapping
- **Softmax** → Probability Distribution

---

## 37. Key Takeaways

- Activation functions introduce non-linearity into neural networks.
- Without them, stacking linear layers would not provide sufficient expressive power.
- ReLU is the most common activation function in traditional CNNs and ResNet.
- ReLU is defined as:

\[
ReLU(x)=\max(0,x)
\]

- ReLU sets negative activations to zero.
- ReLU does not change spatial dimensions or number of channels.
- ReLU can help reduce the vanishing-gradient problem for positive activations.
- ReLU can suffer from the dying ReLU problem.
- Leaky ReLU keeps a small gradient for negative inputs.
- Sigmoid maps values to the range (0,1).
- Tanh maps values to the range (-1,1).
- Sigmoid and tanh can suffer from vanishing gradients in saturated regions.
- Softmax converts class logits into a probability distribution.
- Activation functions are different from convolution filters.
- Activation functions are also different from pooling operations.
- Activation functions play a central role in CNN representation learning.
- In ResNet, ReLU is used throughout the network to introduce non-linearity.
- Activation functions directly influence gradient flow during backpropagation.
- Choosing an appropriate activation function is important in neural network architecture design.

---

## 38. Conceptual Summary

The role of activation functions can be summarized as:

```
Input
  ↓
Weighted Transformation
  ↓
Convolution / Linear Layer
  ↓
Activation Function
  ↓
Non-Linear Representation
  ↓
Next Layer
```

For CNNs:

```
Image
  ↓
Convolution
  ↓
Feature Map
  ↓
ReLU
  ↓
Activated Feature Map
  ↓
Pooling / Downsampling
  ↓
Deeper Convolution
  ↓
Deeper Representation
```

The fundamental idea is:

> Convolution learns features, while activation functions introduce the non-linearity that allows a deep neural network to learn complex representations.

