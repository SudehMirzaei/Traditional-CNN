# Forward Propagation

## 1. Introduction

**Forward Propagation** is the process of passing input data through a neural network from the input layer toward the output layer.

During forward propagation, each layer receives the output of the previous layer, performs a mathematical transformation, and passes the result to the next layer.

The overall process is:

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

The final output of the network is the model's prediction.

---

## 2. Why Is Forward Propagation Important?

Forward propagation is responsible for answering a fundamental question:

> Given this input and the current model parameters, what prediction does the network produce?

For example, in an image classification problem:

```
Input Image
    ↓
Neural Network
    ↓
Prediction
```

If the network is classifying skin lesions, the output might represent probabilities for different lesion classes.

---

## 3. Forward Propagation vs Backpropagation

Forward propagation and backpropagation perform opposite-direction operations.

### Forward Propagation

Data moves from input toward output:

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

### Backpropagation

Gradients move from the loss toward earlier layers:

```
Loss
  ↓
Layer 3
  ↓
Layer 2
  ↓
Layer 1
```

Therefore:

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
    Optimizer
```

---

## 4. Basic Neuron

To understand forward propagation, we first need to understand what happens inside a single neuron.

A neuron receives inputs:

- \(x_1\)
- \(x_2\)
- \(x_3\)

Each input has an associated weight:

- \(w_1\)
- \(w_2\)
- \(w_3\)

The neuron calculates a weighted sum:

\[
z=w_1x_1+w_2x_2+w_3x_3
\]

Then a bias is added:

\[
z=w_1x_1+w_2x_2+w_3x_3+b
\]

Finally, an activation function is applied:

\[
a=f(z)
\]

The complete process is:

```
Inputs
  ↓
Weighted Sum
  ↓
Add Bias
  ↓
Activation Function
  ↓
Output
```

---

## 5. Mathematical Representation

For multiple inputs, the neuron can be written compactly as:

\[
z=\mathbf{w}^T\mathbf{x}+b
\]

Then:

\[
a=f(z)
\]

where:

- \(\mathbf{x}\) = input vector
- \(\mathbf{w}\) = weight vector
- \(b\) = bias
- \(z\) = pre-activation value
- \(f\) = activation function
- \(a\) = activation/output

---

## 6. Example of a Single Neuron

Suppose:

\[
x_1=2
\]

\[
x_2=3
\]

and:

\[
w_1=0.5
\]

\[
w_2=0.2
\]

with:

\[
b=1
\]

The weighted sum is:

\[
z=(0.5)(2)+(0.2)(3)+1
\]

\[
z=1+0.6+1
\]

\[
z=2.6
\]

If the neuron uses ReLU:

\[
a=ReLU(2.6)
\]

Therefore:

\[
a=2.6
\]

---

## 7. Layers

A neural network contains multiple layers.

A simple network might look like:

```
Input Layer
     ↓
Hidden Layer
     ↓
Output Layer
```

A deeper network may contain:

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

During forward propagation, the output of each layer becomes the input to the next layer.

---

## 8. Forward Propagation Through One Layer

For layer \(l\), the mathematical operation is:

\[
z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}
\]

Then the activation function is applied:

\[
a^{(l)} = f(z^{(l)})
\]

Here:

- \(a^{(l-1)}\) = output from the previous layer
- \(W^{(l)}\) = weights of the current layer
- \(b^{(l)}\) = bias
- \(z^{(l)}\) = pre-activation
- \(a^{(l)}\) = activation/output

---

## 9. Multiple Layers

Suppose the network has three layers.

The first layer computes:

\[
z^{(1)} = W^{(1)}x+b^{(1)}
\]

\[
a^{(1)} = f(z^{(1)})
\]

The second layer receives \(a^{(1)}\):

\[
z^{(2)} = W^{(2)}a^{(1)}+b^{(2)}
\]

\[
a^{(2)} = f(z^{(2)})
\]

The third layer computes:

\[
z^{(3)} = W^{(3)}a^{(2)}+b^{(3)}
\]

\[
a^{(3)} = f(z^{(3)})
\]

Conceptually:

```
Input x
  ↓
W₁x + b₁
  ↓
Activation
  ↓
a¹
  ↓
W₂a¹ + b₂
  ↓
Activation
  ↓
a²
  ↓
W₃a² + b₃
  ↓
Activation
  ↓
a³
```

---

## 10. Matrix Multiplication

Neural networks usually process many neurons simultaneously using matrix operations.

Suppose:

- Input Size = 4
- Hidden Layer = 3 neurons

Then:

\[
x\in\mathbb{R}^{4}
\]

and:

\[
W\in\mathbb{R}^{3\times4}
\]

The output is:

\[
z=Wx+b
\]

The result has three values:

- \(z_1\)
- \(z_2\)
- \(z_3\)

One value corresponds to each neuron in the hidden layer.

---

## 11. Why Matrix Operations Are Important

Instead of calculating every neuron separately:

```
Neuron 1 → weighted sum
Neuron 2 → weighted sum
Neuron 3 → weighted sum
```

we can calculate them together:

\[
\mathbf{z}=W\mathbf{x}+\mathbf{b}
\]

This makes neural network computation highly efficient, especially when using GPUs.

---

## 12. Activation Functions

After the linear transformation, the network usually applies an activation function.

The general form is:

\[
a=f(z)
\]

Common activation functions include:

- ReLU
- Sigmoid
- Tanh
- GELU
- Softmax

---

## 13. ReLU

ReLU is commonly used in hidden layers.

It is defined as:

\[
ReLU(x)=\max(0,x)
\]

For example:

\[
ReLU(-2)=0
\]

\[
ReLU(3)=3
\]

Therefore:

Before ReLU:

```
[-2, 1, -4, 5]
```

After ReLU:

```
[0, 1, 0, 5]
```

---

## 14. Why Do We Need Activation Functions?

Without nonlinear activation functions, stacking multiple linear layers would still produce an overall linear transformation.

For example:

\[
z_1=W_1x+b_1
\]

and:

\[
z_2=W_2z_1+b_2
\]

Even though there are two layers, the overall transformation is still effectively linear with respect to the input.

Nonlinear activation functions allow neural networks to model complex nonlinear relationships.

Therefore:

```
Linear Transformation
       ↓
Nonlinear Activation
       ↓
Linear Transformation
       ↓
Nonlinear Activation
```

is much more expressive than stacking linear transformations alone.

---

## 15. Forward Propagation in a Classification Network

For a classification problem, the final layer produces class scores.

For example:

```
Input Image
    ↓
Neural Network
    ↓
Class Scores
```

Suppose there are three classes:

- Class A → 2.1
- Class B → 4.8
- Class C → 1.2

These values are commonly called logits before applying Softmax.

---

## 16. Logits

A logit is an unnormalized score produced by the final layer.

For example:

\[
z=[2.1, 4.8, 1.2]
\]

These values are not probabilities because they do not necessarily lie between 0 and 1 and do not necessarily sum to 1.

Softmax can transform them into probabilities.

---

## 17. Softmax

For class \(i\), Softmax is:

\[
P(y=i) = \frac{e^{z_i}}{\sum_j e^{z_j}}
\]

For example, the network might produce:

- Class A → 0.08
- Class B → 0.87
- Class C → 0.05

The probabilities sum to approximately:

\[
0.08 + 0.87 + 0.05 = 1
\]

The predicted class is usually the class with the highest probability.

Therefore, Class B → 0.87 would be the predicted class.

---

## 18. Binary Classification

For binary classification, the final layer often produces one logit.

A sigmoid function can convert it into a probability:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

For example:

Logit
 ↓
Sigmoid
 ↓
0.91

This can represent a high probability for the positive class.

---

## 19. Regression

Forward propagation is not limited to classification.

For regression, the final layer can directly produce a continuous value.

For example:

```
Input
  ↓
Neural Network
  ↓
Output
  ↓
42.7
```

The prediction could represent:

- Temperature
- House price
- Age
- Blood pressure
- Stock value

depending on the task.

---

## 20. Forward Propagation and Loss

Forward propagation itself produces the prediction.

The loss is calculated afterward.

The complete sequence is:

```
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss Function
  ↓
Loss
```

For example:

- True Label = 1
- Prediction = 0.80

```
        ↓
Loss Function
        ↓
Loss = ...
```

The loss measures how well the current prediction matches the target.

---

## 21. Forward Propagation During Training

During training, forward propagation happens repeatedly.

A simplified training iteration is:

```
Input Batch
    ↓
Forward Propagation
    ↓
Prediction
    ↓
Loss
    ↓
Backpropagation
    ↓
Optimizer
    ↓
Updated Parameters
```

After the parameters are updated, another forward pass is performed.

---

## 22. Forward Propagation During Inference

Forward propagation is also used when the model is making predictions on new data.

During inference:

```
New Input
    ↓
Forward Propagation
    ↓
Prediction
```

There is no need to calculate gradients when simply making predictions.

Conceptually:

**Training:**

```
Input
 ↓
Forward
 ↓
Loss
 ↓
Backward
 ↓
Update
```

**Inference:**

```
Input
 ↓
Forward
 ↓
Prediction
```

---

## 23. Training vs Inference

| Training                    | Inference                        |
|-----------------------------|----------------------------------|
| Forward propagation          | Forward propagation               |
| Loss calculation             | Usually no loss needed            |
| Backpropagation              | No backpropagation                |
| Gradient calculation         | No gradients needed              |
| Parameter updates            | No parameter updates             |

The forward pass is therefore common to both training and inference.

---

## 24. Forward Propagation in CNNs

For a CNN, forward propagation is more specialized.

A typical CNN may perform:

```
Input Image
    ↓
Convolution
    ↓
Batch Normalization
    ↓
Activation
    ↓
Pooling
    ↓
Convolution
    ↓
Activation
    ↓
...
    ↓
Feature Representation
    ↓
Fully Connected Layer
    ↓
Prediction
```

Each operation transforms the representation of the image.

---

## 25. Convolution During Forward Propagation

A convolutional layer applies learnable filters to the input.

For example:

```
Input Image
    ↓
3×3 Convolution
    ↓
Feature Maps
```

Each filter contains learnable weights.

A filter slides over the input and performs local computations.

The result is a feature map.

---

## 26. Feature Maps

Suppose a convolutional layer contains 64 filters.

The output may contain:

\[
64
\]

feature maps.

For example:

```
Input
 ↓
Convolution
 ↓
64 Feature Maps
```

If each feature map has spatial dimensions:

\[
56\times56
\]

the output shape is:

\[
56\times56\times64
\]

These feature maps represent learned responses to different patterns.

---

## 27. Batch Normalization

CNNs often use Batch Normalization after convolution.

A simplified sequence is:

```
Convolution
    ↓
Batch Normalization
    ↓
ReLU
```

Batch Normalization normalizes activations and then applies learnable scale and shift parameters.

Conceptually:

```
Feature Maps
    ↓
Normalization
    ↓
Scaled / Shifted Activations
```

This can help stabilize training.

---

## 28. Pooling During Forward Propagation

Pooling reduces spatial dimensions.

For example:

```
112 × 112 × 64
       ↓
3×3 Max Pool
       ↓
56 × 56 × 64
```

Max Pooling selects the maximum activation within each local region.

Its purpose includes:

- Reducing spatial resolution
- Reducing computation
- Providing some translation robustness
- Retaining strong local activations

---

## 29. Forward Propagation in ResNet

ResNet uses residual blocks during the forward pass.

A simplified residual block is:

```
┌───────────────┐
             │               │
Input ───────┤ Conv → BN → ReLU
  │          │      ↓
  │          │    Conv → BN
  │          │               │
  └──────────┴──────→ Add ←──┘
                         ↓
                       ReLU
                         ↓
                      Output
```

The residual branch computes:

\[
F(x)
\]

while the shortcut provides:

\[
x
\]

The final output is:

\[
y=F(x)+x
\]

---

## 30. Forward Propagation Through a Residual Block

Suppose the input is:

\[
x
\]

The residual branch computes:

\[
F(x)
\]

The shortcut provides:

\[
x
\]

Then:

\[
y=F(x)+x
\]

Finally:

\[
output=ReLU(y)
\]

Conceptually:

```
Input x
   │
   ├───────────────┐
   │               │
   ↓               │
Residual Branch    │
   │               │
   ↓               │
  F(x)             │
   │               │
   └──────→ Add ←──┘
              ↓
            ReLU
              ↓
           Output
```

---

## 31. Projection Shortcut

Sometimes the dimensions of the residual branch and shortcut are different.

For example:

```
56 × 56 × 64
      ↓
28 × 28 × 256
```

The tensors cannot be directly added.

ResNet can use a projection shortcut:

\[
W_sx
\]

usually implemented using a \(1\times1\) convolution.

Then:

\[
y=F(x)+W_sx
\]

Conceptually:

```
Input
  │
  ├────────→ 1×1 Conv ─────┐
  │                         │
  ↓                         │
Residual Branch             │
  │                         │
  ↓                         │
  F(x)                      │
  │                         │
  └──────────────→ Add ←────┘
                       ↓
                     Output
```

---

## 32. Forward Propagation Through ResNet50

A simplified ResNet50 forward pass is:

```
Input Image
     ↓
7×7 Convolution
     ↓
Batch Normalization
     ↓
ReLU
     ↓
3×3 Max Pooling
     ↓
Residual Stage 1
     ↓
Residual Stage 2
     ↓
Residual Stage 3
     ↓
Residual Stage 4
     ↓
Global Average Pooling
     ↓
Fully Connected Layer
     ↓
Logits
     ↓
Softmax
     ↓
Class Probabilities
```

---

## 33. Spatial Resolution During Forward Propagation

As ResNet becomes deeper, spatial resolution generally decreases.

A simplified progression is:

```
224 × 224
     ↓
112 × 112
     ↓
56 × 56
     ↓
28 × 28
     ↓
14 × 14
     ↓
7 × 7
```

At the same time, the number of channels generally increases.

Conceptually:

- Spatial Resolution ↓
- Channels ↑

This allows the network to gradually move from detailed spatial information toward richer high-level representations.

---

## 34. Global Average Pooling

Near the end of ResNet, Global Average Pooling (GAP) converts spatial feature maps into a vector.

Suppose the final feature representation is:

\[
7\times7\times2048
\]

Global Average Pooling averages each of the 2048 feature maps:

```
7 × 7 × 2048
      ↓
Global Average Pooling
      ↓
2048-dimensional Vector
```

Therefore:

\[
7\times7\times2048 \rightarrow 2048
\]

---

## 35. Fully Connected Layer

The final feature vector can then be passed to a fully connected layer.

For example:

```
2048-dimensional Vector
          ↓
Fully Connected Layer
          ↓
Number of Classes
```

If there are 4 classes:

\[
2048 \rightarrow 4
\]

The output contains four logits.

---

## 36. Complete ResNet50 Example

For an input image of:

\[
224\times224\times3
\]

a simplified forward pass is:

```
224 × 224 × 3
       ↓
7×7 Conv
       ↓
112 × 112 × 64
       ↓
3×3 MaxPool
       ↓
56 × 56 × 64
       ↓
Residual Stage 1
       ↓
56 × 56 × 256
       ↓
Residual Stage 2
       ↓
28 × 28 × 512
       ↓
Residual Stage 3
       ↓
14 × 14 × 1024
       ↓
Residual Stage 4
       ↓
7 × 7 × 2048
       ↓
Global Average Pooling
       ↓
2048
       ↓
Fully Connected
       ↓
Class Logits
```

The exact internal transformations depend on the specific ResNet implementation, but this diagram captures the main architectural progression.

---

## 37. Forward Propagation and Representation Learning

Forward propagation does not simply produce the final prediction.

At every stage, the network creates a new representation.

For example:

```
Input Image
    ↓
Pixels
    ↓
Early Feature Maps
    ↓
Edges / Simple Patterns
    ↓
Intermediate Features
    ↓
Textures / Shapes
    ↓
Deep Features
    ↓
High-Level Representation
    ↓
Prediction
```

This is one of the fundamental ideas behind representation learning.

---

## 38. Intermediate Activations

Every layer produces intermediate activations.

For example:

```
Input
 ↓
Conv Layer
 ↓
Activation 1
 ↓
Conv Layer
 ↓
Activation 2
 ↓
Conv Layer
 ↓
Activation 3
 ↓
Output
```

These intermediate representations can be analyzed to understand what the network is learning.

This is especially important in explainable AI and embedding analysis.

---

## 39. Forward Propagation and Embeddings

Deep neural networks can produce high-dimensional feature representations.

For example, a ResNet50 may produce a:

\[
2048
\]

-dimensional feature vector before the final classifier.

Conceptually:

```
Image
  ↓
ResNet50
  ↓
Deep Feature Representation
  ↓
2048-dimensional Vector
```

Each image can therefore be represented as a point in a 2048-dimensional feature space.

These representations can be used for:

- Visualization
- Clustering
- Similarity analysis
- Retrieval
- Classification
- Representation analysis

---

## 40. Why Forward Propagation Is Important for Explainability

Many explainability methods analyze intermediate activations produced during the forward pass.

For example:

- Grad-CAM
- Grad-CAM++
- Activation visualization
- Feature map analysis
- Embedding analysis

A simplified process is:

```
Input Image
    ↓
Forward Propagation
    ↓
Intermediate Activations
    ↓
Feature Representation
    ↓
Prediction
```

These internal representations help researchers understand how the model processes an input.

---

## 41. Forward Propagation in Evaluation

During evaluation, the model performs a forward pass on unseen data.

For example:

```
Test Image
    ↓
Model
    ↓
Prediction
    ↓
Compare with True Label
    ↓
Evaluation Metrics
```

Metrics can include:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

The forward pass provides the predictions required to calculate these metrics.

---

## 42. Forward Propagation in PyTorch

A simplified PyTorch model might look like:

```python
import torch
import torch.nn as nn

class SimpleNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
```

The `forward()` method defines the sequence of operations performed during forward propagation.

---

## 43. Calling the Model

Suppose:

```python
x = torch.randn(1, 4)
```

We can pass it through the network:

```python
output = model(x)
```

Conceptually:

```
x
 ↓
Linear(4 → 8)
 ↓
ReLU
 ↓
Linear(8 → 2)
 ↓
Output
```

The returned tensor contains the model's output.

---

## 44. Forward Propagation and model(x)

In PyTorch, writing:

```python
output = model(x)
```

causes PyTorch to execute the model's forward computation.

You generally do not call:

```python
model.forward(x)
```

directly.

Instead:

```python
model(x)
```

is the standard approach because PyTorch's `nn.Module` machinery handles the forward call appropriately.

---

## 45. Training Example

A simplified training loop is:

```python
for inputs, targets in dataloader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

The forward propagation occurs here:

```python
outputs = model(inputs)
```

Then the loss is calculated:

```python
loss = criterion(outputs, targets)
```

Backpropagation follows:

```python
loss.backward()
```

Finally, parameters are updated:

```python
optimizer.step()
```

---

## 46. The Relationship Between Forward and Backward Passes

Forward propagation and backpropagation form a complete learning cycle.

**Forward Pass**

```
Input → Network → Prediction
```

```
 ↓
    Loss
```

**Backward Pass**

```
Loss → Gradients → Optimizer
```

The updated parameters affect the next forward pass.

Therefore, training is an iterative process.

---

## 47. Forward Propagation Does Not Learn by Itself

A forward pass alone does not change the network parameters.

For example:

```
Input
 ↓
Forward Propagation
 ↓
Prediction
```

The weights remain unchanged.

Learning requires:

```
Forward
 ↓
Loss
 ↓
Backpropagation
 ↓
Optimizer
 ↓
Parameter Update
```

Therefore:

> Forward propagation performs computation, while training changes the parameters based on the gradients calculated during backpropagation.

---

## 48. Common Mistakes

### Mistake 1: Thinking Forward Propagation Updates Weights

It does not.

The weights are updated by the optimizer.

### Mistake 2: Confusing Logits with Probabilities

A model's final output may be logits rather than probabilities.

For multi-class classification, Softmax can convert logits into probabilities.

### Mistake 3: Thinking Backpropagation Happens During Forward Propagation

The two are separate stages.

```
Forward → Prediction → Loss → Backward
```

### Mistake 4: Assuming Every Layer Uses the Same Operation

Different architectures use different layers.

For example, CNNs may use:

- Convolution
- BatchNorm
- ReLU
- Pooling
- Residual Connections
- Global Average Pooling
- Fully Connected Layers

---

## 49. Key Takeaways

The most important concepts are:

1. Forward propagation passes data from input to output.
2. Each layer transforms the representation produced by the previous layer.
3. A basic layer performs: \[ z=Wx+b \]
4. An activation function is often applied after the linear transformation.
5. The final layer produces the model's prediction.
6. In classification, the output may be converted into probabilities.
7. Forward propagation occurs during both training and inference.
8. Forward propagation itself does not update model parameters.
9. During training, the prediction is used to calculate a loss.
10. Backpropagation then calculates gradients.
11. The optimizer uses those gradients to update the parameters.
12. In CNNs, forward propagation transforms images into increasingly rich feature representations.
13. In ResNet, residual blocks provide an important structure for forward computation and deep representation learning.

---

## 50. Final Conceptual Diagram

### FORWARD PROPAGATION

```
Input
  │
  ↓
┌─────────────────────┐
│      Layer 1        │
│    Wx + b            │
│    Activation        │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│      Layer 2        │
│    Wx + b            │
│    Activation        │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│      Layer 3        │
│    Wx + b            │
│    Activation        │
└──────────┬──────────┘
           │
           ↓
      Output Layer
           │
           ↓
       Prediction
           │
           ↓
          Loss
           │
           ↓
    Backpropagation
           │
           ↓
       Gradients
           │
           ↓
       Optimizer
           │
           ↓
   Updated Parameters
```

The central idea is:

> Forward propagation is the process through which input data is transformed layer by layer into a prediction using the current parameters of the neural network.

