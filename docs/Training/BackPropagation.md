# Backpropagation

## 1. Introduction

**Backpropagation** is one of the fundamental algorithms used to train neural networks.

Its main purpose is to calculate how much each learnable parameter contributed to the prediction error.

The basic idea is:

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
Parameter Update
```

Backpropagation does not directly update the weights.

Instead, it calculates the gradients of the loss with respect to the model parameters.

An optimizer then uses these gradients to update the parameters.

---

## 2. Why Do We Need Backpropagation?

A neural network contains many learnable parameters:

- Weights
- Biases

For a large neural network, there may be millions or even billions of parameters.

After the network makes a prediction, we need to determine:

> How should each parameter change to reduce the error?

For a parameter (\(w\)), we want to know:

\[
\frac{\partial L}{\partial w}
\]

where:

- \(L\) = loss
- \(w\) = parameter

This tells us how sensitive the loss is to that parameter.

Backpropagation provides an efficient way to calculate these gradients.

---

## 3. The Complete Training Process

Backpropagation is one part of the larger training process.

The complete process is:

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
Parameter Update
      ↓
Repeat
```

These steps are repeated many times during training.

---

## 4. Forward Propagation

Before backpropagation can happen, the network performs forward propagation.

During forward propagation, data moves from the input layer toward the output layer.

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

Each layer performs a transformation.

For a typical layer:

\[
z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}
\]

Then an activation function is applied:

\[
a^{(l)}=f(z^{(l)})
\]

The final output becomes the prediction:

\[
\hat{y}
\]

---

## 5. Loss Calculation

After the network produces a prediction, it is compared with the true target.

For example:

```
True Label
    ↓
    1

Prediction
    ↓
   0.7
```

A loss function measures the difference between them.

Conceptually:

```
Prediction
     ↓
Compare
     ↓
True Label
     ↓
Loss
```

The goal of training is to minimize this loss.

---

## 6. What Is a Gradient?

A gradient describes how a quantity changes when its input changes.

For a weight (\(w\)):

\[
\frac{\partial L}{\partial w}
\]

means:

> How much does the loss change when the weight changes?

For example:

\[
\frac{\partial L}{\partial w}=2
\]

means that increasing (\(w\)) slightly tends to increase the loss.

If:

\[
\frac{\partial L}{\partial w}=-2
\]

then increasing (\(w\)) slightly tends to decrease the loss.

---

## 7. Why the Sign of the Gradient Matters

Consider the gradient:

\[
\frac{\partial L}{\partial w}=+5
\]

The positive sign indicates that increasing the weight tends to increase the loss.

Therefore, gradient descent moves the weight in the opposite direction.

If:

\[
\frac{\partial L}{\partial w}=-5
\]

the negative sign indicates that increasing the weight tends to decrease the loss.

Gradient descent therefore moves the weight in the positive direction.

This is why the update rule contains a minus sign:

\[
w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w}
\]

---

## 8. Chain Rule

The mathematical foundation of backpropagation is the chain rule from calculus.

Suppose:

\[
x \rightarrow y \rightarrow z
\]

where:

\[
y=f(x)
\]

and:

\[
z=g(y)
\]

Then:

\[
\frac{dz}{dx} = \frac{dz}{dy} \frac{dy}{dx}
\]

The chain rule allows us to calculate how a change in an early variable affects a final output through multiple intermediate operations.

This is exactly what is needed in a neural network.

---

## 9. Simple Example of the Chain Rule

Suppose:

\[
y=2x
\]

and:

\[
L=y^2
\]

We want:

\[
\frac{dL}{dx}
\]

Using the chain rule:

\[
\frac{dL}{dx} = \frac{dL}{dy} \frac{dy}{dx}
\]

We know:

\[
\frac{dL}{dy}=2y
\]

and:

\[
\frac{dy}{dx}=2
\]

Therefore:

\[
\frac{dL}{dx} = 2y \times 2
\]

So:

\[
\frac{dL}{dx}=4y
\]

Backpropagation applies this same principle through the many operations in a neural network.

---

## 10. A Simple Neural Network

Consider a very small network:

```
x
↓
Weighted Sum
↓
Activation
↓
Output
↓
Loss
```

Mathematically:

\[
z=wx+b
\]

\[
a=f(z)
\]

\[
L=L(a,y)
\]

We want to calculate:

\[
\frac{\partial L}{\partial w}
\]

The chain rule gives:

\[
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \frac{\partial a}{\partial z} \frac{\partial z}{\partial w}
\]

Since:

\[
z=wx+b
\]

we have:

\[
\frac{\partial z}{\partial w}=x
\]

Therefore:

\[
\boxed{
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial a} \frac{\partial a}{\partial z} x
}
\]

---

## 11. Backward Direction

During forward propagation, information moves:

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
  ↓
Loss
```

During backpropagation, gradients move in the opposite direction:

```
Loss
  ↓
Layer 3
  ↓
Layer 2
  ↓
Layer 1
  ↓
Input
```

This backward flow of gradients gives backpropagation its name.

---

## 12. What Does "Backpropagation" Actually Propagate?

An important distinction:

> Backpropagation propagates gradients, not predictions.

Forward propagation carries activations:

```
Input
 ↓
Activations
 ↓
Prediction
```

Backpropagation carries gradient information:

```
Loss
 ↓
Gradients
 ↓
Gradients
 ↓
Gradients
```

This distinction is fundamental.

---

## 13. Forward vs Backward Pass

| Forward Pass                  | Backward Pass                     |
|-------------------------------|-----------------------------------|
| Input → Output                | Loss → Earlier Layers             |
| Computes activations          | Computes gradients                 |
| Produces prediction           | Determines parameter sensitivity    |
| Uses model parameters         | Uses derivatives                   |
| Ends with loss calculation    | Ends with parameter gradients      |

Conceptually:

**Forward:**

```
Input → Layer 1 → Layer 2 → Output → Loss
```

**Backward:**

```
Loss → Gradients → Layer 2 → Layer 1
```

---

## 14. Gradient of a Weight

Suppose a network contains:

\[
w_1, w_2, w_3
\]

Backpropagation calculates:

\[
\frac{\partial L}{\partial w_1}
\]

\[
\frac{\partial L}{\partial w_2}
\]

\[
\frac{\partial L}{\partial w_3}
\]

Each gradient tells us how the loss changes with respect to its corresponding parameter.

The optimizer can then update them individually.

---

## 15. Gradient Descent

After backpropagation calculates the gradients, an optimization algorithm updates the parameters.

For a simple weight:

\[
w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w}
\]

where:

- \(w\) = weight
- \(L\) = loss
- \(\eta\) = learning rate

The optimizer attempts to move the parameters toward values that reduce the loss.

---

## 16. Backpropagation vs Gradient Descent

These two concepts are related but they are not the same thing.

**Backpropagation:**

Calculates:

\[
\frac{\partial L}{\partial \theta}
\]

where \(\theta\) represents the model parameters.

**Gradient Descent / Optimizer:**

Uses the gradients to update the parameters:

\[
\theta \leftarrow \theta - \eta \nabla_\theta L
\]

Therefore:

```
Backpropagation
      ↓
Computes Gradients
      ↓
Optimizer
      ↓
Updates Parameters
```

---

## 17. Example of One Training Step

Suppose:

- Weight = 0.5
- Gradient = 2
- Learning Rate = 0.1

The update is:

\[
w_{\text{new}} = 0.5 - (0.1)(2)
\]

\[
w_{\text{new}}=0.3
\]

So the weight changes:

```
0.5 → 0.3
```

The optimizer uses the gradient to move the parameter in a direction intended to reduce the loss.

---

## 18. Multiple Layers

Now consider a network with two layers:

```
Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Output
  ↓
Loss
```

The forward pass is:

\[
x \rightarrow a^{(1)} \rightarrow a^{(2)} \rightarrow \hat{y} \rightarrow L
\]

The backward pass calculates:

\[
\frac{\partial L}{\partial W^{(2)}}
\]

and:

\[
\frac{\partial L}{\partial W^{(1)}}
\]

The gradient for the first layer depends on the gradients coming from later layers.

---

## 19. Backpropagation Through a Layer

For a layer:

\[
z^{(l)} = W^{(l)}a^{(l-1)} + b^{(l)}
\]

and:

\[
a^{(l)} = f(z^{(l)})
\]

Backpropagation calculates gradients such as:

\[
\frac{\partial L}{\partial W^{(l)}}
\]

\[
\frac{\partial L}{\partial b^{(l)}}
\]

and:

\[
\frac{\partial L}{\partial a^{(l-1)}}
\]

The last quantity is particularly important because it allows the gradient to continue propagating toward earlier layers.

---

## 20. Local Gradients

Each operation in a neural network has a local derivative.

For example:

```
x
 ↓
Linear Transformation
 ↓
Activation
 ↓
Next Layer
```

During backpropagation, each operation computes its local derivative and passes the gradient backward.

Conceptually:

```
Incoming Gradient
       ↓
Local Derivative
       ↓
Outgoing Gradient
```

The chain rule combines these local derivatives.

---

## 21. Computational Graph

A neural network can be represented as a computational graph.

For example:

```
x
│
↓
× w
│
↓
+ b
│
↓
Activation
│
↓
Loss
```

Each node represents an operation.

During the forward pass:

```
Input → Operations → Loss
```

During the backward pass:

```
Loss → Derivatives → Gradients
```

Modern deep learning frameworks such as PyTorch use computational graphs to automatically calculate gradients.

---

## 22. Automatic Differentiation

In practical deep learning, we usually do not manually calculate every derivative.

Frameworks such as PyTorch provide automatic differentiation.

For example:

```
loss.backward()
```

calculates the gradients of the loss with respect to parameters that require gradients.

Conceptually:

```
Forward Pass
      ↓
Computational Graph
      ↓
Loss
      ↓
loss.backward()
      ↓
Gradients
```

The underlying mathematics is still based on the chain rule.

---

## 23. PyTorch Example

A simplified PyTorch training step looks like:

```
optimizer.zero_grad()
output = model(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
```

Each line has a specific role.

1. Clear Previous Gradients

```
optimizer.zero_grad()
```

2. Forward Propagation

```
output = model(input)
```

3. Calculate Loss

```
loss = criterion(output, target)
```

4. Backpropagation

```
loss.backward()
```

5. Update Parameters

```
optimizer.step()
```

---

## 24. Why Do We Clear Gradients?

In PyTorch, gradients accumulate by default.

Therefore, before calculating gradients for the next optimization step, we usually clear the previous gradients:

```
optimizer.zero_grad()
```

Otherwise, the new gradients would be added to the old ones.

The simplified cycle is:

```
Clear Gradients
      ↓
Forward
      ↓
Loss
      ↓
Backward
      ↓
Update
      ↓
Repeat
```

---

## 25. Gradient Accumulation

Suppose:

\[
g_1
\]

is the gradient from one operation and:

\[
g_2
\]

is calculated later.

Without clearing:

\[
g_{\text{stored}}=g_1+g_2
\]

This can be useful in some specialized training strategies, but in the standard training loop we usually want fresh gradients for each optimization step.

---

## 26. Backpropagation in a Deep Network

Consider:

```
Input
 ↓
Layer 1
 ↓
Layer 2
 ↓
Layer 3
 ↓
Layer 4
 ↓
Output
 ↓
Loss
```

The backward pass is:

```
Loss
 ↓
Layer 4 Gradient
 ↓
Layer 3 Gradient
 ↓
Layer 2 Gradient
 ↓
Layer 1 Gradient
```

Each layer passes gradient information to the preceding layer.

This allows parameters throughout the network to be trained.

---

## 27. Why Deep Networks Can Have Gradient Problems

As gradients pass through many layers, the chain rule involves many multiplications.

Conceptually:

\[
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial a_n} \frac{\partial a_n}{\partial a_{n-1}} \dots \frac{\partial a_1}{\partial x}
\]

If many of these derivatives are smaller than 1, their product can become extremely small.

This leads to:

> Vanishing Gradients

---

## 28. Vanishing Gradients

Consider:

```
Gradient = 1
     ↓
× 0.5
     ↓
0.5
     ↓
× 0.5
     ↓
0.25
     ↓
× 0.5
     ↓
0.125
     ↓
...
```

After many layers, the gradient can become very small.

As a result:

```
Early Layers
     ↓
Tiny Gradients
     ↓
Tiny Updates
     ↓
Slow Learning
```

This is one reason why very deep networks can be difficult to train.

---

## 29. Exploding Gradients

The opposite can happen when derivatives are repeatedly larger than 1.

For example:

```
Gradient = 1
     ↓
× 2
     ↓
2
     ↓
× 2
     ↓
4
     ↓
× 2
     ↓
8
     ↓
...
```

The gradient can become extremely large.

This is called:

> Exploding Gradients

It can lead to unstable training.

---

## 30. Gradient Flow

The movement of gradients through the network is called gradient flow.

Conceptually:

**Forward:**

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

**Backward:**

```
Input
 ↑
Layer 1
 ↑
Layer 2
 ↑
Layer 3
 ↑
Loss
```

Good gradient flow is important for effective training.

---

## 31. Skip Connections

Very deep networks can improve gradient flow using skip connections.

A skip connection allows the input to bypass one or more layers.

For example:

```
┌─────────────────┐
             │                 │
Input ───────┤ Residual Branch │
  │          │                 │
  │          └────────┬────────┘
  │                   │
  └──────────────────→ Add
                        ↓
                      Output
```

The output is:

\[
y=F(x)+x
\]

where:

- \(F(x)\) = residual branch
- \(x\) = shortcut

---

## 32. Why Skip Connections Help Gradients

During backpropagation, the residual block provides an additional path through which gradients can flow.

Instead of forcing the gradient to pass only through every transformation:

```
Loss
 ↓
Layer
 ↓
Layer
 ↓
Layer
```

there is an additional shortcut:

```
┌───────────────┐
             │               ↓
Loss ───────→ Residual Path
             │
             └──────────────→ Shortcut
```

The shortcut can help gradients reach earlier layers more directly.

This is one of the key ideas behind ResNet.

---

## 33. Backpropagation in ResNet

A simplified residual block is:

\[
y=F(x)+x
\]

During backpropagation:

\[
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \left( \frac{\partial F(x)}{\partial x}+1 \right)
\]

The important term is:

\[
+1
\]

coming from the identity shortcut.

This creates a direct gradient path through the residual connection.

---

## 34. Backpropagation and ResNet

The conceptual relationship is:

```
Deep Neural Networks
        ↓
Backpropagation
        ↓
Gradient Flow
        ↓
Vanishing / Exploding Gradients
        ↓
Optimization Difficulty
        ↓
Residual Learning
        ↓
Skip Connections
        ↓
Improved Gradient Flow
```

Understanding backpropagation is therefore essential for understanding why ResNet uses skip connections.

---

## 35. Backpropagation Is Not Learning by Itself

A common misconception is:

> Backpropagation updates the weights.

More precisely:

```
Backpropagation
      ↓
Computes Gradients
      ↓
Optimizer
      ↓
Updates Weights
```

Backpropagation calculates the information required for learning.

The optimizer performs the parameter update.

---

## 36. Backpropagation vs Forward Propagation

**Forward Propagation**

Answers:

> What prediction does the network make?

```
Input
 ↓
Network
 ↓
Prediction
```

**Backpropagation**

Answers:

> How should the parameters change to reduce the loss?

```
Loss
 ↓
Gradients
 ↓
Parameters
```

Together:

```
Forward
  ↓
Prediction
  ↓
Loss
  ↓
Backward
  ↓
Gradients
  ↓
Optimizer
```

---

## 37. Backpropagation vs Training

Backpropagation is not the entire training process.

Training includes:

```
Forward Propagation
+
Loss Calculation
+
Backpropagation
+
Optimization
```

Therefore:

> Backpropagation is a fundamental component of neural network training, not the complete training algorithm.

---

## 38. Important Terms

- **Forward Propagation**: Passes data from input to output.
- **Loss**: Measures prediction error.
- **Backpropagation**: Calculates gradients using the chain rule.
- **Gradient**: Measures how the loss changes with respect to a parameter.
- **Optimizer**: Uses gradients to update parameters.
- **Learning Rate**: Controls the size of parameter updates.
- **Gradient Flow**: Describes how gradients propagate through the network.
- **Vanishing Gradient**: Gradients become extremely small.
- **Exploding Gradient**: Gradients become extremely large.

---

## 39. Complete Training Cycle

The complete process can be summarized as:

```
Training Step

                     Input
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
              Parameter Updates
                       ↓
                    Repeat
```

---

## 40. Conceptual Example

Suppose a neural network predicts:

- True Label: 1
- Prediction: 0.30

The model produces a relatively large loss.

Backpropagation determines:

> Which parameters contributed to the error?

It calculates:

\[
\frac{\partial L}{\partial W_1}
\]
\[
\frac{\partial L}{\partial W_2}
\]
\[
\frac{\partial L}{\partial W_3}
\]
...

The optimizer then updates the parameters.

After many training iterations, the model may produce:

- Prediction: 0.85

for the same type of example.

---

## 41. Backpropagation and Representation Learning

Backpropagation does more than train the final classifier.

Because gradients propagate through the entire network, earlier layers can also learn useful representations.

Conceptually:

```
Raw Input
   ↓
Early Layers
   ↓
Low-Level Features
   ↓
Middle Layers
   ↓
Intermediate Features
   ↓
Deep Layers
   ↓
High-Level Representation
   ↓
Classifier
```

The loss at the output influences parameter updates throughout the network.

This is one of the key mechanisms behind representation learning.

---

## 42. Backpropagation in CNNs

Backpropagation also works with convolutional neural networks.

A CNN may contain:

```
Image
 ↓
Convolution
 ↓
Activation
 ↓
Pooling
 ↓
Convolution
 ↓
Activation
 ↓
Classifier
 ↓
Loss
```

During backpropagation, gradients flow backward through:

```
Loss
 ↓
Classifier
 ↓
Pooling
 ↓
Activation
 ↓
Convolution
 ↓
Earlier Layers
```

The convolutional filters are updated based on the gradients.

---

## 43. Learning Convolutional Filters

Initially, convolutional filters contain learned parameters.

During training:

```
Random / Initialized Filters
          ↓
Forward Propagation
          ↓
Prediction
          ↓
Loss
          ↓
Backpropagation
          ↓
Filter Gradients
          ↓
Optimizer
          ↓
Updated Filters
```

After many iterations, filters can learn useful visual patterns.

For example, early filters may respond to:

- Edges
- Textures
- Simple Patterns

while deeper layers can represent more complex structures.

---

## 44. Backpropagation in ResNet50

In ResNet50, the network contains many convolutional and residual layers.

A simplified training process is:

```
MRI / Image
    ↓
Initial Convolution
    ↓
Residual Stages
    ↓
Global Average Pooling
    ↓
Fully Connected Layer
    ↓
Prediction
    ↓
Loss
    ↓
Backpropagation
    ↓
Gradients through Residual Blocks
    ↓
Optimizer
    ↓
Updated Parameters
```

The skip connections provide additional paths for gradient propagation.

---

## 45. Key Mathematical Formula

The core idea of backpropagation is the chain rule:

\[
\boxed{
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x}
}
\]

For a deep network, this principle is applied repeatedly:

\[
\boxed{
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z_n} \frac{\partial z_n}{\partial z_{n-1}} \cdots \frac{\partial z_1}{\partial x}
}
\]

This allows the gradient to propagate from the output back toward the input.

---

## 46. Key Takeaways

The most important concepts are:

1. Backpropagation is used to calculate gradients.
2. It is based on the chain rule.
3. Gradients flow backward from the loss toward earlier layers.
4. Backpropagation does not directly update weights.
5. An optimizer uses the gradients to update parameters.
6. Forward propagation calculates predictions.
7. The loss measures prediction error.
8. Backpropagation determines how parameters contributed to that error.
9. Very deep networks can suffer from vanishing or exploding gradients.
10. Skip connections can provide additional paths for gradient flow.
11. ResNet uses residual connections to make very deep networks easier to optimize.

---

## 47. Final Conceptual Diagram

### FORWARD PASS

```
Input
  │
  ↓
Layer 1
  │
  ↓
Layer 2
  │
  ↓
Layer 3
  │
  ↓
Output
  │
  ↓
Prediction
  │
  ↓
Loss
  │
  │
  │
  ↓
```

### BACKWARD PASS

```
Gradients
     ↑
     │
Layer 3
     ↑
     │
Layer 2
     ↑
     │
Layer 1
     ↑
     │
   Input
```

```
     ↓
   Optimizer
     ↓
Updated Parameters
```

The central idea is:

> Backpropagation efficiently uses the chain rule to propagate error information backward through a neural network and calculate how each learnable parameter should change to reduce the loss.

