# Gradient Descent

## 1. Introduction

**Gradient Descent** is an optimization algorithm used to minimize the loss function of a machine learning model.

In neural networks, the model contains many learnable parameters such as:

- Weights
- Biases

During training, we want to find parameter values that produce a smaller prediction error.

Gradient Descent provides a systematic way to update these parameters.

The basic training process is:

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
Gradient Descent / Optimizer
  ↓
Updated Parameters
```

---

## 2. Why Do We Need Gradient Descent?

Suppose a neural network makes a prediction that is different from the correct answer.

For example:

- True Label: 1
- Prediction: 0.30

The model has an error.

The loss function measures this error:

\[
L = \text{Loss}(y,\hat{y})
\]

The goal of training is:

\[
\boxed{\min_{\theta} L(\theta)}
\]

where:

- \(L\) = loss
- \(\theta\) = model parameters

The parameters may include millions of weights and biases.

Gradient Descent provides a method for changing these parameters in a direction that reduces the loss.

---

## 3. The Basic Idea

The central idea is simple:

> Find the direction in which the loss increases, then move the parameters in the opposite direction.

The gradient tells us the direction of increasing loss.

Therefore, Gradient Descent moves in the opposite direction.

Conceptually:

```
Gradient
   ↓
Direction of increasing loss
   ↓
Move in the opposite direction
   ↓
Lower Loss
```

---

## 4. What Is a Gradient?

For a parameter (\(w\)), the gradient is:

\[
\frac{\partial L}{\partial w}
\]

It tells us how sensitive the loss is to changes in (\(w\)).

For example:

\[
\frac{\partial L}{\partial w}=5
\]

means that increasing (\(w\)) tends to increase the loss.

If:

\[
\frac{\partial L}{\partial w}=-5
\]

increasing (\(w\)) tends to decrease the loss.

---

## 5. Gradient Descent Update Rule

The fundamental update rule is:

\[
\boxed{
w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial L}{\partial w}
}
\]

where:

- \(w\) = parameter
- \(L\) = loss
- \(\frac{\partial L}{\partial w}\) = gradient
- \(\eta\) = learning rate

The minus sign is important.

It means that the parameter moves in the opposite direction of the gradient.

---

## 6. Why Do We Subtract the Gradient?

Suppose:

\[
\frac{\partial L}{\partial w}=4
\]

The gradient is positive.

This means increasing (\(w\)) tends to increase the loss.

Therefore, we want to decrease (\(w\)).

Gradient Descent does exactly that:

\[
w_{\text{new}} = w_{\text{old}} - \eta(4)
\]

The parameter moves toward a region of lower loss.

---

## 7. Simple Numerical Example

Suppose:

\[
w=0.8
\]

and:

\[
\frac{\partial L}{\partial w}=2
\]

with learning rate:

\[
\eta=0.1
\]

Then:

\[
w_{\text{new}} = 0.8 - (0.1)(2)
\]

\[
w_{\text{new}}=0.6
\]

Therefore:

```
Old Weight
    ↓
   0.8
    ↓
Gradient Descent
    ↓
   0.6
    ↓
New Weight
```

---

## 8. Example With a Negative Gradient

Suppose:

\[
w=0.8
\]

and:

\[
\frac{\partial L}{\partial w}=-2
\]

with:

\[
\eta=0.1
\]

Then:

\[
w_{\text{new}} = 0.8 - (0.1)(-2)
\]

\[
w_{\text{new}}=1.0
\]

So:

```
0.8 → 1.0
```

The weight increased because the gradient was negative.

---

## 9. The Role of the Learning Rate

The learning rate determines how large each parameter update is.

It is usually represented by:

\[
\eta
\]

For example:

- **Small Learning Rate** → Small Parameter Updates
- **Large Learning Rate** → Large Parameter Updates

The learning rate is one of the most important hyperparameters in neural network training.

---

## 10. Learning Rate Too Small

Suppose the learning rate is extremely small:

\[
\eta=0.000001
\]

The updates may become very small.

Conceptually:

```
Loss
 ↓
 ↓
 ↓
 ↓
Very Slow Progress
```

Training may require a very large number of iterations.

---

## 11. Learning Rate Too Large

Now suppose:

\[
\eta=10
\]

The updates may become extremely large.

The optimizer might jump over good parameter values.

Conceptually:

```
Minimum
         ↓
   ↙         ↘
Large Jumps
```

This can cause unstable training.

The loss may oscillate or even increase.

---

## 12. Finding a Good Learning Rate

A suitable learning rate should allow the model to make meaningful progress without making excessively large updates.

Conceptually:

- **Too Small** → Slow Training
- **Good** → Stable Progress
- **Too Large** → Unstable Training

In practice, learning rates are usually selected experimentally or using learning-rate schedules and adaptive optimizers.

---

## 13. Loss Landscape

To understand Gradient Descent visually, imagine that the loss function creates a landscape.

For one parameter:

```
Loss
 ↑
 │       *
 │      / \
 │     /   \
 │    /     \
 │___/_______\____→ Parameter
        Minimum
```

The goal is to find a parameter value where the loss is minimized.

Gradient Descent moves the parameter toward a lower-loss region.

---

## 14. Local Minimum

A local minimum is a point where the loss is smaller than nearby points.

Conceptually:

```
Loss
 ↑
 │      \       /
 │       \_____/
 │         ↑
 │    Local Minimum
 └────────────────→ Parameter
```

Gradient Descent can converge toward a local minimum depending on the shape of the loss landscape and the optimization process.

---

## 15. Global Minimum

The global minimum is the lowest possible value of the loss over the relevant parameter space.

Conceptually:

```
Loss
 ↑
 │       \      /
 │        \    /
 │         \__/
 │           ↑
 │    Global Minimum
 └──────────────────→ Parameter
```

In modern deep neural networks, the loss landscape is extremely high-dimensional and much more complex than this simple one-dimensional illustration.

---

## 16. Gradient Descent in Multiple Dimensions

A neural network does not have just one parameter.

Suppose:

\[
\theta = [w_1, w_2, w_3, \dots, w_n]
\]

Then we calculate a gradient for every parameter:

\[
\nabla_\theta L =
\begin{bmatrix}
\frac{\partial L}{\partial w_1}\\
\frac{\partial L}{\partial w_2}\\
\frac{\partial L}{\partial w_3}\\
\vdots\\
\frac{\partial L}{\partial w_n}
\end{bmatrix}
\]

The update becomes:

\[
\boxed{
\theta_{\text{new}} = \theta_{\text{old}} - \eta\nabla_\theta L
}
\]

This is the general form of Gradient Descent.

---

## 17. Gradient Vector

The gradient is therefore not necessarily a single number.

For a neural network with many parameters, it is a vector containing the gradient of the loss with respect to every parameter.

For example:

```
Gradient
   ↓
[ 0.2,
 -0.5,
  0.1,
  0.8,
 -0.3,
   ... ]
```

Each value corresponds to a parameter.

---

## 18. Gradient Descent and Backpropagation

Gradient Descent and Backpropagation are closely related, but they are not the same thing.

### Backpropagation

Calculates:

\[
\nabla_\theta L
\]

### Gradient Descent

Uses that gradient:

\[
\theta \leftarrow \theta - \eta\nabla_\theta L
\]

Therefore:

```
Backpropagation
      ↓
Calculate Gradients
      ↓
Gradient Descent
      ↓
Update Parameters
```

This distinction is very important.

---

## 19. Complete Training Step

A single training iteration can be represented as:

```
Input Batch
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
Gradient Descent
    ↓
Parameter Update
```

This process is repeated many times.

---

## 20. Iterative Optimization

Suppose the initial parameter is:

\[
w_0
\]

Gradient Descent performs:

\[
w_1 = w_0 - \eta\frac{\partial L}{\partial w}
\]

Then another iteration:

\[
w_2 = w_1 - \eta\frac{\partial L}{\partial w}
\]

and so on.

Conceptually:

```
w₀
 ↓
w₁
 ↓
w₂
 ↓
w₃
 ↓
w₄
 ↓
...
 ↓
Low-Loss Region
```

---

## 21. Epochs and Gradient Descent

An epoch means processing the entire training dataset once.

For example:

```
Dataset
   ↓
Batch 1
   ↓
Parameter Update

Batch 2
   ↓
Parameter Update

Batch 3
   ↓
Parameter Update

...
```

After all training batches have been processed:

1 Epoch

Training usually involves many epochs.

---

## 22. Mini-Batch Gradient Descent

In neural networks, using the entire dataset for every update can be computationally expensive.

Instead, the dataset is divided into smaller batches.

For example:

```
10000 Training Samples
        ↓
Batch 1 → 32 samples
Batch 2 → 32 samples
Batch 3 → 32 samples
...
```

The model performs an update after processing each batch.

This approach is called Mini-Batch Gradient Descent.

---

## 23. Batch Gradient Descent

In standard Batch Gradient Descent, the gradient is calculated using the entire training dataset before making an update.

Conceptually:

```
Entire Dataset
      ↓
Forward Pass
      ↓
Loss
      ↓
Gradient
      ↓
One Parameter Update
```

**Advantages:**

- Stable gradient estimate
- Deterministic update for a fixed dataset

**Disadvantages:**

- Can be computationally expensive
- Requires more memory
- Updates are less frequent

---

## 24. Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) uses a single training example for each update.

Conceptually:

```
Sample 1
   ↓
Gradient
   ↓
Update

Sample 2
   ↓
Gradient
   ↓
Update

Sample 3
   ↓
Gradient
   ↓
Update
```

**Advantages:**

- Frequent updates
- Can be computationally efficient
- Noise in updates can sometimes help exploration

**Disadvantages:**

- Updates can be noisy
- Loss may fluctuate significantly

---

## 25. Mini-Batch Gradient Descent

Mini-batch Gradient Descent lies between the two.

```
Batch Gradient Descent
      ↓
Entire Dataset

Mini-Batch Gradient Descent
      ↓
Small Batch

Stochastic Gradient Descent
      ↓
One Sample
```

Modern deep learning systems commonly use mini-batches.

For example, Batch Size = 32 means that the gradient is calculated from 32 training samples at a time.

---

## 26. Comparison

| Method              | Data per Update        | Main Characteristic               |
|---------------------|-----------------------|------------------------------------|
| Batch GD            | Entire Dataset        | Stable but expensive                |
| SGD                 | 1 Sample              | Fast but noisy                     |
| Mini-Batch GD       | Small Batch           | Efficient and practical            |

---

## 27. Gradient Descent in Neural Networks

Consider a neural network:

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

Suppose the network contains:

\[
\theta = \{W_1, b_1, W_2, b_2, W_3, b_3\}
\]

Backpropagation calculates:

\[
\frac{\partial L}{\partial W_1}
\]

\[
\frac{\partial L}{\partial b_1}
\]

\[
\frac{\partial L}{\partial W_2}
\]

\[
\frac{\partial L}{\partial b_2}
\]

and so on.

Gradient Descent then updates all these parameters.

---

## 28. Weight Update

For a weight:

\[
W
\]

the update is:

\[
W_{\text{new}} = W_{\text{old}} - \eta \frac{\partial L}{\partial W}
\]

For a bias:

\[
b
\]

the update is:

\[
b_{\text{new}} = b_{\text{old}} - \eta \frac{\partial L}{\partial b}
\]

Therefore, both weights and biases can be optimized.

---

## 29. Gradient Descent in CNNs

CNNs also use Gradient Descent to update convolutional filters.

Suppose a convolutional filter contains weights:

```
[ w₁  w₂  w₃ ]
[ w₄  w₅  w₆ ]
[ w₇  w₈  w₉ ]
```

During training:

```
Input Image
    ↓
Convolution
    ↓
Feature Maps
    ↓
Prediction
    ↓
Loss
    ↓
Backpropagation
    ↓
Gradients for Filter Weights
    ↓
Gradient Descent
    ↓
Updated Filter
```

The filter values are therefore learned rather than manually specified.

---

## 30. Gradient Descent and Feature Learning

In traditional feature engineering, humans might manually define features.

For example:

```
Image
  ↓
Hand-Crafted Features
  ↓
Classifier
```

In deep learning:

```
Image
  ↓
CNN
  ↓
Learned Features
  ↓
Classifier
```

Gradient-based optimization allows the network to learn these features automatically.

The loss provides the training signal, backpropagation computes gradients, and the optimizer updates the parameters.

---

## 31. Gradient Descent and ResNet

ResNet uses gradient-based optimization just like other neural networks.

A simplified ResNet training process is:

```
Input Image
     ↓
ResNet50
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
Updated ResNet Parameters
```

The important innovation of ResNet is not a replacement for Gradient Descent.

Instead, Residual Learning and Skip Connections make optimization of very deep networks easier.

---

## 32. Gradient Descent and Skip Connections

A residual block computes:

\[
y=F(x)+x
\]

During backpropagation, the shortcut provides an additional path for gradient flow.

Therefore:

```
Residual Learning
       ↓
Better Gradient Flow
       ↓
Easier Optimization
       ↓
Deep Networks Can Be Trained
```

Gradient Descent is still used to update the parameters.

---

## 33. Optimizers Beyond Basic Gradient Descent

Basic Gradient Descent is important conceptually, but modern neural networks often use more advanced optimizers.

Common examples include:

- SGD
- SGD with Momentum
- Adam
- AdamW
- RMSProp

These methods are based on the same fundamental idea:

> Use gradient information to improve the model parameters.

---

## 34. Momentum

Momentum adds information from previous updates.

A simplified formulation is:

\[
v_t = \beta v_{t-1} + (1-\beta)\nabla L
\]

Then:

\[
\theta_t = \theta_{t-1} - \eta v_t
\]

The idea is that the optimizer can maintain some memory of previous gradient directions.

This can help accelerate optimization and reduce oscillations.

---

## 35. Adam

Adam combines ideas related to momentum and adaptive learning rates.

It maintains estimates of:

- The first moment of gradients
- The second moment of gradients

Conceptually:

```
Gradient
   ↓
Adam
   ↓
Adaptive Parameter Update
```

Adam is widely used in deep learning.

---

## 36. AdamW

AdamW is a commonly used optimizer that separates weight decay from the adaptive gradient update.

A simplified conceptual training loop is:

```
Forward
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
```

For example, a ResNet training configuration might use:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)
```

---

## 37. Gradient Descent vs Optimizer

In practice, the word "optimizer" is often used instead of simply saying "Gradient Descent."

The relationship is:

```
Gradient-Based Optimization
        │
        ├── Gradient Descent
        ├── SGD
        ├── Momentum
        ├── Adam
        └── AdamW
```

These algorithms differ in how they use gradients to update parameters.

---

## 38. Learning Rate Scheduling

The learning rate does not always need to remain constant.

A learning rate scheduler changes the learning rate during training.

For example:

- Epoch 1: Learning Rate = 0.001
- Epoch 10: Learning Rate = 0.0005
- Epoch 20: Learning Rate = 0.0001

Conceptually:

```
Learning Rate
     ↓
Start Higher
     ↓
Gradually Reduce
     ↓
Fine-Tune Parameters
```

Learning-rate schedules can help optimization converge effectively.

---

## 39. Why Reduce the Learning Rate?

Early in training, the parameters may be far from a good solution.

Larger updates can help the model make rapid progress.

Later in training, smaller updates can help the model make finer adjustments.

Conceptually:

```
Early Training
      ↓
Larger Updates

Later Training
      ↓
Smaller Updates
```

---

## 40. Gradient Descent and Loss Curves

During successful training, the loss often decreases.

For example:

- Epoch 1  → Loss = 1.20
- Epoch 2  → Loss = 0.90
- Epoch 3  → Loss = 0.70
- Epoch 4  → Loss = 0.55
- Epoch 5  → Loss = 0.42

Conceptually:

```
Loss
 ↑
 │\
 │ \
 │  \
 │   \
 │    \____
 │
 └────────────→ Training
```

The exact behavior depends on the model, dataset, optimizer, and learning rate.

---

## 41. What Happens If the Loss Does Not Decrease?

If training loss remains high or unstable, possible causes include:

- Learning rate too large
- Learning rate too small
- Poor initialization
- Incorrect loss function
- Data preprocessing problems
- Gradient problems
- Model architecture issues

Gradient Descent itself may be mathematically correct while the training configuration is unsuitable.

---

## 42. Gradient Descent and Vanishing Gradients

When gradients become extremely small:

\[
\nabla_\theta L \approx 0
\]

the parameter updates become small:

\[
\Delta\theta = -\eta\nabla_\theta L
\]

Therefore:

> Vanishing Gradient
       ↓
Tiny Gradient
       ↓
Tiny Parameter Update
       ↓
Slow Learning

This is one reason why optimization becomes difficult in some very deep networks.

---

## 43. Gradient Descent and Exploding Gradients

If gradients become extremely large:

\[
\|\nabla_\theta L\| \gg 1
\]

then:

\[
\Delta\theta = -\eta\nabla_\theta L
\]

can also become extremely large.

This can cause:

> Exploding Gradient
       ↓
Huge Updates
       ↓
Unstable Training
       ↓
Loss May Become Very Large

---

## 44. Gradient Clipping

One technique for controlling very large gradients is gradient clipping.

Conceptually:

```
Large Gradient
      ↓
Gradient Clipping
      ↓
Controlled Gradient
      ↓
Parameter Update
```

A common PyTorch example is:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

This can help prevent excessively large parameter updates.

---

## 45. Gradient Descent in PyTorch

A basic training loop using SGD might look like:

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)

for inputs, targets in dataloader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

The important sequence is:

```
zero_grad()
     ↓
Forward
     ↓
Loss
     ↓
backward()
     ↓
step()
```

---

## 46. What Does loss.backward() Do?

This command:

```
loss.backward()
```

performs backpropagation.

It calculates gradients such as:

\[
\frac{\partial L}{\partial W}
\]

for parameters involved in the computation graph.

It does not itself perform the final parameter update.

---

## 47. What Does optimizer.step() Do?

This command:

```
optimizer.step()
```

uses the gradients to update the model parameters.

Conceptually:

```
loss.backward()
       ↓
Gradients
       ↓
optimizer.step()
       ↓
Updated Parameters
```

Therefore:

```
backward() ≠ step()
```

They perform different operations.

---

## 48. Complete PyTorch Training Cycle

```python
for inputs, targets in dataloader:
    # Clear old gradients
    optimizer.zero_grad()

    # Forward propagation
    outputs = model(inputs)

    # Calculate loss
    loss = criterion(outputs, targets)

    # Backpropagation
    loss.backward()

    # Update parameters
    optimizer.step()
```

This loop is repeated for many batches and epochs.

---

## 49. Conceptual Relationship

The entire process can be summarized as:

```
Neural Network Training

                       Input
                         ↓
                 Forward Propagation
                         ↓
                     Prediction
                         ↓
                    Loss Function
                         ↓
                        Loss
                         ↓
                   Backpropagation
                         ↓
                      Gradients
                         ↓
                   Gradient Descent
                         ↓
                Updated Parameters
                         ↓
                       Repeat
```

---

## 50. Key Takeaways

The most important concepts are:

1. Gradient Descent is an optimization method.
2. Its goal is to minimize the loss function.
3. The gradient indicates the direction of increasing loss.
4. Gradient Descent moves in the opposite direction.
5. The basic update rule is:

\[
\theta_{\text{new}} = \theta_{\text{old}} - \eta\nabla_\theta L
\]

6. The learning rate controls the size of updates.
7. Backpropagation calculates the gradients.
8. The optimizer uses those gradients to update parameters.
9. Mini-batch training is commonly used in deep learning.
10. SGD, Adam, and AdamW are examples of gradient-based optimizers.
11. Learning-rate schedules can change the learning rate during training.
12. Vanishing gradients can make updates extremely small.
13. Exploding gradients can make updates extremely large.
14. ResNet uses residual connections to improve optimization and gradient flow in deep networks.

---

## 51. Final Conceptual Diagram

### TRAINING

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
                             ↓
                     Backpropagation
                             ↓
                         Gradients
                             ↓
                    Gradient-Based Optimizer
                             ↓
                     Parameter Update
                             ↓
                     Updated Network
                             ↓
                           Repeat
```

The central idea is:

> Gradient Descent uses the gradients calculated by backpropagation to iteratively update the parameters of a neural network in a direction that tends to reduce the loss.

