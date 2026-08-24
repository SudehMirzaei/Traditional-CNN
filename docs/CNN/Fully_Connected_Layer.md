# Fully Connected Layer

## 1. Introduction

A **Fully Connected Layer (FC Layer)**, also called a **Dense Layer**, is a neural network layer in which every neuron is connected to every value in the input.

Fully Connected layers are traditionally used near the end of a CNN to transform learned visual features into outputs suitable for classification.

A simplified CNN structure is:

```
Input Image
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
Feature Maps
     ↓
Fully Connected Layer
     ↓
Classification
```

The convolutional part of the network learns visual representations, while the fully connected part uses those representations to make a final prediction.

---

## 2. What Does "Fully Connected" Mean?

Suppose a layer has:

4 Input Values

and:

3 Neurons

Every neuron is connected to every input:

```
x₁ ─────┬────→ Neuron 1
x₂ ─────┤
x₃ ─────┤
x₄ ─────┘

x₁ ─────┬────→ Neuron 2
x₂ ─────┤
x₃ ─────┤
x₄ ─────┘

x₁ ─────┬────→ Neuron 3
x₂ ─────┤
x₃ ─────┤
x₄ ─────┘
```

Therefore, each neuron receives information from all input values.

This is why the layer is called:

> Fully Connected

---

## 3. A Single Neuron

A neuron in a fully connected layer performs a weighted sum followed by an activation function.

Suppose the input is:

\[
x=[x_1,x_2,x_3]
\]

The neuron has weights:

\[
w=[w_1,w_2,w_3]
\]

and a bias:

\[
b
\]

The neuron first calculates:

\[
z=w_1x_1+w_2x_2+w_3x_3+b
\]

Then an activation function can be applied:

\[
a=f(z)
\]

So the process is:

Input  
  ↓  
Multiply by Weights  
  ↓  
Weighted Sum  
  ↓  
Add Bias  
  ↓  
Activation Function  
  ↓  
Output

---

## 4. Example

Suppose:

\[
x=[2,3,1]
\]

and:

\[
w=[0.5,0.2,0.8]
\]

with:

\[
b=1
\]

The neuron calculates:

\[
z=(2)(0.5)+(3)(0.2)+(1)(0.8)+1
\]

\[
z=1+0.6+0.8+1
\]

\[
z=3.4
\]

If ReLU is used:

\[
ReLU(3.4)=3.4
\]

Therefore, the neuron outputs:

\[
a=3.4
\]

---

## 5. Multiple Neurons

A fully connected layer usually contains many neurons.

Suppose we have:

Input:  
4 values

Output:  
3 neurons

Conceptually:

```
┌──→ Neuron 1
                 │
x₁ ──────────────┤
x₂ ──────────────┤
x₃ ──────────────┤
x₄ ──────────────┤
                 │
                 ├──→ Neuron 2
                 │
                 └──→ Neuron 3
```

Each neuron has its own weights and bias.

Therefore:

Neuron 1 → learns one weighted combination  
Neuron 2 → learns another weighted combination  
Neuron 3 → learns another weighted combination

---

## 6. Matrix Representation

Instead of calculating every neuron separately, we can represent the entire fully connected layer using matrix multiplication.

The basic equation is:

\[
y=Wx+b
\]

where:

- \( x \) = input vector
- \( W \) = weight matrix
- \( b \) = bias vector
- \( y \) = output vector

For example:

Input size  = 4  
Output size = 3

Then:

\[
W\in\mathbb{R}^{3\times4}
\]

and:

\[
x\in\mathbb{R}^{4}
\]

Therefore:

\[
Wx
\]

produces:

\[
y\in\mathbb{R}^{3}
\]

---

## 7. Number of Parameters

Suppose a fully connected layer has:

Input Features = 4  
Output Neurons = 3

Each output neuron needs 4 weights.

Therefore:

\[
4\times3=12
\]

weights are required.

Each neuron also has one bias:

\[
3
\]

biases.

Therefore, the total number of trainable parameters is:

\[
12+3=15
\]

So:

\[
\boxed{
Parameters=(Input\ Size\times Output\ Size)+Output\ Size
}
\]

or:

\[
\boxed{
Parameters=(n_{in}+1)\times n_{out}
}
\]

---

## 8. Fully Connected Layer in CNNs

In a CNN, the input to the fully connected layer is usually a learned feature representation produced by convolutional layers.

For example:

```
Input Image
224 × 224 × 3
       ↓
Convolution
       ↓
Feature Maps
       ↓
Deeper Convolution
       ↓
Deep Feature Maps
       ↓
Fully Connected Layer
       ↓
Classification
```

The convolutional layers are responsible for learning visual features.

The fully connected layer uses those learned features for prediction.

---

## 9. The Problem: Feature Maps Are 3D

Suppose the convolutional part of a CNN produces:

\[
7 \times 7 \times 512
\]

This means:

7 × 7 Spatial Dimensions  
×  
512 Feature Maps

A fully connected layer expects a vector rather than a 3D tensor.

Therefore, we need to transform:

\[
7\times7\times512
\]

into a one-dimensional vector.

This process is called:

> Flattening

---

## 10. Flattening

Suppose we have:

\[
7 \times 7 \times 512
\]

The total number of values is:

\[
7\times7\times512
\]

\[
=25088
\]

Therefore, flattening produces:

25088-dimensional vector

Conceptually:

```
7 × 7 × 512
      ↓
   Flatten
      ↓
25088
```

The actual values are not changed.

Only their arrangement is changed from a multidimensional tensor into a vector.

---

## 11. Example of Flattening

Suppose a feature map is:

\[
2 \times 2 \times 3
\]

There are:

\[
2\times2\times3=12
\]

values.

Before flattening:

Feature Map 1

```
[ a  b ]
[ c  d ]
```

Feature Map 2

```
[ e  f ]
[ g  h ]
```

Feature Map 3

```
[ i  j ]
[ k  l ]
```

After flattening:

```
[a,b,c,d,e,f,g,h,i,j,k,l]
```

Now the CNN has a:

12-dimensional vector

---

## 12. Connecting Feature Maps to Neurons

Suppose the flattened representation contains:

25088 values

and the fully connected layer contains:

512 neurons

Then every neuron receives all 25088 values.

Conceptually:

```
25088 Features
      │
      ├────────→ Neuron 1
      ├────────→ Neuron 2
      ├────────→ Neuron 3
      │
      │
      └────────→ Neuron 512
```

The layer therefore performs:

\[
25088\rightarrow512
\]

---

## 13. Parameter Explosion

A major problem with flattening large feature maps is the number of parameters.

Suppose:

Feature Maps:  
\( 7 \times 7 \times 512 \)

Fully Connected:  
512 neurons

The number of weights is:

\[
25088\times512
\]

which equals:

\[
12,845,056
\]

weights.

Adding 512 biases gives:

\[
12,845,568
\]

trainable parameters.

This is a very large number.

Therefore, modern CNN architectures often avoid connecting large feature maps directly to huge fully connected layers.

---

## 14. Global Average Pooling

A common alternative is Global Average Pooling (GAP).

Suppose the feature representation is:

\[
7 \times 7 \times 512
\]

Global Average Pooling averages all spatial values within each feature map.

Therefore:

```
7 × 7 × 512
       ↓
Global Average Pooling
       ↓
512
```

Each feature map becomes one value.

For example:

Feature Map 1 → Average → Value 1  
Feature Map 2 → Average → Value 2  
Feature Map 3 → Average → Value 3  
...  
Feature Map 512 → Average → Value 512

The result is:

512-dimensional vector

---

## 15. Fully Connected Layer After Global Average Pooling

Now suppose we have:

512 Features

and:

4 Output Classes

The final fully connected layer is:

\[
512\rightarrow4
\]

The number of weights is:

\[
512\times4=2048
\]

and the number of biases is:

\[
4
\]

Therefore:

\[
2048+4=2052
\]

trainable parameters.

Compare this with:

\[
7 \times 7 \times 512 \rightarrow 512
\]

which required more than 12.8 million parameters.

Global Average Pooling can therefore dramatically reduce the number of parameters.

---

## 16. Fully Connected Layer as a Classifier

In a classification CNN, the final fully connected layer often produces one value per class.

Suppose there are four classes:

Class 1  
Class 2  
Class 3  
Class 4  

The final FC layer produces:

\[
[2.1, -0.5, 1.7, 0.2]
\]

These values are called:

> Logits

The highest logit corresponds to the class with the strongest predicted score.

For example:

Class 1 → 2.1  
Class 2 → -0.5  
Class 3 → 1.7  
Class 4 → 0.2  

The largest value is:

\[
2.1
\]

Therefore:

Predicted Class = Class 1

---

## 17. Fully Connected Layer and Softmax

For multi-class classification, Softmax can be applied to the final logits.

The pipeline becomes:

```
Feature Representation
        ↓
Fully Connected Layer
        ↓
Logits
        ↓
Softmax
        ↓
Class Probabilities
```

For example:

Logits:

\[
[2.1, -0.5, 1.7, 0.2]
\]

        ↓ Softmax

Probabilities:

\[
[0.56, 0.04, 0.31, 0.09]
\]

The probabilities sum to 1.

---

## 18. Fully Connected Layer Does Not Extract Local Features

This is an important distinction between convolution and fully connected layers.

A convolutional layer looks at local spatial neighborhoods.

For example:

3 × 3 Kernel

The same kernel is moved across the image.

Therefore, convolution is designed to detect spatial patterns.

A fully connected layer behaves differently.

After flattening:

```
Feature Vector
      ↓
Fully Connected Layer
      ↓
Every neuron sees every feature
```

The FC layer does not preserve the original spatial arrangement in the same way convolution does.

---

## 19. Convolution vs Fully Connected Layer

| Property                     | Convolution        | Fully Connected   |
|------------------------------|--------------------|--------------------|
| Connections                   | Local              | All-to-all         |
| Spatial structure             | Preserved          | Mostly removed after flattening |
| Weight sharing                | Yes                | No                 |
| Main role                    | Feature extraction  | Feature combination / classification |
| Parameters                    | Usually fewer      | Can be very large  |
| Input                         | Feature maps / images | Usually a vector   |
| Spatial awareness             | Strong             | Limited            |

---

## 20. Why CNNs Use Convolution Before Fully Connected Layers

Consider an image classification task.

The network can progressively learn:

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
High-Level Features
  ↓
Classification
```

Convolutional layers perform most of the feature extraction:

```
Image
 ↓
Conv
 ↓
Conv
 ↓
Conv
 ↓
Deep Feature Representation
```

Then the fully connected layer maps this representation to the target classes:

```
Deep Features
     ↓
Fully Connected Layer
     ↓
Class Scores
```

---

## 21. Fully Connected Layer and Representation Learning

The convolutional layers learn a representation of the input.

For example:

```
Original Image
       ↓
Low-Level Features
       ↓
Mid-Level Features
       ↓
High-Level Features
       ↓
Feature Vector
```

The fully connected layer then learns how to combine these features for classification.

For example:

```
Feature 1 ─────┐
Feature 2 ─────┤
Feature 3 ─────┤
Feature 4 ─────┼──→ Classifier
Feature 5 ─────┤
Feature 6 ─────┘
```

The classifier learns which combinations of features are associated with each class.

---

## 22. Example: Skin Lesion Classification

Suppose a CNN receives a dermoscopic image.

The convolutional layers may learn features related to:

- Edges
- Textures
- Colors
- Shapes
- Patterns
- Lesion Structures

The deeper layers combine these into high-level representations.

Finally:

```
Image
 ↓
Convolutional Layers
 ↓
Deep Feature Representation
 ↓
Global Average Pooling
 ↓
Feature Vector
 ↓
Fully Connected Layer
 ↓
Class Scores
```

For example:

- Melanoma
- Nevus
- Basal Cell Carcinoma
- Actinic Keratosis

The final FC layer learns how the extracted features relate to these classes.

---

## 23. Example: Brain MRI Classification

The same idea applies to brain MRI classification.

A CNN might progressively learn:

```
MRI
 ↓
Edges
 ↓
Textures
 ↓
Anatomical Structures
 ↓
Abnormal Patterns
 ↓
High-Level Brain Representation
 ↓
Fully Connected Layer
 ↓
Tumor Classes
```

The FC layer does not manually identify the tumor.

Instead, it learns a mapping from the learned feature representation to the target classes.

---

## 24. Fully Connected Layers in ResNet50

ResNet50 uses a different strategy from older CNN architectures that relied on a large flattening operation.

Near the end of ResNet50:

```
Deep Feature Maps
2048 × 7 × 7
       ↓
Global Average Pooling
       ↓
2048
       ↓
Fully Connected Layer
       ↓
Number of Classes
```

For ImageNet, the final classification layer is:

\[
2048 \rightarrow 1000
\]

because ImageNet contains 1000 classes.

If ResNet50 is adapted to a four-class classification problem, the final layer can be changed to:

\[
2048 \rightarrow 4
\]

For example:

```python
model.fc = nn.Linear(2048, 4)
```

---

## 25. Why ResNet50 Uses Global Average Pooling

Suppose ResNet50 produces:

\[
7 \times 7 \times 2048
\]

Directly flattening this would produce:

\[
7\times7\times2048=100352
\]

features.

A fully connected layer mapping:

\[
100352 \rightarrow 4
\]

would require:

\[
100352\times4=401408
\]

weights.

Instead, ResNet50 uses Global Average Pooling:

```
7 × 7 × 2048
       ↓
Global Average Pooling
       ↓
2048
       ↓
Fully Connected
       ↓
4 Classes
```

Now the FC layer requires only:

\[
2048\times4=8192
\]

weights, plus 4 biases.

This is significantly more parameter-efficient.

---

## 26. Fully Connected Layer in Transfer Learning

In transfer learning, a pretrained CNN is often adapted to a new classification task.

For example, suppose a pretrained ResNet50 was originally trained for:

1000 classes

Its final layer is:

\[
2048 \rightarrow 1000
\]

For a new four-class problem, we can replace the final layer:

\[
2048 \rightarrow 4
\]

The earlier convolutional layers can retain their pretrained weights.

Conceptually:

```
Pretrained CNN
      ↓
Learned Feature Extractor
      ↓
Replace Final FC Layer
      ↓
New Classifier
```

This is one of the key ideas behind transfer learning.

---

## 27. Fully Connected Layer and Trainable Parameters

The weights of the FC layer are learnable.

During training:

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
Weight Updates
```

The optimizer updates:

\[
W
\]

and:

\[
b
\]

so that the classifier becomes better at mapping the learned features to the correct classes.

---

## 28. Fully Connected Layer and Backpropagation

Suppose:

\[
y=Wx+b
\]

During backpropagation, gradients are calculated for:

\[
W
\]

and:

\[
b
\]

The optimizer then updates them.

Conceptually:

```
Prediction
    ↓
Loss
    ↓
Backpropagation
    ↓
∂L/∂W
∂L/∂b
    ↓
Optimizer
    ↓
Updated W and b
```

Therefore, the FC layer learns during training just like convolutional filters do.

---

## 29. Fully Connected Layer Does Learn

It is important to understand that the FC layer is not simply a fixed mathematical operation.

Its:

- Weights
- Biases

are trainable parameters.

During training:

```
Initial Weights
      ↓
Forward Pass
      ↓
Loss
      ↓
Backpropagation
      ↓
Gradient
      ↓
Optimizer
      ↓
Updated Weights
```

The FC layer gradually learns which feature combinations are useful for classification.

---

## 30. A Complete CNN Pipeline

A traditional CNN can be summarized as:

```
Input Image
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
Pooling
     ↓
Deep Feature Maps
     ↓
Flatten
     ↓
Fully Connected Layer
     ↓
Output Layer
     ↓
Softmax
     ↓
Class Probabilities
```

Modern architectures such as ResNet often replace the large Flatten + FC design with:

```
Deep Feature Maps
     ↓
Global Average Pooling
     ↓
Feature Vector
     ↓
Fully Connected Layer
     ↓
Class Scores
```

---

## 31. Key Differences: Flatten vs Global Average Pooling

**Flatten**

```
7 × 7 × 512
     ↓
Flatten
     ↓
25088
```

Flatten preserves all individual values but creates a large vector.

**Global Average Pooling**

```
7 × 7 × 512
     ↓
Global Average Pooling
     ↓
512
```

GAP summarizes each feature map into a single value.

Therefore:

- Flatten → Keeps all spatial values
- GAP → Aggregates spatial information

---

## 32. Important Conceptual Picture

The overall relationship can be visualized as:

```
CNN
                   │
        ┌──────────┴──────────┐
        │                     │
Feature Extraction       Classification
        │                     │
        ↓                     ↓
Convolutional Layers     Fully Connected
        │                     │
        ↓                     ↓
Feature Maps             Class Scores
        │                     │
        └──────────┬──────────┘
                   ↓
              Prediction
```

The convolutional layers answer:

> What visual features exist in the image?

The fully connected classifier answers:

> Given these features, which class is most likely?

---

## 33. Key Takeaways

- A Fully Connected Layer is also called a Dense Layer.
- Every neuron is connected to every input value.
- A neuron computes a weighted sum plus bias.
- The basic equation is:

\[
y=Wx+b
\]

- Fully connected layers contain trainable weights and biases.
- In traditional CNNs, feature maps are often flattened before entering the FC layer.
- Flattening converts a tensor into a one-dimensional vector.
- Large flattened feature maps can create a huge number of parameters.
- Modern CNNs often use Global Average Pooling before the final FC layer.
- ResNet50 uses Global Average Pooling before its final fully connected layer.
- ResNet50 produces a 2048-dimensional vector after Global Average Pooling.
- The final FC layer maps this representation to the desired number of classes.
  
For a four-class problem:

\[
2048\rightarrow4
\]

- The FC layer learns how to combine high-level features for classification.
- Convolutional layers primarily learn spatial and visual representations.
- Fully connected layers primarily perform global feature combination and classification.
- The FC layer is trained using backpropagation and gradient-based optimization.

---

## 34. Conceptual Summary

The role of the Fully Connected Layer in a CNN can be summarized as:

```
Input Image
     ↓
Convolutional Layers
     ↓
Learned Feature Maps
     ↓
Global Average Pooling / Flatten
     ↓
Feature Vector
     ↓
Fully Connected Layer
     ↓
Class Scores
     ↓
Softmax
     ↓
Class Probabilities
```

The fundamental idea is:

> Convolutional layers learn useful visual representations, while the Fully Connected Layer learns how to combine those representations to make the final classification decision.

