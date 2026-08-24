# Convolution in CNNs

## 1. Introduction

**Convolution** is the core operation used in Convolutional Neural Networks (CNNs) to extract local patterns from input data.

The basic idea is simple:

> A small set of learnable weights moves across the input, performs mathematical operations with local regions, and produces an output called a **feature map**.

Conceptually:

```
Input
  ↓
Convolution
  ↓
Feature Map

Convolution allows a CNN to transform raw pixel values into increasingly useful representations.
```

---

## 2. Why Do We Need Convolution?

Images contain spatial structure.

For example, nearby pixels are usually related to each other:

```
Pixel ─ Pixel ─ Pixel
  │
Pixel ─ Pixel ─ Pixel
  │
Pixel ─ Pixel ─ Pixel
```

A CNN should therefore be able to examine local neighborhoods instead of treating every pixel as completely independent.

Convolution provides exactly this capability.

Instead of connecting every pixel to every neuron, convolution examines small regions of the image.

For example:

```
3 × 3 local region
```

can be processed at a time.

This allows the network to detect local patterns such as:

- Edges
- Corners
- Textures
- Intensity changes
- Small shapes

---

## 3. Basic Convolution Operation

Suppose we have a small input:

```
5 × 5
```

and a:

```
3 × 3
```

kernel.

The input may look like:

```
┌────┬────┬────┬────┬────┐
│  1 │  2 │  3 │  4 │  5 │
├────┼────┼────┼────┼────┤
│  6 │  7 │  8 │  9 │ 10 │
├────┼────┼────┼────┼────┤
│ 11 │ 12 │ 13 │ 14 │ 15 │
├────┼────┼────┼────┼────┤
│ 16 │ 17 │ 18 │ 19 │ 20 │
├────┼────┼────┼────┼────┤
│ 21 │ 22 │ 23 │ 24 │ 25 │
└────┴────┴────┴────┴────┘
```

The kernel examines a 3 × 3 region at a time.

For example, the first region is:

```
┌────┬────┬────┐
│  1 │  2 │  3 │
├────┼────┼────┤
│  6 │  7 │  8 │
├────┼────┼────┤
│ 11 │ 12 │ 13 │
└────┴────┴────┘
```

---

## 4. The Kernel

Suppose the kernel contains:

```
┌────┬────┬────┐
│ w₁ │ w₂ │ w₃ │
├────┼────┼────┤
│ w₄ │ w₅ │ w₆ │
├────┼────┼────┤
│ w₇ │ w₈ │ w₉ │
└────┴────┴────┘
```

These values are learnable parameters.

The convolution operation multiplies corresponding values of the input region and kernel.

For example:

Input Region:

```
┌────┬────┬────┐
│  1 │  2 │  3 │
├────┼────┼────┤
│  6 │  7 │  8 │
├────┼────┼────┤
│ 11 │ 12 │ 13 │
└────┴────┴────┘
```

Kernel:

```
┌────┬────┬────┐
│ w₁ │ w₂ │ w₃ │
├────┼────┼────┤
│ w₄ │ w₅ │ w₆ │
├────┼────┼────┤
│ w₇ │ w₈ │ w₉ │
└────┴────┴────┘
```

The result is:

\[
z =
1 × w_1 +
2 × w_2 +
3 × w_3 +
6 × w_4 +
7 × w_5 +
8 × w_6 +
11 × w_7 +
12 × w_8 +
13 × w_9
\]

---

## 5. One Local Region Produces One Number

This is one of the most important ideas to understand.

A kernel does not produce an entire feature map from one position.

Instead:

```
One Local Region
       ↓
Element-wise Multiplication
       ↓
Summation
       ↓
One Number
```

Then the kernel moves to another location.

```
Region 1 → Number 1
Region 2 → Number 2
Region 3 → Number 3
...
```

These numbers are arranged spatially to form a feature map.

---

## 6. Sliding the Kernel Across the Input

The kernel moves across the input according to the stride.

For example, with:

```
Kernel = 3 × 3
Stride = 1
```

the kernel moves one pixel at a time.

Conceptually:

Position 1

```
┌───────────────┐
│ Kernel        │
│               │
└───────────────┘
```

        ↓

Position 2

```
   ┌───────────────┐
   │ Kernel        │
   │               │
   └───────────────┘
```

        ↓

Position 3

```
      ┌───────────────┐
      │ Kernel        │
      │               │
      └───────────────┘
```

The same kernel weights are reused at every position.

---

## 7. Weight Sharing

One of the most important properties of convolution is weight sharing.

The same kernel is applied to every spatial location.

For example:

```
Location 1
    ↓
Same Kernel

Location 2
    ↓
Same Kernel

Location 3
    ↓
Same Kernel
```

The network does not learn a completely separate kernel for every pixel location.

Instead, one set of weights is reused.

This greatly reduces the number of parameters.

---

## 8. Why Is Weight Sharing Useful?

Suppose a kernel learns to detect a vertical edge.

Because the same kernel is used across the image, it can detect vertical edges regardless of where they appear.

For example:

```
┌──────────────────────┐
│       Edge           │
│                      │
│             Edge     │
│                      │
│   Edge               │
└──────────────────────┘
```

The same learned filter can respond to all of these locations.

This is an important property of convolutional networks.

---

## 9. Feature Map

After the kernel has scanned the input, we obtain a collection of activation values.

For example:

```
┌───────────────┐
│ 0.1  0.4  0.8 │
│ 0.2  0.7  0.9 │
│ 0.0  0.3  0.6 │
└───────────────┘
```

This is called a Feature Map.

Each value corresponds to the response of the filter at a particular spatial location.

Therefore:

```
Input
  ↓
One Filter
  ↓
Convolution
  ↓
One Feature Map
```

---

## 10. Strong and Weak Activations

The values in a feature map indicate how strongly the filter responds to different regions.

For example:

```
0.1  0.2  0.9
0.0  0.3  0.8
0.1  0.1  0.7
```

The larger values indicate stronger responses.

If the filter has learned to detect a certain pattern, locations containing that pattern may produce stronger activations.

Conceptually:

```
Strong Activation
       ↓
Pattern detected strongly

Weak Activation
       ↓
Pattern detected weakly
```

---

## 11. Multiple Filters

A convolutional layer normally contains multiple filters.

Suppose a layer contains:

```
64 Filters
```

Then:

```
Filter 1  → Feature Map 1
Filter 2  → Feature Map 2
Filter 3  → Feature Map 3
...
Filter 64 → Feature Map 64
```

Therefore:

```
64 Filters
      ↓
64 Feature Maps
```

The output has 64 channels.

For example:

```
Input: 224 × 224 × 3
        ↓
Convolution
64 Filters
        ↓
Output: 224 × 224 × 64
```

assuming stride and padding preserve the spatial dimensions.

---

## 12. Multi-Channel Input

Images can have multiple channels.

An RGB image has:

```
Height × Width × 3
```

where the three channels are:

- Red
- Green
- Blue

Suppose the input is:

```
224 × 224 × 3
```

and we use:

```
3 × 3
```

convolution.

A complete filter must operate across all three input channels.

Therefore, one filter has dimensions:

```
3 × 3 × 3
```

Conceptually:

```
Red Channel
     ↓
3 × 3 Kernel

Green Channel
     ↓
3 × 3 Kernel

Blue Channel
     ↓
3 × 3 Kernel
      ↓
Combine
      ↓
One Output Value
```

---

## 13. Convolution Across Channels

This is an important concept.

Suppose the input has:

```
3 channels
```

and one filter has:

```
3 × 3 × 3
```

weights.

The filter performs convolution independently on each channel and combines the results.

Conceptually:

```
Red:
3×3 region × Red weights
             ↓
          Partial Sum

Green:
3×3 region × Green weights
             ↓
          Partial Sum

Blue:
3×3 region × Blue weights
             ↓
          Partial Sum

          ↓
       Combine
          ↓
     One Activation
```

Therefore, one filter produces one output value at each spatial location.

---

## 14. General Shape of a Convolutional Filter

If the input has:

```
C_in
```

channels, and the kernel has spatial dimensions:

```
K × K
```

then one complete filter has:

```
K × K × C_in
```

weights.

If there are:

```
C_out
```

filters, the convolutional layer contains:

```
K × K × C_in × C_out
```

weights.

---

## 15. Kernel Size

The kernel size determines the spatial region examined at each step.

Common kernel sizes include:

- `1 × 1`
- `3 × 3`
- `5 × 5`
- `7 × 7`

For example:

```
3 × 3
```

Examines:

```
9 pixels per channel
```

```
5 × 5
```

Examines:

```
25 pixels per channel
```

```
7 × 7
```

Examines:

```
49 pixels per channel
```

Larger kernels can capture larger local spatial patterns but require more computation.

---

## 16. Why 3 × 3 Convolutions Are Common

The 3 × 3 kernel is extremely common in CNN architectures.

It provides a useful balance between:

- Local spatial coverage
- Number of parameters
- Computational cost
- Network depth

For example, instead of one:

```
7 × 7
```

convolution, multiple:

```
3 × 3
```

convolutions can be stacked.

This allows the network to introduce additional non-linearities while keeping the operations relatively efficient.

---

## 17. Stride

Stride determines how far the kernel moves between two consecutive positions.

**With** `Stride = 1`

The kernel moves one pixel at a time.

```
Position 1
    ↓
Position 2
    ↓
Position 3
```

This generally produces a larger output.

**With** `Stride = 2`

The kernel moves two pixels at a time.

```
Position 1
      ↓
Position 3
      ↓
Position 5
```

This reduces the spatial resolution.

Therefore:

```
Larger Stride
      ↓
Smaller Spatial Output
```

---

## 18. Padding

When convolution is applied without padding, the kernel cannot be centered on boundary pixels in the same way as interior pixels.

As a result, the output becomes smaller.

For example:

```
Input: 5 × 5
Kernel: 3 × 3
Stride: 1
No Padding: Output: 3 × 3
```

Padding adds values around the border of the input.

The most common padding is zero padding.

Conceptually:

Original:

```
┌─────────────┐
│             │
│    Image    │
│             │
└─────────────┘
```

After Padding:

```
┌─────────────────┐
│ 0 0 0 0 0 0 0   │
│ 0             0 │
│ 0    Image    0 │
│ 0             0 │
│ 0 0 0 0 0 0 0   │
└─────────────────┘
```

---

## 19. Same Padding

For a `3 × 3` kernel with:

```
Stride = 1
Padding = 1
```

the spatial dimensions can remain unchanged.

For example:

```
Input: 224 × 224
       ↓
3×3 Conv
Padding = 1
Stride = 1
       ↓
Output: 224 × 224
```

This is commonly used in CNN architectures.

---

## 20. Convolution Is a Learnable Operation

The convolution operation itself uses learnable weights.

Suppose a filter is:

```
W
```

and the input is:

```
X
```

Then the output can be represented conceptually as:

\[
Y = X * W + b
\]

where:

- \(X\) = Input
- \(W\) = Learnable filter weights
- \(b\) = Bias
- \(Y\) = Output
- \(*\) = Convolution operation

During training, the values of \(W\) are updated.

---

## 21. How Are Convolution Weights Learned?

The weights are learned through backpropagation.

The training process is:

```
Input Image
     ↓
Convolution
     ↓
Feature Maps
     ↓
Activation
     ↓
Prediction
     ↓
Loss
     ↓
Backpropagation
     ↓
Gradients
     ↓
Update Kernel Weights
```

For example, if a filter is not producing useful activations, the gradients can modify its weights.

Over many iterations, the filter becomes better suited to the task.

---

## 22. Convolution and Feature Detection

A convolutional filter can learn to respond to particular patterns.

For example, an early filter might learn an edge detector:

```
┌───────────────┐
│ -1   0    1   │
│ -1   0    1   │
│ -1   0    1   │
└───────────────┘
```

If this pattern is applied to an image region containing a strong vertical intensity transition, the output activation may be large.

The network can therefore use convolution to detect local structures.

However, CNN filters are learned from data rather than manually designed as in this illustrative example.

---

## 23. Convolution Produces Hierarchical Features

One convolutional layer can detect relatively simple patterns.

When multiple convolutional layers are stacked, later layers operate on the feature maps produced by earlier layers.

For example:

```
Input Image
     ↓
Conv Layer 1
     ↓
Edges
     ↓
Conv Layer 2
     ↓
Textures
     ↓
Conv Layer 3
     ↓
Patterns
     ↓
Conv Layer 4
     ↓
Shapes
```

This produces hierarchical representation learning.

---

## 24. Convolution in Deep CNNs

A deep CNN may contain many convolutional layers.

Conceptually:

```
Input
  ↓
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
  ↓
...
  ↓
High-Level Representation
```

As depth increases, the network can combine simpler features into more complex representations.

However, making CNNs very deep also introduces optimization challenges.

These challenges motivated architectures such as ResNet.

---

## 25. Convolution in ResNet

ResNet is a CNN architecture that relies heavily on convolution.

For example, the initial layer of ResNet50 is:

```
Input: 224 × 224 × 3
      ↓
7 × 7 Convolution
64 Filters
Stride = 2
      ↓
112 × 112 × 64
```

After this, the network performs:

```
Batch Normalization
      ↓
ReLU
      ↓
Max Pooling
      ↓
Residual Blocks
```

Inside ResNet50 Bottleneck Blocks, convolutional layers include:

```
1 × 1 Conv
      ↓
3 × 3 Conv
      ↓
1 × 1 Conv
```

The 3 × 3 convolution performs the main spatial feature extraction, while the 1 × 1 convolutions control the channel dimension.

---


## 26. Parameter Efficiency

Consider an RGB input:

```
224 × 224 × 3
```

and a convolutional layer with:

```
3 × 3
64 Filters
```

The number of weights is:

\[
3 × 3 × 3 × 64 = 1728
\]

With biases:

\[
1728 + 64 = 1792
\]

This is dramatically smaller than connecting all 150,528 input values directly to many neurons.

This parameter efficiency is one of the reasons CNNs work well with image data.

---

## 27. Important Distinction: Kernel vs Output Feature Map

These concepts should not be confused.

### Kernel

The kernel contains the learnable weights:

```
3 × 3
```

or, for RGB input:

```
3 × 3 × 3
```

### Feature Map

The feature map is the output generated by applying the kernel across the input:

```
Height × Width
```

Therefore:

```
Kernel
   ↓
Convolution
   ↓
Feature Map
```

The kernel is the learned detector.

The feature map is the spatial response of that detector.

---

## 28. One Filter vs Multiple Filters

For one filter:

```
Input
  ↓
Filter 1
  ↓
Feature Map 1
```

For multiple filters:

```
Input
  │
  ├── Filter 1 ──→ Feature Map 1
  ├── Filter 2 ──→ Feature Map 2
  ├── Filter 3 ──→ Feature Map 3
  │
  └── Filter N ──→ Feature Map N
```


## 29. Complete Example

Suppose we have:

```
Input: 224 × 224 × 3
```

and:

```
Kernel Size: 3 × 3
Number of Filters: 64
Stride: 1
Padding: 1
```

Each filter has:

```
3 × 3 × 3
```

weights.

The number of weights is:

\[
3 × 3 × 3 × 64 = 1728
\]

With bias:

\[
1728+64=1792
\]


Therefore:

```
Input
224 × 224 × 3
        ↓
3×3 Convolution
64 Filters
        ↓
224 × 224 × 64
```

This means:

- The layer has 64 learned filters.
- Each filter operates across all 3 input channels.
- Each filter produces one feature map.
- The output contains 64 feature maps.
- Each feature map has spatial dimensions of 224 × 224.

---

## 30. Convolution as a Transformation

A useful way to think about convolution is:

```
Raw Pixel Space
      ↓
Local Pattern Detection
      ↓
Feature Maps
      ↓
Feature Representation
```

The convolutional layer transforms the input into a representation that is more useful for later layers.

Early layers may transform:

Pixels

into:

Edges and Textures

while deeper layers transform these into:

Complex Patterns

and eventually:

High-Level Representations

---

## 31. Important Parameters of a Convolution

A convolutional layer is mainly characterized by:

- **Kernel Size**: Controls the spatial size of the filter.
  - `3 × 3`
  - `5 × 5`
  - `7 × 7`
  
- **Number of Filters**: Controls the number of output channels.
  - `32`
  - `64`
  - `128`
  - `256`
  
- **Stride**: Controls how far the kernel moves.
  - `Stride = 1`
  - `Stride = 2`
  
- **Padding**: Controls how boundaries are handled.
  - `Padding = 0`
  - `Padding = 1`
  
- **Input Channels**: Determines the depth of each complete filter.
  - `RGB → 3 channels`

---

## 32. Summary

Convolution is the fundamental operation that allows CNNs to extract local features from images.

The process can be summarized as:

```
Input Image
     ↓
Select Local Region
     ↓
Apply Kernel
     ↓
Element-wise Multiplication
     ↓
Summation
     ↓
Add Bias
     ↓
One Activation
     ↓
Move Kernel
     ↓
Repeat
     ↓
Feature Map
```

Multiple filters produce multiple feature maps:

```
Input
  ↓
┌────────────────────────────┐
│ Filter 1                   │ → Feature Map 1
│ Filter 2                   │ → Feature Map 2
│ Filter 3                   │ → Feature Map 3
│ ...                        │
│ Filter N                   │ → Feature Map N
└────────────────────────────┘
  ↓
Output Tensor
```

The most important relationships are:

\[
\boxed{
\text{One Filter} \rightarrow \text{One Feature Map}
}
\]

and:

\[
\boxed{
\text{Number of Filters}
=
\text{Number of Output Channels}
}
\]

and:

\[
\boxed{
\text{Parameters}
=
K_h \times K_w \times C_{in} \times C_{out}
}
\]

Convolution therefore provides the fundamental mechanism through which CNNs transform raw input data into learned visual representations.

---

## 33. Key Takeaways

- A convolution uses a small set of learnable weights.
- The kernel examines a local region of the input.
- Element-wise multiplication followed by summation produces an activation.
- The kernel slides across the input.
- The same weights are reused at different spatial locations.
- This is known as weight sharing.
- One filter produces one feature map.
- Multiple filters produce multiple output feature maps.
- The number of filters determines the number of output channels.
- Kernel size determines the spatial region examined.
- Stride controls how far the kernel moves.
- Padding controls boundary handling and output size.
- For multi-channel input, a complete filter spans all input channels.
- Convolutional weights are learned through backpropagation.
- Stacking convolutional layers enables hierarchical feature learning.
- ResNet uses convolution extensively and adds residual connections to make very deep CNNs easier to optimize.

