# Histogram of Oriented Gradients (HOG)

## 1. Introduction

**Histogram of Oriented Gradients (HOG)** is a classical feature extraction technique used in computer vision.

HOG represents an image by describing the **local distribution of edge directions**.

Instead of directly giving raw pixel values to a machine learning model, HOG transforms the image into a numerical feature vector that captures important shape and edge information.

The basic idea is:

```text
Input Image
     ↓
Calculate Gradients
     ↓
Gradient Magnitude & Orientation
     ↓
Divide Image into Cells
     ↓
Build Orientation Histograms
     ↓
Normalize Blocks
     ↓
HOG Feature Vector
     ↓
Machine Learning Model
```

HOG became particularly popular for object detection and image classification before the widespread use of deep learning.

---

## 2. Why Feature Engineering?

Before deep learning became dominant, machine learning algorithms generally could not work effectively with raw images directly.

For example, an image might contain:

224 × 224 × 3 values.

A traditional machine learning model does not automatically understand that certain groups of pixels represent:

- edges
- corners
- shapes
- contours
- textures

Therefore, researchers manually designed methods to extract useful information from images. This process is called:

> Feature Engineering

HOG is one example of manually designed visual feature extraction.

---

## 3. Raw Pixels vs HOG Features

Consider an image:

```
Input Image
     ↓
Raw Pixel Values
     ↓
Machine Learning Model
```

The model receives pixel intensities directly.

With HOG:

```
Input Image
     ↓
HOG Feature Extraction
     ↓
Edge / Shape Representation
     ↓
Feature Vector
     ↓
Machine Learning Model
```

The important difference is that HOG explicitly transforms the image into features related to local edge structure.

---

## 4. What Does HOG Actually Detect?

HOG focuses primarily on:

- Edges
- Contours
- Local Shapes
- Gradient Directions

For example, consider a simple vertical edge:

```
██████│░░░░░
██████│░░░░░
██████│░░░░░
██████│░░░░░
```

The pixel intensity changes strongly around the boundary.

This produces a strong gradient.

HOG captures the direction of such gradients.

Therefore, HOG can represent the structural shape of an object without relying directly on its exact pixel values.

---

## 5. Image Gradients

The first important concept in HOG is the image gradient.

A gradient describes how quickly image intensity changes from one pixel to neighboring pixels.

Suppose we have:

```
10  10  10
10  50  50
10  50  50
```

There is a strong intensity change between the darker and brighter regions.

This change can be detected using gradient operators.

Two important quantities are calculated:

1. Gradient magnitude
2. Gradient orientation

---

## 6. Horizontal and Vertical Gradients

For each pixel, HOG estimates how intensity changes in two directions:

- Horizontal Gradient → Gx
- Vertical Gradient   → Gy

Conceptually:

```
Gy
             ↑
             │
             │
             └────────→ Gx
```

Common operators such as Sobel filters can be used to estimate these gradients.

For example:

- Gx ≈ Horizontal intensity change
- Gy ≈ Vertical intensity change

---

## 7. Gradient Magnitude

Once \( G_x \) and \( G_y \) are known, the gradient magnitude can be calculated as:

\[
G=\sqrt{G_x^2+G_y^2}
\]

The magnitude tells us:

> How strong is the intensity change at this location?

For example:

- Small Magnitude → Weak Edge
- Large Magnitude → Strong Edge

Therefore, a large gradient magnitude indicates a strong local change in image intensity.

---

## 8. Gradient Orientation

The gradient also has a direction.

The orientation can be calculated using:

\[
\theta=\arctan\left(\frac{G_y}{G_x}\right)
\]

More commonly, implementations use:

\[
\theta=\operatorname{atan2}(G_y,G_x)
\]

The orientation tells us:

> In which direction is the strongest intensity change occurring?

Conceptually:

```
↑
        │
        │
        └────────→
```

Different edges produce different gradient orientations.

---

## 9. Edge Direction vs Gradient Direction

This distinction is important.

The gradient direction represents the direction of the strongest intensity change.

The visible edge itself is generally perpendicular to the gradient.

For example:

Vertical Edge

```
██████│░░░░░
██████│░░░░░
██████│░░░░░
```

The edge is vertical.

The strongest intensity change occurs horizontally.

Therefore, the gradient points approximately horizontally.

This distinction is important when interpreting HOG orientation histograms.

---

## 10. Example of Gradient Calculation

Suppose:

\[
G_x=3
\]

and:

\[
G_y=4
\]

The magnitude is:

\[
G=\sqrt{3^2+4^2}
\]

\[
G=\sqrt{9+16}
\]

\[
G=5
\]

The orientation is approximately:

\[
\theta=\operatorname{atan2}(4,3)
\]

which is approximately:

\[
53.1^\circ
\]

Therefore, this pixel has:

- Magnitude ≈ 5
- Orientation ≈ 53°

---

## 11. Dividing the Image into Cells

HOG does not build one histogram for the entire image.

Instead, it divides the image into small spatial regions called:

> Cells

For example:

```
Image
┌──────┬──────┬──────┬──────┐
│ Cell │ Cell │ Cell │ Cell │
├──────┼──────┼──────┼──────┤
│ Cell │ Cell │ Cell │ Cell │
├──────┼──────┼──────┼──────┤
│ Cell │ Cell │ Cell │ Cell │
├──────┼──────┼──────┼──────┤
│ Cell │ Cell │ Cell │ Cell │
└──────┴──────┴──────┴──────┘
```

A common configuration is:

8 × 8 pixels per cell, although other configurations are possible.

---

## 12. Why Divide the Image into Cells?

If we calculated one global histogram for the entire image, we would lose spatial information.

For example:

```
Entire Image
     ↓
One Histogram
```

would tell us which edge directions are present, but not where they occur.

By dividing the image into cells:

```
Image
 ↓
Cells
 ↓
Local Histograms
```

HOG retains information about the local spatial structure of the image.

This is important because shape depends not only on which edges exist, but also on where they occur.

---

## 13. Orientation Histogram

For every cell, HOG creates a histogram of gradient orientations.

Suppose a cell contains gradients with orientations:

- 10°
- 20°
- 15°
- 90°
- 85°
- 180°

The orientations are grouped into predefined bins.

For example:

```
0° ───────────────→ 180°

|    |    |    |    |    |    |    |    |    |
0   20   40   60   80  100  120  140  160  180
```

Each bin records how much gradient information belongs to that orientation.

---

## 14. Why Magnitude Matters

HOG does not simply count how many pixels have a particular orientation.

The gradient magnitude is used as a weight.

Suppose two pixels have the same orientation:

- Pixel A:
  - Magnitude = 1
  
- Pixel B:
  - Magnitude = 10

Pixel B has a much stronger gradient.

Therefore, it contributes more strongly to the histogram.

Conceptually:

```
Gradient
    ↓
Orientation → Select Histogram Bin
    ↓
Magnitude → Determine Contribution
```

So HOG captures both:

- Where the gradient points
- How strong the gradient is

---

## 15. Building a Histogram

Suppose a cell has four gradients:

| Orientation | Magnitude |
|-------------|-----------|
| 20°        | 2         |
| 25°        | 4         |
| 90°        | 8         |
| 95°        | 3         |

The first two contribute to the bin around:

20°–30°

and the other two contribute to a bin around:

90°–100°

The histogram might conceptually look like:

```
Magnitude
   │
  12│
   │
   │              █
   │              █
   │              █
   │      █       █
   │      █       █
   └────────────────────
       20°       90°
```

The high bars indicate dominant gradient orientations.

---

## 16. Orientation Bins

A common HOG configuration uses:

9 orientation bins

for orientations from:

\[
0^\circ \text{ to } 180^\circ
\]

This means the orientation range is divided into approximately:

\[
20^\circ
\]

per bin.

For example:

- Bin 1 → 0°–20°
- Bin 2 → 20°–40°
- Bin 3 → 40°–60°
- ...
- Bin 9 → 160°–180°

The exact configuration can vary depending on the implementation.

---

## 17. Unsigned vs Signed Gradients

HOG commonly uses unsigned gradient orientations.

That means:

\[
0^\circ
\]

and:

\[
180^\circ
\]

are treated as equivalent.

Therefore, the orientation range is:

\[
0^\circ \rightarrow 180^\circ
\]

rather than:

\[
0^\circ \rightarrow 360^\circ
\]

This is useful because an edge often has the same structural meaning regardless of which direction its intensity changes.

---

## 18. Block Formation

After creating histograms for individual cells, HOG groups neighboring cells into larger regions called:

> Blocks

For example:

```
┌────────┬────────┐
│ Cell 1 │ Cell 2 │
├────────┼────────┤
│ Cell 3 │ Cell 4 │
└────────┴────────┘
```

        ↓

      Block

A common configuration is:

2 × 2 cells per block.

---

## 19. Why Use Blocks?

Different regions of an image may have different lighting or contrast.

For example:

- Bright Region → Large Gradient Values
- Dark Region  → Smaller Gradient Values

Without normalization, HOG could become sensitive to such changes.

Block normalization reduces this sensitivity.

---

## 20. Block Normalization

Suppose a block contains a vector of histogram values:

\[
v=[v_1,v_2,\dots,v_n]
\]

The vector can be normalized using an (L_2)-based normalization such as:

\[
v'=\frac{v}{\sqrt{\|v\|_2^2+\epsilon^2}}
\]

where:

- \( v \) = original block histogram vector
- \( v' \) = normalized vector
- \( \epsilon \) = small constant for numerical stability

The exact normalization method can vary.

---

## 21. Why Normalization Helps

Normalization makes HOG more robust to changes in:

- Lighting
- Contrast
- Overall Intensity

For example:

```
Same Shape
    +
Different Brightness
    ↓
Similar Gradient Structure
    ↓
More Similar HOG Representation
```

Therefore, HOG focuses more on shape and edge structure than absolute brightness.

---

## 22. From Histograms to a Feature Vector

After calculating and normalizing all block histograms, HOG concatenates them into one long vector.

Conceptually:

```
Cell Histograms
      ↓
Block Histograms
      ↓
Normalized Blocks
      ↓
Concatenate
      ↓
HOG Feature Vector
```

For example:

```
Block 1 → [ ... ]
Block 2 → [ ... ]
Block 3 → [ ... ]
Block 4 → [ ... ]
```

        ↓

```
[ ... all features ... ]
```

The final result is a one-dimensional numerical vector.

---

## 23. Example HOG Pipeline

A complete HOG pipeline can be represented as:

```
Input Image
     ↓
Convert to Grayscale
     ↓
Calculate Gx and Gy
     ↓
Calculate Gradient Magnitude
     ↓
Calculate Gradient Orientation
     ↓
Divide Image into Cells
     ↓
Create Orientation Histograms
     ↓
Group Cells into Blocks
     ↓
Normalize Blocks
     ↓
Concatenate Features
     ↓
HOG Feature Vector
```

---

## 24. Why Is Grayscale Often Used?

HOG is traditionally applied to grayscale images.

The reason is that HOG primarily focuses on:

- Intensity Changes
- Gradients
- Edges
- Shape

Color information is therefore not the primary focus of the original HOG descriptor.

For an RGB image:

- R
- G
- B

the image can first be converted into a grayscale representation.

Then gradients are calculated from the grayscale image.

---

## 25. HOG for Object Detection

One of the most famous applications of HOG was human detection.

For example:

```
Image
  ↓
HOG
  ↓
Shape / Edge Representation
  ↓
Classifier
  ↓
Human / Not Human
```

The human body creates characteristic arrangements of edges and contours.

HOG captures these local edge patterns.

A machine learning classifier such as an SVM can then use the HOG feature vector.

---

## 26. HOG + SVM

A classic computer vision pipeline was:

```
Input Image
      ↓
HOG Feature Extraction
      ↓
Feature Vector
      ↓
SVM
      ↓
Prediction
```

Here:

- HOG extracts visual features.
- SVM uses those features to perform classification.

This is an example of a traditional computer vision system where feature extraction and classification are separate components.

---

## 27. HOG vs CNN

HOG and CNNs represent two different approaches to visual feature extraction.

**Traditional Approach**

```
Image
 ↓
Hand-Designed Feature Extraction
 ↓
HOG
 ↓
Feature Vector
 ↓
SVM / Other Classifier
 ↓
Prediction
```

**Deep Learning Approach**

```
Image
 ↓
CNN
 ↓
Learned Features
 ↓
Classifier
 ↓
Prediction
```

The major difference is:

> HOG uses a manually designed feature extraction method, while CNNs learn feature representations directly from data.

---

## 28. HOG and Representation Learning

This is an important connection to the concept of Representation Learning.

With HOG:

- Human designs the feature extraction method
  ↓
- HOG
  ↓
- Feature Representation

The fundamental properties of the representation are predefined by the algorithm.

With CNNs:

```
Image
  ↓
CNN
  ↓
Training
  ↓
Learned Representation
```

The network learns its filters and representations from the training data.

Therefore:

- HOG → Hand-Crafted Features
- CNN → Learned Features

---

## 29. HOG in the Context of CNNs

A CNN may learn features that resemble some classical visual features.

Early CNN layers can learn:

- Edges
- Corners
- Textures

HOG explicitly describes:

- Gradient Directions
- Edge Structure
- Local Shape

However, CNNs can go beyond these manually defined descriptors.

As depth increases, CNNs can learn:

- Edges
  ↓
- Textures
  ↓
- Shapes
  ↓
- Object Parts
  ↓
- High-Level Semantic Features

This is one reason deep learning became so powerful in computer vision.

---

## 30. Advantages of HOG

HOG has several advantages:

1. **Captures Shape Information**: HOG focuses strongly on edges and contours.

2. **Relatively Robust to Lighting**: Block normalization reduces sensitivity to global intensity changes.

3. **Interpretable**: The meaning of the representation is relatively easy to understand.

4. **Computationally Simpler**: Compared with training a deep CNN, extracting HOG features can be computationally inexpensive.

5. **Works Well with Traditional Classifiers**: HOG features can be combined with:
   - SVM
   - Logistic Regression
   - Random Forest
   - k-NN

depending on the task.

---

## 31. Limitations of HOG

HOG also has important limitations.

1. **Hand-Crafted**: The feature representation is designed manually.

2. **Limited Semantic Understanding**: HOG mainly describes local edge structure. It does not naturally learn high-level semantic concepts.

3. **Sensitive to Some Geometric Changes**: Large changes in:
   - Rotation
   - Scale
   - Deformation
   - Viewpoint

can affect the HOG representation.

4. **Limited Color Information**: Traditional HOG primarily focuses on intensity gradients.

5. **Less Flexible Than Learned Representations**: CNNs can adapt their learned filters to the specific dataset and task.

---

## 32. HOG Feature Vector Example

Suppose the final HOG representation contains:

324 features

The image is then represented as:

```
[0.12, 0.05, 0.31, 0.00, ..., 0.17]
```

This vector can be given to a machine learning algorithm:

```
HOG Feature Vector
       ↓
SVM
       ↓
Prediction
```

The classifier does not directly receive the original image. It receives the engineered representation.

---

## 33. HOG and Dimensionality

The size of the HOG feature vector depends on parameters such as:

- Image size
- Cell size
- Number of orientation bins
- Block size
- Block stride

For example:

- Smaller Cells → More Cells → More Histograms → Larger Feature Vector

Similarly:

- More Orientation Bins → More Histogram Values → Larger Feature Vector

Therefore, HOG configuration directly affects the dimensionality of the final representation.

---

## 34. Important HOG Parameters

Common HOG parameters include:

- **Pixels per Cell**: Defines the spatial size of each cell. Example: 8 × 8
- **Cells per Block**: Defines how many cells form a normalization block. Example: 2 × 2
- **Orientations**: Defines the number of orientation bins. Example: 9

These parameters control the spatial and directional resolution of the HOG descriptor.

---

## 35. Conceptual Example

Imagine an image containing a simple object:

```
██
       ████
      ██  ██
     ██    ██
    ██████████
```

HOG does not try to understand:

> "This is a particular object."

Instead, it extracts information such as:

- Strong vertical gradients
- Strong horizontal gradients
- Diagonal gradients
- Local edge distributions

These patterns are encoded into histograms.

The final feature vector describes the object's local edge structure.

---

## 36. HOG as a Hand-Crafted Representation

The complete idea can be summarized as:

```
HOG
                 │
        ┌────────┴────────┐
        │                 │
     Gradient          Spatial
     Information       Information
        │                 │
        ↓                 ↓
 Orientation         Local Cells
        │                 │
        └────────┬────────┘
                 ↓
          Local Histograms
                 ↓
          Block Normalization
                 ↓
          Feature Vector
```

HOG therefore combines:

- Gradient Direction
- Gradient Strength
- Local Spatial Organization

to create a structured image representation.

---

## 37. HOG vs Raw Pixels

Consider two representations of the same image.

**Raw Pixels**:

```
[125, 128, 130, 140, ...]
```

This representation contains the original intensity values.

**HOG**:

```
[0.12, 0.03, 0.41, 0.08, ...]
```

This representation describes the distribution of local gradient orientations.

Therefore, HOG transforms the problem from:

> "What are the pixel values?"

to something closer to:

> "What edge and shape patterns exist in different regions of the image?"

---

## 38. HOG and Feature Engineering

HOG is a classic example of Feature Engineering.

The workflow is:

```
Raw Data
   ↓
Human-designed Feature Extraction
   ↓
HOG
   ↓
Engineered Feature Representation
   ↓
Machine Learning Algorithm
```

This differs from modern deep learning:

```
Raw Data
   ↓
Neural Network
   ↓
Automatic Feature Learning
   ↓
Learned Representation
   ↓
Prediction
```

This distinction is fundamental to understanding the transition from traditional computer vision to deep learning.

---

## 39. Key Takeaways

- HOG stands for Histogram of Oriented Gradients.
- HOG is a classical feature extraction technique.
- It describes local edge and shape information.
- HOG begins by calculating image gradients.
- Each pixel can have a gradient magnitude and orientation.
- The image is divided into small spatial cells.
- Each cell receives an orientation histogram.
- Gradient magnitude determines how strongly a pixel contributes to a histogram.
- Neighboring cells are grouped into blocks.
- Block normalization improves robustness to illumination and contrast changes.
- All normalized histograms are concatenated into a feature vector.
- HOG is a hand-crafted feature representation.
- HOG has historically been used with classifiers such as SVM.
- HOG does not learn its representation from data in the same way a CNN does.
- CNNs learn filters and hierarchical representations automatically.
- HOG primarily captures local edge, gradient, and shape information.
- The final HOG feature vector can be used as input to traditional machine learning algorithms.

---

## 40. Final Conceptual Summary

The core idea behind HOG is:

```
Input Image
                     ↓
              Compute Gradients
                     ↓
          ┌──────────┴──────────┐
          ↓                     ↓
     Magnitude              Orientation
          │                     │
          └──────────┬──────────┘
                     ↓
               Divide into Cells
                     ↓
           Orientation Histograms
                     ↓
               Group into Blocks
                     ↓
              Normalize Blocks
                     ↓
             Concatenate Features
                     ↓
              HOG Feature Vector
                     ↓
             Machine Learning Model
                     ↓
                 Prediction
```

In one sentence:

> HOG represents an image by describing how strong gradients are distributed across different orientations and spatial regions of the image.

This makes HOG a classic example of hand-crafted feature engineering, in contrast to CNNs where the feature representation is learned automatically from data.

