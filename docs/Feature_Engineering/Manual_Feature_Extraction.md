# Manual Feature Extraction

## 1. Introduction

**Manual Feature Extraction**, also known as **Hand-Crafted Feature Extraction**, is a traditional approach in computer vision and machine learning.

In this approach, humans design or select methods for extracting useful information from raw images before the data is given to a machine learning model.

The general pipeline is:

```text
Input Image
     ↓
Preprocessing
     ↓
Manual Feature Extraction
     ↓
Feature Vector
     ↓
Machine Learning Model
     ↓
Prediction
```

Unlike modern deep learning systems, the feature extraction process is not learned automatically through training.

---

## 2. What Is a Feature?

A feature is a measurable property or characteristic extracted from data that can help a machine learning model distinguish between different classes or patterns.

For images, features can represent:

- Edges
- Corners
- Textures
- Shapes
- Colors
- Gradients
- Local patterns
- Statistical properties
- Geometric structures

For example:

```
Image
  ↓
Feature Extraction
  ↓
Edges + Texture + Shape
```

These features provide a more informative representation than simply using raw pixel values.

---

## 3. Raw Pixels vs Features

Consider an RGB image with a resolution of:

\[
224 \times 224
\]

It contains:

\[
224 \times 224 \times 3 = 150,528
\]

pixel values.

The raw image can therefore be represented as:

```
224 × 224 × 3
```

However, raw pixels do not explicitly describe concepts such as:

- Edges
- Shapes
- Textures
- Corners
- Object boundaries

Manual feature extraction attempts to transform raw pixel information into more meaningful numerical representations.

```
Raw Image
    ↓
Feature Extraction
    ↓
Meaningful Features
```

---

## 4. Hand-Crafted Features

Features designed using human knowledge are called:

> Hand-Crafted Features

The idea is to determine which characteristics of an image are likely to be useful for a particular task.

For example, suppose we are analyzing objects based on their shape.

We might extract:

- Area
- Perimeter
- Width
- Height
- Aspect Ratio
- Circularity

Alternatively, for texture analysis:

- Contrast
- Energy
- Homogeneity
- Local Patterns

The choice of features depends heavily on the problem domain.

---

## 5. Traditional Computer Vision Pipeline

A traditional image classification system can be represented as:

```
Input Image
                       ↓
                 Preprocessing
                       ↓
              Feature Extraction
                       ↓
                Feature Vector
                       ↓
              Machine Learning
                       ↓
                  Prediction
```

For example:

```
Image
  ↓
HOG
  ↓
Feature Vector
  ↓
SVM
  ↓
Classification
```

Here:

- HOG acts as the feature extractor.
- SVM acts as the classifier.

---

## 6. Common Manual Feature Extraction Methods

Several classical methods have been developed for extracting hand-crafted image features.

Important examples include:

| Method           | Main Information                      |
|------------------|---------------------------------------|
| Sobel            | Image gradients                       |
| Canny            | Edges                                 |
| HOG              | Gradient orientation and shape        |
| SIFT             | Local distinctive features            |
| LBP              | Local texture                         |
| GLCM             | Texture statistics                    |
| Color Histogram   | Color distribution                    |
| Shape Descriptors | Geometric structure                   |

Each method focuses on a different type of visual information.

---

## 7. Edge Features

An edge is a location where image intensity changes significantly.

For example:

```
████████│░░░░░░
████████│░░░░░░
████████│░░░░░░
████████│░░░░░░
```

The boundary between the two regions represents an edge.

Edges are useful because they often correspond to:

- Object boundaries
- Contours
- Shape changes
- Structural transitions

Common edge detection methods include:

- Sobel
- Prewitt
- Roberts
- Canny

---

## 8. Sobel Operator

The Sobel operator is a classical method for estimating image gradients.

It uses two kernels:

- Horizontal Gradient → Gx
- Vertical Gradient → Gy

The gradient magnitude can be calculated as:

\[
G = \sqrt{G_x^2 + G_y^2}
\]

The gradient orientation can be calculated as:

\[
\theta = \operatorname{atan2}(G_y,G_x)
\]

Therefore, Sobel provides information about:

- How strong is the intensity change?
            +
- In which direction does it occur?

---

## 9. Canny Edge Detection

Canny Edge Detection is another classical method for extracting edge information.

A simplified pipeline is:

```
Input Image
     ↓
Gaussian Smoothing
     ↓
Gradient Calculation
     ↓
Non-Maximum Suppression
     ↓
Double Threshold
     ↓
Edge Tracking
     ↓
Edge Map
```

The final output is an edge representation of the image.

```
Original Image
      ↓
    Canny
      ↓
   Edge Map
```

---

## 10. Texture Features

Texture describes the local visual pattern or surface structure of an image.

For example:

**Smooth Texture:**

```
████████████
████████████
████████████
```

versus:

**Rough Texture:**

```
██░█░██░░█
░██░░███░
█░░██░█░█
```

Texture features can describe properties such as:

- Contrast
- Regularity
- Homogeneity
- Roughness
- Local intensity variation

Texture analysis is especially useful when visual differences are based on surface patterns.

---

## 11. GLCM

Gray-Level Co-occurrence Matrix (GLCM) is a classical texture descriptor.

Instead of looking at individual pixels independently, GLCM analyzes relationships between neighboring intensity values.

Conceptually:

```
Image
  ↓
Pixel Intensity Relationships
  ↓
GLCM
  ↓
Texture Features
```

Common features derived from GLCM include:

- Contrast
- Correlation
- Energy
- Homogeneity

GLCM is therefore useful when the spatial relationship between intensity values is important.

---

## 12. Local Binary Pattern

Local Binary Pattern (LBP) is another classical texture descriptor.

LBP compares a central pixel with its neighboring pixels.

For example:

**Neighbor pixels**

```
1  1  0
1  C  0
0  1  1
```

Each neighboring pixel is compared with the center pixel.

The comparisons produce binary values:

```
1 1 0
1   0
0 1 1
```

These binary patterns are then converted into numerical codes.

The overall pipeline is:

```
Image
  ↓
Local Neighborhoods
  ↓
Binary Patterns
  ↓
LBP Histogram
  ↓
Feature Vector
```

LBP is particularly useful for local texture analysis.

---

## 13. HOG

Histogram of Oriented Gradients (HOG) is one of the most well-known hand-crafted feature extraction methods.

HOG describes the distribution of gradient orientations in local regions of an image.

The basic pipeline is:

```
Image
  ↓
Calculate Gradients
  ↓
Magnitude + Orientation
  ↓
Divide Image into Cells
  ↓
Create Orientation Histograms
  ↓
Normalize Blocks
  ↓
Feature Vector
```

HOG is particularly useful for describing:

- Edges
- Contours
- Shapes
- Object structure

---

## 14. SIFT

Scale-Invariant Feature Transform (SIFT) is designed to detect and describe distinctive local image structures.

The simplified pipeline is:

```
Image
  ↓
Scale-Space Construction
  ↓
Keypoint Detection
  ↓
Keypoint Localization
  ↓
Orientation Assignment
  ↓
Descriptor Construction
  ↓
SIFT Descriptors
```

The classical SIFT descriptor has:

\[
128
\]

dimensions.

SIFT is particularly useful for:

- Feature matching
- Object recognition
- Image registration
- Image stitching

Unlike HOG, SIFT focuses on distinctive local keypoints.

---

## 15. Shape Features

Shape features describe the geometry of objects.

Examples include:

- Area
- Perimeter
- Width
- Height
- Aspect Ratio
- Circularity
- Centroid
- Contour

For example:

\[
Aspect\ Ratio = \frac{Width}{Height}
\]

These features can help distinguish objects with different geometric structures.

---

## 16. Color Features

Color can also be used as a manually extracted feature.

Examples include:

- Mean Color
- Color Histogram
- Color Variance
- Color Distribution

A common pipeline is:

```
RGB Image
     ↓
Color Representation
     ↓
Color Statistics
     ↓
Feature Vector
```

Different color spaces can be used, including:

- RGB
- HSV
- Lab

---

## 17. Statistical Features

Statistical properties of an image can also be used as features.

Examples include:

- Mean
- Variance
- Standard Deviation
- Minimum
- Maximum
- Median
- Entropy

For example, the mean intensity is:

\[
\mu = \frac{1}{N} \sum_{i=1}^{N} x_i
\]

where \( x_i \) represents a pixel value.

These features provide information about the overall distribution of image intensities.

---

## 18. Geometric Features

Geometric features describe the spatial and structural properties of an object.

Examples:

- Width
- Height
- Area
- Perimeter
- Aspect Ratio
- Centroid
- Orientation
- Bounding Box

For example:

```
Object
┌───────────────┐
│               │
│      ●        │
│               │
└───────────────┘
```

The bounding box, centroid, and dimensions can all be converted into numerical features.

---

## 19. Feature Vector

After extracting features, they are usually represented as a numerical vector.

For example:

- Edge Density = 0.42
- Texture Energy = 0.71
- Mean Intensity = 128
- Area = 5230
- Aspect Ratio = 1.24

The resulting feature vector could be:

```
[0.42, 0.71, 128, 5230, 1.24]
```

This vector can then be provided to a machine learning algorithm.

---

## 20. Image to Feature Vector

The complete transformation is:

```
Input Image
                     ↓
          Manual Feature Extraction
                     ↓
        ┌────────────┼────────────┐
        ↓            ↓            ↓
      Shape        Texture       Edge
        ↓            ↓            ↓
        └────────────┼────────────┘
                     ↓
               Feature Vector
                     ↓
             Machine Learning
                     ↓
                 Prediction
```

The original image has been transformed into a lower-level numerical representation designed to contain useful information.

---

## 21. Feature Extraction vs Feature Selection

These concepts are related but different.

### Feature Extraction

Feature extraction creates a new representation from the original data.

For example:

```
Image
  ↓
HOG
  ↓
New Feature Vector
```

### Feature Selection

Feature selection chooses useful features from an existing set.

For example:

```
100 Features
     ↓
Feature Selection
     ↓
20 Important Features
```

Therefore:

- Feature Extraction → Creating or transforming features
- Feature Selection → Choosing useful features

---

## 22. Combining Multiple Feature Types

Different feature extractors can be combined.

For example:

```
Image
                      ↓
          ┌───────────┼───────────┐
          ↓           ↓           ↓
         HOG         LBP       Color
          ↓           ↓           ↓
        Shape      Texture      Color
       Features    Features    Features
          ↓           ↓           ↓
          └───────────┼───────────┘
                      ↓
             Combined Feature Vector
                      ↓
                 Classifier
```

This approach is called Feature Fusion.

The goal is to combine complementary information from different feature types.

---

## 23. Manual Feature Extraction + SVM

A classic machine learning pipeline is:

```
Image
  ↓
HOG
  ↓
Feature Vector
  ↓
SVM
  ↓
Classification
```

Here:

- HOG extracts image features.
- SVM learns the decision boundary between classes.

The classifier therefore does not need to directly process the raw image.

---

## 24. Why SVM Was Popular

Before deep learning became dominant, Support Vector Machines were widely used for image classification.

A typical system was:

```
Raw Image
    ↓
Hand-Crafted Feature Extractor
    ↓
Feature Vector
    ↓
SVM
    ↓
Prediction
```

For example:

```
Image
  ↓
HOG
  ↓
SVM
  ↓
Person / Background
```

The HOG algorithm already provided information about edges and shape, allowing SVM to focus on classification.

---

## 25. Role of Human Knowledge

Manual feature extraction depends heavily on human knowledge.

Suppose a researcher believes that texture is important for a classification task.

They might choose:

- GLCM
- LBP

If shape is important:

- HOG
- Contours
- Geometric Descriptors

If local distinctive structures are important:

- SIFT

Therefore, the choice of feature extractor depends on assumptions about the problem.

---

## 26. The Main Limitation

The biggest limitation of manual feature extraction is:

> The human must decide which characteristics are useful.

Suppose we choose:

- Texture
- Shape
- Color

but the actual task depends mostly on another subtle visual pattern.

Our feature representation may fail to capture that information.

Therefore:

```
Feature Quality
       ↓
Depends on
       ↓
Human Feature Design
```

---

## 27. Manual Features Are Not Learned

One of the most important differences between traditional feature engineering and deep learning is that manual features are not normally learned through backpropagation.

For example:

```
Image
 ↓
HOG
 ↓
Feature Vector
```

HOG does not update its fundamental feature extraction mechanism by receiving classification gradients from an SVM.

Similarly:

```
Image
 ↓
SIFT
 ↓
Descriptors
```

SIFT does not learn its descriptor through neural network training.

The extraction algorithm is predefined.

---

## 28. Manual Feature Extraction vs Representation Learning

This distinction is fundamental.

### Manual Feature Extraction

```
Image
  ↓
Human-designed Feature Extractor
  ↓
Feature Vector
  ↓
Classifier
```

**Example:**

```
Image
  ↓
HOG
  ↓
SVM
  ↓
Prediction
```

### Representation Learning

```
Image
  ↓
Neural Network
  ↓
Learned Representation
  ↓
Classifier
  ↓
Prediction
```

In Representation Learning, the model learns useful representations from data.

---

## 29. Traditional Computer Vision

Traditional computer vision often follows:

```
Raw Image
    ↓
Preprocessing
    ↓
Feature Engineering
    ↓
Feature Extraction
    ↓
Feature Vector
    ↓
Machine Learning
    ↓
Prediction
```

Examples include:

- HOG + SVM
- SIFT + Matching
- LBP + SVM
- GLCM + Random Forest

The feature extractor and classifier are generally separate components.

---

## 30. Deep Learning Approach

Modern deep learning can combine feature learning and classification into a single trainable system.

For example:

```
Image
  ↓
Convolution
  ↓
Feature Maps
  ↓
Deeper Convolutions
  ↓
Deep Representation
  ↓
Fully Connected Layer
  ↓
Prediction
```

The convolutional filters are learned from training data.

---

## 31. Traditional Features vs CNN Features

Consider HOG.

HOG explicitly calculates gradient orientations.

```
Image
  ↓
Gradient Calculation
  ↓
Orientation Histograms
  ↓
HOG Feature Vector
```

A CNN works differently:

```
Image
  ↓
Learned Convolution Filters
  ↓
Feature Maps
  ↓
Higher-Level Representations
```

The CNN can learn filters that respond to patterns useful for the specific task.

---

## 32. Hierarchical Representation

A major advantage of CNNs is that they can learn hierarchical representations.

For example:

- Early Layers
    ↓
- Edges
    ↓
- Textures
    ↓
- Parts
    ↓
- Complex Structures
    ↓
- Object-Level Features

These representations are learned automatically.

Manual feature extraction methods generally do not provide the same end-to-end hierarchical learning mechanism.

---

## 33. Manual Feature Engineering in Medical Imaging

Manual feature extraction has historically been important in medical image analysis.

For example:

- MRI
- CT
- X-Ray
- Ultrasound
- Dermoscopic Images

Researchers may extract:

- Texture
- Shape
- Intensity
- Edges
- Local Patterns

The pipeline can be:

```
Medical Image
      ↓
Preprocessing
      ↓
Manual Feature Extraction
      ↓
Feature Vector
      ↓
Machine Learning
      ↓
Classification
```

---

## 34. Example: Skin Lesion Analysis

For skin lesion images, traditional systems could extract features related to:

- Color
- Shape
- Asymmetry
- Border
- Texture
- Diameter

Conceptually:

```
Dermoscopic Image
        ↓
Feature Extraction
        ↓
┌───────┼────────┐
↓       ↓        ↓
Color  Shape   Texture
        ↓
Feature Vector
        ↓
Classifier
        ↓
Diagnosis
```

This approach relies on the assumption that these manually designed features contain enough information to distinguish lesions.

---

## 35. Example: Brain MRI

Manual feature engineering can also be applied to MRI images.

For example:

```
Brain MRI
    ↓
Texture Features
    ↓
Shape Features
    ↓
Intensity Features
    ↓
Feature Vector
    ↓
Machine Learning Model
    ↓
Classification
```

Possible features include:

- Mean intensity
- Intensity variance
- Texture statistics
- Region size
- Shape descriptors
- Edge information

However, modern deep learning approaches can learn more complex representations directly from MRI images.

---

## 36. Advantages of Manual Feature Extraction

1. **Interpretability**: The meaning of many features is understandable.
   - For example:
     - HOG → Gradient Orientation
     - GLCM → Texture Relationships
     - SIFT → Local Image Structure
     - LBP → Local Texture Pattern

2. **Less Training Data**: Traditional methods can sometimes work well with relatively small datasets.

3. **Lower Computational Requirements**: Many hand-crafted feature extraction methods require significantly fewer computational resources than training large deep neural networks.

4. **Domain Knowledge**: Human expertise can be explicitly incorporated into the representation.

5. **Useful Baseline**: Manual features can provide strong classical baselines for comparison with deep learning systems.

---

## 37. Limitations of Manual Feature Extraction

1. **Human Dependency**: The quality of the representation depends on feature design.

2. **Limited Adaptability**: A feature extractor designed for one problem may not work well for another.

3. **Difficult Feature Design**: Complex visual tasks can require extremely sophisticated feature engineering.

4. **Limited Hierarchical Learning**: Traditional descriptors generally do not learn multi-level representations like deep neural networks.

5. **Separate Optimization**: Feature extraction and classification are often optimized separately.

---

## 38. Manual Feature Engineering vs Deep Learning

| Property                          | Manual Feature Extraction   | Deep Learning            |
|-----------------------------------|-----------------------------|--------------------------|
| Feature design                    | Human-designed              | Learned from data        |
| Feature learning                  | Limited / absent            | Yes                      |
| End-to-end training               | Usually no                  | Yes                      |
| HOG                               | Yes                         | No                       |
| SIFT                              | Yes                         | No                       |
| LBP                               | Yes                         | No                       |
| CNN                               | No                          | Yes                      |
| Human domain knowledge            | High                        | Lower                    |
| Data requirements                 | Often lower                 | Often higher             |
| Representation flexibility         | Limited                     | High                     |
| Hierarchical features              | Limited                     | Strong                   |

---

## 39. Evolution of Computer Vision

The development of computer vision can be viewed as a transition:

```
Raw Pixels
    ↓
Manual Feature Engineering
    ↓
HOG / SIFT / LBP / GLCM
    ↓
Traditional Machine Learning
```

toward:

```
Raw Pixels
    ↓
Convolutional Neural Network
    ↓
Learned Feature Maps
    ↓
Learned Representation
    ↓
Classification
```

This transition is one of the most important changes brought by deep learning.

---

## 40. From Feature Engineering to Representation Learning

The key conceptual difference is:

### Feature Engineering

Humans decide:

> "What information should be extracted from the image?"

```
Image
 ↓
Human-designed algorithm
 ↓
Features
 ↓
Classifier
```

### Representation Learning

The model learns:

> "Which representation helps solve this task?"

```
Image
 ↓
Neural Network
 ↓
Training
 ↓
Learned Features
 ↓
Prediction
```

This is the fundamental idea behind Representation Learning.

---

## 41. Why CNNs Reduced the Need for Manual Features

CNNs introduced a powerful alternative to hand-crafted feature engineering.

Instead of explicitly designing:

- Edge Detector
- Texture Descriptor
- Shape Descriptor

we can provide images directly to a CNN:

```
Image
 ↓
CNN
 ↓
Learned Filters
 ↓
Feature Maps
 ↓
Deep Representation
 ↓
Prediction
```

During training, the network adjusts its parameters to minimize the loss function.

Therefore, the feature representation can adapt to the target task.

---

## 42. Important Concept: Feature Learning

In a CNN, the convolution filters are trainable parameters.

Initially:

- Random / Pretrained Weights

During training:

```
Forward Propagation
       ↓
Loss
       ↓
Backpropagation
       ↓
Gradient
       ↓
Weight Update
```

Repeated training gradually changes the filters.

The network therefore learns useful visual representations.

---

## 43. Manual Features and Explainability

Manual features can sometimes be easier to interpret because their definitions are explicit.

For example:

- Feature: Texture Contrast has a clear mathematical meaning.

A deep CNN feature may instead be represented by:

```
Activation Vector
[0.21, -0.04, 0.83, ...]
```

Its semantic meaning may not be immediately obvious.

This is one reason hand-crafted features remain useful for understanding traditional computer vision and feature engineering.

---

## 44. Key Concepts

The most important concepts are:

- Raw Image
    ↓
- Feature Engineering
    ↓
- Hand-Crafted Features
    ↓
- Feature Vector
    ↓
- Machine Learning

Common methods:

- Edges
- HOG
- SIFT
- LBP
- GLCM
- Color Features
- Shape Features
- Statistical Features

The central idea is:

> Convert raw image data into a manually designed numerical representation before classification.

---

## 45. Complete Comparison

### Traditional Computer Vision

```
Raw Image
                     ↓
              Preprocessing
                     ↓
          Manual Feature Extraction
                     ↓
       ┌─────────────┼─────────────┐
       ↓             ↓             ↓
      HOG           SIFT          LBP
       ↓             ↓             ↓
       └─────────────┼─────────────┘
                     ↓
               Feature Vector
                     ↓
              Machine Learning
                     ↓
                  Prediction
```

### Deep Learning

```
Raw Image
                     ↓
                    CNN
                     ↓
              Feature Maps
                     ↓
          Learned Representation
                     ↓
                 Classifier
                     ↓
                 Prediction
```

---

## 46. Key Takeaways

- Manual Feature Extraction is a traditional computer vision technique.
- It converts raw image data into manually designed numerical features.
- These features are often called hand-crafted features.
- Common features describe edges, textures, shapes, colors, gradients, or local structures.
- HOG, SIFT, LBP, and GLCM are important examples.
- The extracted features are usually represented as a feature vector.
- Traditional machine learning algorithms such as SVM can then operate on these vectors.
- Feature Extraction is different from Feature Selection.
- Manual feature extraction depends strongly on human knowledge and assumptions.
- Deep learning introduced Representation Learning, where useful features can be learned automatically from data.
- CNNs can learn hierarchical representations from raw images.
- The transition from hand-crafted features to learned representations is a fundamental development in modern computer vision.

---

## 47. Final Concept

The central difference can be summarized as:

### Traditional Computer Vision

```
Human
  ↓
Design Features
  ↓
Extract Features
  ↓
Feature Vector
  ↓
Machine Learning
```

versus:

### Deep Learning

```
Data
  ↓
Neural Network
  ↓
Learn Features
  ↓
Learn Representation
  ↓
Prediction
```

Therefore:

> Manual Feature Extraction asks humans to design the representation, while Representation Learning allows the model to learn useful representations directly from data.

