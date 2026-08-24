# Texture Features

## 1. Introduction

**Texture Features** are visual features that describe the local or global patterns, intensity variations, and spatial relationships within an image.

While color describes **what intensity or color values are present**, texture describes **how those values are arranged and vary across a region**.

For example:

```text
Smooth Surface

████████████
████████████
████████████
████████████

versus:

Rough Surface

██░█░██░░█
░██░░███░
█░░██░█░█
██░█░░██

Both images may have similar average intensity, but their spatial patterns are different.

Texture features attempt to capture these differences numerically.

---

## 2. Why Are Texture Features Important?

Texture can contain important information about an object's:

- Surface structure
- Material
- Local patterns
- Regularity
- Roughness
- Homogeneity
- Spatial organization

For example, two regions may have similar colors but different textures:

```
Region A
████████████

Region B
██░█░░██░█░
░██░█░██░░
```

A color-based feature may consider them similar, while a texture-based feature can distinguish them.

---

## 3. What Is Texture?

Texture can be understood as a pattern of local intensity or color variations.

Consider a grayscale image:

```
10  10  10  10
10  10  10  10
10  10  10  10
10  10  10  10
```

There is almost no local variation.

Therefore, the image has a very uniform texture.

Now consider:

```
10  80  20  90
70  30  90  15
20  85  10  75
95  25  80  30
```

There are strong local intensity variations.

This produces a more complex texture.

---

## 4. Texture Is About Relationships

A key idea in texture analysis is:

> Texture is not only about individual pixel values; it is also about the relationships between neighboring pixels.

For example:

- Pixel A = 100
- Pixel B = 102

The difference is small.

But:

- Pixel A = 20
- Pixel B = 200

The difference is large.

Texture descriptors analyze these local relationships to characterize the image.

---

## 5. Texture Feature Extraction Pipeline

A traditional texture analysis pipeline can be represented as:

```
Input Image
     ↓
Preprocessing
     ↓
Texture Analysis
     ↓
Texture Descriptor
     ↓
Texture Features
     ↓
Feature Vector
     ↓
Machine Learning Model
     ↓
Prediction
```

For example:

```
Image
  ↓
GLCM
  ↓
Contrast
Correlation
Energy
Homogeneity
  ↓
Feature Vector
  ↓
Classifier
```

---

## 6. Types of Texture Analysis

Texture features can be categorized in several ways.

Common approaches include:

- Statistical Methods
- Structural Methods
- Model-Based Methods
- Transform-Based Methods
- Local Pattern Methods

Some of the most commonly used techniques include:

- GLCM
- LBP
- Gabor Filters
- Wavelet Features
- Laws' Texture Energy
- Local Phase Quantization

---

## 7. Statistical Texture Features

Statistical texture analysis uses statistics of image intensities and their spatial relationships.

Examples include:

- Mean
- Variance
- Contrast
- Correlation
- Energy
- Entropy
- Homogeneity

These features attempt to summarize the texture numerically.

---

## 8. First-Order Statistics

First-order statistics analyze individual pixel intensities without explicitly considering relationships between neighboring pixels.

For example:

```
10  20  20
30  40  50
50  60  70
```

We can calculate statistics such as:

- Mean
- Variance
- Standard deviation
- Minimum
- Maximum
- Histogram
- Entropy

These provide information about the distribution of pixel intensities.

However, they do not explicitly capture spatial relationships.

---

## 9. Mean

The mean represents the average intensity.

For (N) pixels:

\[
\mu =
\frac{1}{N}
\sum_{i=1}^{N}x_i
\]

where:

- \(x_i\) is the intensity of pixel \(i\)
- \(N\) is the number of pixels

A higher mean generally indicates a brighter region.

However:

> Mean intensity alone is not a texture descriptor.

Two images can have the same mean but very different textures.

---

## 10. Variance

Variance measures how much pixel intensities differ from the mean.

\[
\sigma^2 =
\frac{1}{N}
\sum_{i=1}^{N}(x_i-\mu)^2
\]

A low variance often corresponds to a more uniform region.

A high variance indicates stronger intensity variation.

Conceptually:

**Low Variance**

```
50  51  49
50  50  51
49  50  50
```

versus:

**High Variance**

```
10  90  20
80  15  95
25  70  5
```

Variance can therefore provide a simple measure of texture complexity.

---

## 11. Entropy

Entropy measures the uncertainty or randomness of an intensity distribution.

For probabilities \(p_i\):

\[
H=-\sum_i p_i\log_2(p_i)
\]

A highly uniform image may have relatively low entropy.

A more varied intensity distribution may have higher entropy.

Conceptually:

**Uniform Region**
      ↓
Low Intensity Diversity
      ↓
Lower Entropy

while:

**Complex Region**
      ↓
High Intensity Diversity
      ↓
Higher Entropy

---

## 12. Gray-Level Co-occurrence Matrix

One of the most important traditional texture descriptors is:

> Gray-Level Co-occurrence Matrix (GLCM)

GLCM captures how frequently pairs of gray-level values occur at a particular spatial relationship.

Instead of asking:

> "What is the intensity of this pixel?"

GLCM asks:

> "How often does a pixel with intensity (i) occur next to a pixel with intensity (j)?"

---

## 13. GLCM Concept

Suppose an image contains:

```
1  1  2
1  2  2
2  2  3
```

For a specific direction and distance, we examine neighboring pixel pairs.

For example:

```
1 → 1
1 → 2
2 → 2
2 → 3
```

The frequencies of these pairs are stored in a matrix.

Conceptually:

```
Neighbor
       1  2  3
     ┌─────────
  1  │ a  b  c
  2  │ d  e  f
  3  │ g  h  i
```

The matrix represents the frequency of intensity transitions.

---

## 14. GLCM Distance and Direction

GLCM depends on the spatial relationship between pixels.

For example:

**Horizontal**:

```
● → ●
```

or:

**Vertical**:

```
●
↓
●
```

or:

**Diagonal**:

```
●
  ↘
    ●
```

Therefore, GLCM can be calculated at different:

- Distances
- Angles

Common directions include:

- 0°
- 45°
- 90°
- 135°

This allows texture orientation to be analyzed.

---

## 15. GLCM Contrast

GLCM Contrast measures local intensity differences.

A common definition is:

\[
Contrast =
\sum_{i,j}(i-j)^2P(i,j)
\]

where:

- (i) and (j) are gray levels
- \(P(i,j)\) is the normalized GLCM

Large differences between neighboring intensities produce larger contrast values.

Therefore:

**Strong Intensity Differences**
          ↓
       **High Contrast**

---

## 16. GLCM Homogeneity

Homogeneity measures how close the values in the GLCM are to its diagonal.

One common formulation is:

\[
Homogeneity =
\sum_{i,j}
\frac{P(i,j)}
{1+|i-j|}
\]

If neighboring pixels tend to have similar intensity values:

\(i \approx j\)

the homogeneity tends to be higher.

Therefore:

**Uniform Texture**
      ↓
**High Homogeneity**

---

## 17. GLCM Energy

Energy measures the uniformity of the GLCM.

A common definition is:

\[
Energy =
\sum_{i,j}P(i,j)^2
\]

A highly concentrated GLCM tends to have higher energy.

Conceptually:

**Regular / Uniform Texture**
          ↓
**Concentrated GLCM**
          ↓
**Higher Energy**

---

## 18. GLCM Correlation

Correlation measures the relationship between the gray levels of neighboring pixels.

A common formulation is:

\[
Correlation =
\sum_{i,j}
\frac{(i-\mu_i)(j-\mu_j)P(i,j)}
{\sigma_i\sigma_j}
\]

It attempts to capture how strongly neighboring pixel intensities are related.

---

## 19. GLCM Feature Vector

Several GLCM properties can be combined:

- Contrast
- Correlation
- Energy
- Homogeneity

For example:

```
[
  12.4,
  0.81,
  0.35,
  0.72
]
```

This becomes a texture feature vector.

```
Image
 ↓
GLCM
 ↓
Texture Statistics
 ↓
Feature Vector
```

---

## 20. Local Binary Pattern

Another important texture descriptor is:

> Local Binary Pattern (LBP)

LBP analyzes the relationship between a central pixel and its neighbors.

Consider:

```
120  130  100
140  125   90
110  150  135
```

The center pixel is:

```
125
```

Each neighboring pixel is compared with 125.

If:

\[
Neighbor \ge Center
\]

we assign:

1

Otherwise:

0

This produces a binary pattern.

---

## 21. LBP Example

Suppose the comparisons produce:

```
1  1  0
1     0
0  1  1
```

The neighboring values form a binary sequence.

This binary pattern is converted into a numerical code.

Repeating this process across the image produces many LBP codes.

These codes can then be represented as a histogram.

```
Image
 ↓
LBP Codes
 ↓
Histogram
 ↓
Texture Feature Vector
```

---

## 22. Why LBP Is Useful

LBP is useful because it describes local texture patterns.

It can capture patterns such as:

- Flat Region
- Edge
- Corner
- Spot
- Texture Pattern

LBP is computationally simple and has been widely used in traditional computer vision.

---

## 23. Gabor Filters

Gabor filters are another powerful method for texture analysis.

A Gabor filter is sensitive to specific:

- Frequencies
- Orientations

Conceptually:

```
Image
  ↓
Gabor Filters
  ↓
Different Orientations
  ↓
Different Frequencies
  ↓
Texture Responses
```

A collection of Gabor filters can analyze texture at multiple scales and orientations.

---

## 24. Gabor Filter Bank

Instead of using one Gabor filter, we can use multiple filters.

For example:

```
Image
               ↓
       ┌───────┼───────┐
       ↓       ↓       ↓
      0°      45°     90°
       ↓       ↓       ↓
   Response Response Response
       └───────┼───────┘
               ↓
        Texture Features
```

This makes Gabor filters useful for analyzing oriented textures.

---

## 25. Wavelet-Based Texture Features

Wavelet transforms analyze images at multiple scales.

Conceptually:

```
Image
 ↓
Wavelet Transform
 ↓
Multiple Scales
 ↓
Frequency Components
 ↓
Texture Features
```

Wavelets can capture both:

- Spatial information
- Frequency information

This makes them useful for textures that exist at different scales.

---

## 26. Texture Feature Vector

After applying one or more texture descriptors, we can create a feature vector.

For example:

```
GLCM Contrast       = 12.4
GLCM Energy         = 0.35
GLCM Homogeneity    = 0.72
LBP Entropy         = 4.21
Gabor Response      = 0.63
```

The final representation might be:

```
[
  12.4,
  0.35,
  0.72,
  4.21,
  0.63
]
```

This vector can be passed to a classifier.

---

## 27. Texture Features + Machine Learning

A traditional classification pipeline may look like:

```
Image
 ↓
Texture Feature Extraction
 ↓
Feature Vector
 ↓
SVM / Random Forest / Logistic Regression
 ↓
Prediction
```

For example:

```
Image
 ↓
GLCM
 ↓
Texture Features
 ↓
SVM
 ↓
Class
```

The machine learning model learns the relationship between texture features and class labels.

---

## 28. Texture Features in Medical Imaging

Texture analysis has historically been important in medical imaging.

Medical images often contain subtle differences in tissue structure.

Texture features can potentially capture differences that are difficult to describe using only simple intensity statistics.

Examples include:

- MRI
- CT
- X-Ray
- Ultrasound
- Dermoscopic Images

A simplified pipeline is:

```
Medical Image
      ↓
Region of Interest
      ↓
Texture Extraction
      ↓
Texture Features
      ↓
Feature Vector
      ↓
Classifier
```

---

## 29. Texture Features in MRI

For MRI images, texture descriptors can be used to quantify local intensity patterns.

For example:

```
MRI
 ↓
Tumor / Region of Interest
 ↓
GLCM
 ↓
Contrast
Correlation
Energy
Homogeneity
 ↓
Feature Vector
```

These features can then be used by a machine learning model.

However, texture features should not automatically be interpreted as direct biological properties. Their usefulness depends on image acquisition, preprocessing, region selection, and the specific task.

---

## 30. Texture Features in Skin Lesions

Texture can also be relevant in dermoscopic image analysis.

A skin lesion may contain different local visual patterns.

A traditional pipeline could be:

```
Dermoscopic Image
       ↓
Preprocessing
       ↓
Lesion Segmentation
       ↓
Texture Extraction
       ↓
GLCM / LBP / Gabor
       ↓
Texture Feature Vector
       ↓
Classifier
```

Texture may be combined with:

- Color Features
- Shape Features
- Border Features

to create a larger representation.

---

## 31. Combining Texture, Shape, and Color

A traditional system can combine multiple feature categories.

```
Image
                      ↓
          ┌───────────┼───────────┐
          ↓           ↓           ↓
       Texture      Shape       Color
          ↓           ↓           ↓
       GLCM/HOG    Geometry    Histogram
          ↓           ↓           ↓
          └───────────┼───────────┘
                      ↓
             Combined Feature Vector
                      ↓
                  Classifier
```

This is an example of Feature Fusion.

---

## 32. Texture Feature Extraction vs CNN

Traditional texture extraction:

```
Image
 ↓
GLCM / LBP / Gabor
 ↓
Texture Features
 ↓
Classifier
```

CNN-based approach:

```
Image
 ↓
Convolution
 ↓
Feature Maps
 ↓
Deeper Layers
 ↓
Learned Representation
 ↓
Classifier
```

The major difference is that traditional descriptors are designed by humans, while CNN filters are learned from data.

---

## 33. Texture Features and CNNs

It is important to understand that CNNs can also learn texture-related representations.

For example:

```
Early CNN Layers
      ↓
Edges
      ↓
Simple Texture Patterns
      ↓
More Complex Patterns
      ↓
High-Level Structures
```

However, the network is not explicitly told:

> "This filter must detect texture."

Instead, the filters are optimized through training to minimize the task loss.

---

## 34. Hand-Crafted Texture vs Learned Texture

### Hand-Crafted

```
Image
 ↓
LBP
 ↓
Texture Descriptor
```

The extraction rules are predefined.

### Learned

```
Image
 ↓
CNN
 ↓
Learned Filters
 ↓
Feature Maps
```

The network learns filters that are useful for the task.

This difference is a central distinction between traditional feature engineering and representation learning.

---

## 35. Advantages of Texture Features

Manual texture descriptors have several advantages:

1. **Interpretability**: Many texture measurements have clear mathematical definitions.

2. **Low Computational Cost**: Some descriptors are relatively inexpensive to compute.

3. **Small Dataset Compatibility**: They can be useful when the available dataset is too small for training a large neural network effectively.

4. **Domain Knowledge**: They allow researchers to explicitly incorporate knowledge about texture.

5. **Classical Baselines**: They are useful for comparing traditional machine learning approaches against deep learning.

---

## 36. Limitations of Texture Features

1. **Hand-Crafted Design**: The descriptor is designed independently of the final prediction task.

2. **Limited Adaptability**: A texture descriptor that works well in one dataset may perform poorly in another.

3. **Sensitivity to Imaging Conditions**: Changes in:
   - Lighting
   - Resolution
   - Contrast
   - Noise
   - Acquisition settings

can affect texture measurements.

4. **Limited Representation Capacity**: A small set of texture statistics may not capture all the information required for a complex classification problem.

5. **Region Dependence**: Texture features can depend strongly on which region of the image is analyzed.

---

## 37. Texture Feature Engineering Pipeline

A complete traditional workflow can be:

```
Raw Image
                     ↓
                Preprocessing
                     ↓
             Region Selection
                     ↓
             Texture Extraction
                     ↓
        ┌────────────┼────────────┐
        ↓            ↓            ↓
       GLCM         LBP         Gabor
        ↓            ↓            ↓
        └────────────┼────────────┘
                     ↓
              Feature Vector
                     ↓
             Feature Selection
                     ↓
              Machine Learning
                     ↓
                 Prediction
```

---

## 38. Texture Feature Comparison

| Method                   | Main Idea                               | Typical Information               |
|--------------------------|-----------------------------------------|-----------------------------------|
| GLCM                     | Pixel pair relationships                | Texture statistics                |
| LBP                      | Local binary patterns                   | Local texture                     |
| Gabor Filters            | Frequency + orientation                 | Oriented texture                  |
| Wavelet                  | Multi-scale analysis                    | Spatial-frequency structure        |
| Variance                 | Intensity variation                     | Texture roughness                 |
| Entropy                  | Intensity uncertainty                   | Texture complexity                 |

---

## 39. Important Concept: Texture Is Spatial

One of the most important ideas in texture analysis is that texture is strongly related to spatial organization.

Consider:

**Image A:**

```
10 10 10
10 10 10
10 10 10
```

and:

**Image B:**

```
10 20 10
20 10 20
10 20 10
```

The individual intensity values may overlap, but their spatial arrangements are different.

Texture descriptors attempt to capture this organization.

---

## 40. Texture and Receptive Regions

Texture is often analyzed within local regions.

For example:

```
Image
┌─────────────────────┐
│                     │
│   ┌───────────┐     │
│   │ Local     │     │
│   │ Region    │     │
│   └───────────┘     │
│                     │
└─────────────────────┘
```

A descriptor such as LBP or GLCM can analyze the local intensity relationships within this region.

Different regions can then produce different texture descriptors.

---

## 41. Local vs Global Texture

Texture features can be computed at different levels.

### Local Texture

Analyzes small neighborhoods.

Examples:

- LBP
- Local Gabor Responses

### Global Texture

Analyzes the overall texture distribution.

Examples:

- Global Histogram
- Overall Entropy
- Aggregated GLCM Statistics

Both approaches can provide useful information.

---

## 42. Multi-Scale Texture

Texture can exist at different spatial scales.

For example:

- **Fine Texture**

```
░█░█░█░█░█░█
```

- **Medium Texture**

```
██░░██░░██░░
```

- **Coarse Texture**

```
████████░░░░
████████░░░░
```

Multi-scale methods attempt to capture these different levels of structure.

Examples include:

- Wavelets
- Multi-scale Gabor filters
- Multi-scale LBP

---

## 43. Texture Feature Normalization

After extracting texture features, normalization may be necessary.

Suppose we have:

```
Contrast = 250
Energy = 0.31
Homogeneity = 0.72
```

The numerical ranges are very different.

A machine learning model may benefit from scaling them.

Common approaches include:

- Standardization
- Min-Max Scaling
- Robust Scaling

For standardization:

\[
z=\frac{x-\mu}{\sigma}
\]

where:

- \(x\) is the original feature
- \(\mu\) is the mean
- \(\sigma\) is the standard deviation

---

## 44. Texture Features and Feature Selection

A large number of texture descriptors can produce a high-dimensional feature vector.

For example:

```
GLCM
 ↓
4 directions
 ×
5 distances
 ×
4 statistics
```

This can generate many features.

Feature selection can then be applied:

```
Many Texture Features
        ↓
Feature Selection
        ↓
Most Informative Features
        ↓
Classifier
```

This can reduce redundancy and improve model efficiency.

---

## 45. Important Limitations in Medical Applications

When texture features are used in medical imaging, several factors should be carefully controlled.

These include:

- **Image resolution**
- **Scanner/device differences**
- **Acquisition protocols**
- **Preprocessing**
- **Segmentation**
- **Region of interest**
- **Noise**
- **Intensity normalization**

A texture descriptor can change when the imaging conditions change.

Therefore:

> Texture features should be interpreted in the context of the imaging pipeline rather than as completely invariant biological measurements.

---

## 46. Texture Features vs Raw Pixel Statistics

Consider two approaches.

### Raw Statistics

```
Image
 ↓
Mean
Variance
Histogram
```

These describe the distribution of intensity values.

### Texture Analysis

```
Image
 ↓
Spatial Relationships
 ↓
GLCM / LBP / Gabor
 ↓
Texture Features
```

Texture analysis provides information about how intensity values are spatially organized.

---

## 47. Texture Features vs Color Features

These two feature categories describe different properties.

### Color Features

Answer:

> What colors or intensities are present?

- Color Histogram
- Mean Color
- Color Variance

### Texture Features

Answer:

> How are local intensity/color patterns organized?

- GLCM
- LBP
- Gabor

They can therefore be complementary.

---

## 48. Texture Features vs Shape Features

### Shape Features

Describe geometry:

- Area
- Perimeter
- Circularity
- Aspect Ratio
- Contour

### Texture Features

Describe local visual patterns:

- Contrast
- Homogeneity
- Energy
- Local Patterns

A complete traditional vision system may use both.

---

## 49. Key Takeaways

Texture describes patterns and spatial variations within an image.

Texture is not simply the average pixel intensity.

Spatial relationships between neighboring pixels are often essential.

Texture features can describe:

- Contrast
- Homogeneity
- Energy
- Correlation
- Entropy
- Local patterns
- Frequency and orientation

GLCM captures relationships between neighboring gray levels.

LBP describes local binary intensity patterns.

Gabor filters analyze texture at different orientations and frequencies.

Texture features can be combined with shape and color features.

Traditional texture descriptors are hand-crafted.

CNNs can learn texture-related representations automatically.

Texture features are still useful for small datasets, classical machine learning, and interpretable feature engineering.

In medical imaging, texture measurements can be sensitive to acquisition, preprocessing, segmentation, and image quality.

---

## 50. Final Concept

The central idea of texture feature extraction is:

```
Image
                   ↓
        Analyze Local / Spatial
            Intensity Patterns
                   ↓
        ┌──────────┼──────────┐
        ↓          ↓          ↓
      GLCM        LBP       Gabor
        ↓          ↓          ↓
        └──────────┼──────────┘
                   ↓
            Texture Features
                   ↓
             Feature Vector
                   ↓
             Machine Learning
                   ↓
               Prediction
```

In traditional computer vision:

> Texture features are manually designed numerical representations that describe how visual patterns and intensity values are organized within an image.

In deep learning:

```
Image
 ↓
CNN
 ↓
Learned Feature Maps
 ↓
Learned Texture / Shape / Pattern Representations
 ↓
Prediction
```

The fundamental difference is that traditional methods define texture descriptors explicitly, while deep neural networks can learn useful representations from the training data.

