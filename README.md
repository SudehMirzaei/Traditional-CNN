## Traditional CNN

A structured collection of fundamental concepts and techniques related to Traditional Convolutional Neural Networks (CNNs), classical feature extraction, neural networks, and deep learning.

This repository is designed as a learning and reference resource for understanding how image classification systems are built, starting from fundamental neural network concepts and progressing toward convolutional neural networks and handcrafted feature extraction methods.

---


🧠 Foundations

This section introduces the fundamental concepts required to understand neural networks and deep learning.

**Perceptron**

The Perceptron is one of the simplest computational models of a neuron and provides the foundation for understanding neural networks.

📖 "Read Perceptron" (docs/Foundation/Perceptron.md)

**Neural Networks**

Introduces the basic structure and concepts behind Artificial Neural Networks (ANNs), including neurons, weights, biases, and layers.

📖 "Read Neural Networks" (docs/Foundation/Neural_Networks.md)

**Deep Neural Networks**

Explains how neural networks can be extended into deeper architectures and why depth is important for learning complex representations.

📖 "Read Deep Neural Networks" (docs/Foundation/Deep_Neural_Networks.md)

---

🧮 Training

This section explains how neural networks learn from data.

**Forward Propagation**

Explains how input data moves through the network to produce predictions.

📖 "Read Forward Propagation" (docs/Training/Forward_Propagation.md)

**Loss Function**

Introduces loss functions and explains how they measure the difference between predictions and target values.

📖 "Read Loss Function" (docs/Training/Loss_Function.md)

**Backpropagation**

Explains how errors are propagated backward through a neural network to calculate gradients.

📖 "Read Backpropagation" (docs/Training/BackPropagation.md)

**Gradient Descent**

Explains how optimization algorithms use gradients to update model parameters and minimize the loss function.

📖 "Read Gradient Descent" (docs/Training/Gradient_Descent.md)

---

🖼️ Convolutional Neural Networks

This section focuses on the fundamental components of Convolutional Neural Networks (CNNs) and how CNNs process images.

**CNN Introduction**

Provides an introduction to CNNs, their motivation, architecture, and applications in computer vision.

📖 "Read CNN Introduction" (docs/CNN/CNN_Introduction.md)

**Filters and Kernels**

Explains the role of filters and kernels in extracting visual patterns from images.

📖 "Read Filters and Kernels" (docs/CNN/Filters_and_Kernels.md)

**Convolution**

Explains the convolution operation and how kernels interact with image pixels to generate new representations.

📖 "Read Convolution" (docs/CNN/Convolution.md)

**Feature Maps**

Explains how convolution operations produce feature maps and how these maps represent visual information learned by the network.

📖 "Read Feature Maps" (docs/CNN/Feature_Maps.md)

**Activation Functions**

Introduces activation functions and explains their role in introducing non-linearity into neural networks.

📖 "Read Activation Functions" (docs/CNN/Activation_Functions.md)

**Pooling**

Explains pooling operations such as Max Pooling and Average Pooling and how they reduce spatial dimensions while preserving important information.

📖 "Read Pooling" (docs/CNN/Pooling.md)

**Fully Connected Layer**

Explains the role of fully connected layers in CNN architectures and how extracted features are transformed into final predictions.

📖 "Read Fully Connected Layer" (docs/CNN/Fully_Connected_Layer.md)

---

🔍 Feature Engineering

Before the widespread use of deep learning, computer vision systems often relied on handcrafted features designed to represent important visual patterns.

This section introduces several classical feature extraction techniques.

**Manual Feature Extraction**

Introduces the general concept of manually designing and extracting useful features from images.

📖 "Read Manual Feature Extraction" (docs/Feature_Engineering/Manual_Feeature_Extraction.md)

**HOG — Histogram of Oriented Gradients**

HOG is a classical feature descriptor that represents objects based on the distribution of local gradient orientations.

📖 "Read HOG" (docs/Feature_Engineering/HOG.md)

**SIFT — Scale-Invariant Feature Transform**

SIFT is a local feature extraction method designed to detect and describe distinctive image features while providing robustness to scale and rotation changes.

📖 "Read SIFT" (docs/Feature_Engineering/SIFT.md)

**Texture Features**

Introduces methods for representing image texture and extracting information related to local intensity patterns and spatial relationships.

📖 "Read Texture Features" (docs/Feature_Engineering/Texture_Features.md)

---

🗺️ Learning Path

The documentation can be studied in the following order:

- Perceptron
    - ↓
- Neural Networks
    - ↓
- Deep Neural Networks
    - ↓
- Forward Propagation
    - ↓
- Loss Function
    - ↓
- Backpropagation
    - ↓
- Gradient Descent
    - ↓
- CNN Introduction
    - ↓
- Filters & Kernels
    - ↓
- Convolution
    - ↓
- Feature Maps
    - ↓
- Activation Functions
    - ↓
- Pooling
    - ↓
- Fully Connected Layer
    - ↓
- Feature Engineering
    - ├── Manual Feature Extraction
    - ├── HOG
    - ├── SIFT
    - └── Texture Features

---

🎯 Goals

The main goals of this repository are:

- Build a strong foundation in neural networks and deep learning.
- Understand the mathematical and conceptual foundations of CNNs.
- Understand how convolution transforms image data.
- Learn how filters and kernels extract visual patterns.
- Understand feature maps and spatial representations.
- Understand the role of pooling and activation functions.
- Learn the basic training process of neural networks.
- Understand classical handcrafted feature extraction methods.
- Compare traditional feature engineering approaches with learned CNN representations.

---

📖 Documentation

| Section | Description |
|---------|-------------|
| "Foundations" (docs/Foundation/) | Neural network fundamentals |
| "Training" (docs/Training/) | Learning and optimization |
| "CNN" (docs/CNN/) | Convolutional neural network concepts |
| "Feature Engineering" (docs/Feature_Engineering/) | Classical image feature extraction |

---

🛠️ Topics Covered

- Perceptrons
- Artificial Neural Networks
- Deep Neural Networks
- Forward Propagation
- Loss Functions
- Backpropagation
- Gradient Descent
- Convolutional Neural Networks
- Filters and Kernels
- Convolution
- Feature Maps
- Activation Functions
- Pooling
- Fully Connected Layers
- Manual Feature Extraction
- HOG
- SIFT
- Texture Features

---

📌 Purpose

This repository is primarily intended for learning, documentation, and conceptual understanding of traditional computer vision and CNN-based image processing.

The documentation follows a progression from basic neural network concepts to CNN architectures and classical feature engineering techniques.
```
