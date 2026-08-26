# Traditional CNN

A structured documentation repository covering the fundamentals of Neural Networks, Deep Learning, Convolutional Neural Networks (CNNs), Training, and Traditional Feature Engineering techniques.

The purpose of this repository is to build a clear conceptual understanding of how image-based machine learning systems work, from basic neural networks to CNNs and classical handcrafted feature extraction methods.

---

📚 Documentation

The documentation is organized into four main sections.

---

🧠 Foundations

This section covers the fundamental concepts required to understand neural networks and deep learning.

- **Perceptron** (docs/Foundation/Perceptron.md)  
  The Perceptron is one of the simplest mathematical models of a neuron and provides the foundation for understanding artificial neural networks.

- **Neural Networks** (docs/Foundation/Neural_Networks.md)  
  Introduces the basic concepts of artificial neural networks, including neurons, weights, biases, layers, and network structure.

- **Deep Neural Networks** (docs/Foundation/Deep_Neural_Networks.md)  
  Explains how neural networks become deeper architectures and how multiple layers allow models to learn increasingly complex representations.

---

🏋️ Training

This section explains how neural networks learn from data and optimize their parameters.

- **Forward Propagation** (docs/Training/Forward_Propagation.md)  
  Explains how input data moves through the network from the input layer to the output layer to produce predictions.

- **Loss Function** (docs/Training/Loss_Function.md)  
  Explains how the difference between the model's prediction and the target value is measured using a loss function.

- **Backpropagation** (docs/Training/BackPropagation.md)  
  Explains how the error is propagated backward through the network to calculate gradients for the model parameters.

- **Gradient Descent** (docs/Training/Gradient_Descent.md)  
  Explains how gradients are used to update model parameters and minimize the loss during training.

---

🖼️ Convolutional Neural Networks

This section covers the fundamental components and operations of Convolutional Neural Networks.

- **CNN Introduction** (docs/CNN/CNN_Introduction.md)  
  Introduces Convolutional Neural Networks and explains why they are particularly useful for image processing and computer vision.

- **Filters and Kernels** (docs/CNN/Filters_and_Kernels.md)  
  Explains the concept of filters and kernels and how they are used to detect visual patterns in images.

- **Convolution** (docs/CNN/Convolution.md)  
  Explains the convolution operation and how a kernel moves across an image to generate new representations.

- **Feature Maps** (docs/CNN/Feature_Maps.md)  
  Explains how convolution produces feature maps and how these maps represent visual patterns detected by CNN filters.

- **Activation Functions** (docs/CNN/Activation_Functions.md)  
  Explains activation functions and their role in introducing non-linearity into neural networks.

- **Pooling** (docs/CNN/Pooling.md)  
  Explains pooling operations such as Max Pooling and Average Pooling and how they reduce the spatial dimensions of feature maps.

- **Fully Connected Layer** (docs/CNN/Fully_Connected_Layer.md)  
  Explains the role of fully connected layers in CNNs and how extracted features are transformed into final predictions.

---

🔍 Feature Engineering

This section focuses on classical image feature extraction techniques that were widely used before deep learning-based feature learning became dominant.

- **Manual Feature Extraction** (docs/Feature_Engineering/Manual_Feeature_Extraction.md)  
  Introduces the concept of manually designing and extracting meaningful features from images.

- **HOG — Histogram of Oriented Gradients** (docs/Feature_Engineering/HOG.md)  
  Explains HOG, a classical feature descriptor that represents objects using the distribution of local gradient orientations.

- **SIFT — Scale-Invariant Feature Transform** (docs/Feature_Engineering/SIFT.md)  
  Explains SIFT and how it detects and describes distinctive local image features while providing robustness to scale and rotation changes.

- **Texture Features** (docs/Feature_Engineering/Texture_Features.md)  
  Introduces techniques for representing image texture and extracting information from local intensity patterns and spatial relationships.

---

🗂️ Repository Structure

```
Traditional-CNN/
│
├── README.md
│
└── docs/
    │
    ├── CNN/
    │   ├── Activation_Functions.md
    │   ├── CNN_Introduction.md
    │   ├── Convolution.md
    │   ├── Feature_Maps.md
    │   ├── Filters_and_Kernels.md
    │   ├── Fully_Connected_Layer.md
    │   └── Pooling.md
    │
    ├── Feature_Engineering/
    │   ├── HOG.md
    │   ├── Manual_Feeature_Extraction.md
    │   ├── SIFT.md
    │   └── Texture_Features.md
    │
    ├── Foundation/
    │   ├── Deep_Neural_Networks.md
    │   ├── Neural_Networks.md
    │   └── Perceptron.md
    │
    └── Training/
        ├── BackPropagation.md
        ├── Forward_Propagation.md
        ├── Gradient_Descent.md
        └── Loss_Function.md
```

---

🗺️ Suggested Learning Path

A recommended order for studying the documentation is:

1. [Perceptron](docs/Foundation/Perceptron.md)
2. [Neural Networks](docs/Foundation/Neural_Networks.md)
3. [Deep Neural Networks](docs/Foundation/Deep_Neural_Networks.md)
4. [Forward Propagation](docs/Training/Forward_Propagation.md)
5. [Loss Function](docs/Training/Loss_Function.md)
6. [Backpropagation](docs/Training/BackPropagation.md)
7. [Gradient Descent](docs/Training/Gradient_Descent.md)
8. [CNN Introduction](docs/CNN/CNN_Introduction.md)
9. [Filters and Kernels](docs/CNN/Filters_and_Kernels.md)
10. [Convolution](docs/CNN/Convolution.md)
11. [Feature Maps](docs/CNN/Feature_Maps.md)
12. [Activation Functions](docs/CNN/Activation_Functions.md)
13. [Pooling](docs/CNN/Pooling.md)
14. [Fully Connected Layer](docs/CNN/Fully_Connected_Layer.md)
15. [Manual Feature Extraction](docs/Feature_Engineering/Manual_Feeature_Extraction.md)
16. [HOG](docs/Feature_Engineering/HOG.md)
17. [SIFT](docs/Feature_Engineering/SIFT.md)
18. [Texture Features](docs/Feature_Engineering/Texture_Features.md)

---

🎯 Goals

The main goals of this repository are to:

- Understand the foundations of artificial neural networks.
- Understand how neural networks are trained.
- Learn the core concepts behind CNNs.
- Understand convolution, filters, kernels, and feature maps.
- Understand the role of activation functions and pooling.
- Understand how CNNs perform image feature extraction.
- Learn classical handcrafted feature extraction methods.
- Build a bridge between traditional computer vision and deep learning.

---

📖 Topics Covered

**Foundations**

- Perceptron
- Neural Networks
- Deep Neural Networks

**Training**

- Forward Propagation
- Loss Functions
- Backpropagation
- Gradient Descent

**CNN**

- CNN Introduction
- Filters and Kernels
- Convolution
- Feature Maps
- Activation Functions
- Pooling
- Fully Connected Layers

**Feature Engineering**

- Manual Feature Extraction
- HOG
- SIFT
- Texture Features
```
