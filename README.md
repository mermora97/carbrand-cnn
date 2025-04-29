# 50 Car Brand Image Classifier Using Multiple CNNs

A deep‑learning project aimed at classifying **50 different car brands** in real‑world photographs. The project illustrates how data curation + transfer learning can turn a small, noisy dataset into a competitive image classifier.

This repository contains a deep learning project aimed at classifying images into **50 different car brands**. The dataset comes from [Kaggle](https://www.kaggle.com/datasets/yamaerenay/100-images-of-top-50-car-brands?select=companies.csv) and consists of 4,598 images. However, it includes a significant amount of noisy data (e.g., photos of people named “Hudson” in the Hudson category). To tackle this noise problem, three primary filtering strategies have been applied:

1. **Car vs. Non-Car Classification** (Transfer Learning with a ResNet50 model)
2. **Unsupervised Clustering** (to group and eliminate non-car clusters)
3. **Image Similarity / Outlier Detection** (to identify brand-specific outliers)

## Table of Contents

1. [1. Project Motivation](#project-overview)
2. [2. Dataset](#dataset)
3. [Filtering and Preprocessing](#filtering-and-preprocessing)
    - [1. Car vs. Non-Car Classification](#1-car-vs-non-car-classification)
    - [2. Unsupervised Clustering](#2-unsupervised-clustering)
    - [3. Image Similarity / Outlier Detection](#3-image-similarity--outlier-detection)
4. [Model Architecture](#model-architecture)
5. [How to Use](#how-to-use)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

## 1. Project Motivation
Car brand recognition powers applications ranging from automated insurance claims and smart traffic analytics to dealership inventory management. Real‑world images are messy, so this project focuses on cleaning the data first. With 50 brands, noisy labels, and real-world images, this dataset presents a challenging, multi-class classification problem that’s ideal for exploring deep learning techniques.

## 2. Dataset

- Source: [Kaggle – 100 images of top 50 car brands](https://www.kaggle.com/datasets/yamaerenay/100-images-of-top-50-car-brands?select=companies.csv)  
- Total Images: 4,598 (before filtering) ~ 100 images/brand  
- Challenges:
    - High intra‑class variance (angles, lighting, backgrounds).
    - Severe label noise: non‑car images. Some categories (e.g., “Hudson”) include photos of people named Hudson rather than the Hudson car brand.

You can download the dataset from Kaggle and place it in the `data/` folder within this repository (or update paths in the code accordingly).

## 3. Methodology
### 3.1. Filtering and Preprocessing


### 1. Car vs. Non-Car Classification

A **ResNet50** model pretrained on ImageNet is leveraged to identify whether an image contains a car. By mapping its output to known “car” or “vehicle” categories, images that are confidently not cars are filtered out.

1. Load ResNet50 with ImageNet weights.
2. Pass each image through the network.
3. If the highest-probability class belongs to a non-car category, label the image as noise.

### 2. Unsupervised Clustering

For additional noise reduction, an unsupervised clustering method is used on **feature embeddings** extracted from a pretrained model:

1. Extract feature vectors for each image (e.g., from ResNet50’s penultimate layer).
2. Run a clustering algorithm (e.g., K-Means or DBSCAN) on the embeddings.
3. Inspect each cluster to label entire clusters that do not contain cars or that clearly do not match their assigned brand category.

### 3.2. Model Training 

After cleaning the dataset, the final classification model is trained using a deep CNN:
- Architecture: **ResNet50** pretrained on ImageNet; replace FC layer with a 50‑unit head.
- Loss / Optimizer: Cross‑entropy, AdamW, cyclic LR.
- Data Augmentation: Random crop, horizontal flip, color jitter.
- Split: 70 % train · 15 % val · 15 % test (stratified by brand).

## 4. Results

## Roadmap
