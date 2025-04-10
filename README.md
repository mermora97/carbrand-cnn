# Car Brand Image Classification Project

This repository contains a deep learning project aimed at classifying images into **50 different car brands**. The dataset comes from [Kaggle](https://www.kaggle.com/datasets/yamaerenay/100-images-of-top-50-car-brands?select=companies.csv) and consists of 4,598 images. However, it includes a significant amount of noisy data (e.g., photos of people named “Hudson” in the Hudson category). To tackle this noise problem, three primary filtering strategies have been applied:

1. **Car vs. Non-Car Classification** (Transfer Learning with a ResNet50 model)
2. **Unsupervised Clustering** (to group and eliminate non-car clusters)
3. **Image Similarity / Outlier Detection** (to identify brand-specific outliers)

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Filtering and Preprocessing](#filtering-and-preprocessing)
    - [1. Car vs. Non-Car Classification](#1-car-vs-non-car-classification)
    - [2. Unsupervised Clustering](#2-unsupervised-clustering)
    - [3. Image Similarity / Outlier Detection](#3-image-similarity--outlier-detection)
4. [Model Architecture](#model-architecture)
5. [How to Use](#how-to-use)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)

---

## Project Overview

**Goal**: Train a robust image classifier for 50 car brands.

However, the dataset includes a variety of “noise” images that are irrelevant to a specific car brand (such as images of people or unrelated objects). Therefore, before training a final classifier, three filtering strategies were used to remove non-car or irrelevant images.

---

## Dataset

- **Source**: [Kaggle – 100 images of top 50 car brands](https://www.kaggle.com/datasets/yamaerenay/100-images-of-top-50-car-brands?select=companies.csv)  
- **Total Images**: ~4,598 (before filtering)  
- **Noise Examples**: Some categories (e.g., “Hudson”) include photos of people named Hudson rather than the Hudson car brand.

You can download the dataset from Kaggle and place it in the `data/` folder within this repository (or update paths in the code accordingly).

---

## Filtering and Preprocessing

Because the raw images contain noise, the project implements three main filtering techniques:

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

### 3. Image Similarity / Outlier Detection

Within each brand’s subset of images, image-to-image similarity is computed to detect outliers:

1. Compute feature vectors for each image.
2. Calculate a similarity (or distance) metric for all image pairs in the same brand.
3. Remove images with low similarity to the majority (likely noise or incorrect brand entries).

---

## Model Architecture

After cleaning the dataset, the final classification model is trained on **50 car brand classes** using a deep CNN:

1. Start with a pretrained **ResNet50** (or similar CNN).
2. Replace the final layer to output 50 classes.
3. Fine-tune the model on the filtered dataset.

---

## How to Use

1. **Installation**:  
   ```bash
   git clone https://github.com/yourusername/car-brand-classifier.git
   cd car-brand-classifier
   pip install -r requirements.txt
