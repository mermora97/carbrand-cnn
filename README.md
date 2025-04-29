# 50 Car Brand Image Classifier Using Multiple CNNs

A deep‑learning project aimed at classifying **50 different car brands** in real‑world photographs. The project illustrates how data curation + transfer learning can turn a small, noisy dataset into a competitive image classifier.

## 1. Project Motivation
Car brand recognition powers applications ranging from automated insurance claims and smart traffic analytics to dealership inventory management. Real‑world images are messy, so this project focuses on cleaning the data first. With 50 brands, noisy labels, and real-world images, this dataset presents a challenging, multi-class classification problem that’s ideal for exploring deep learning techniques.

## 2. Dataset
- Source: [Kaggle – 100 images of top 50 car brands](https://www.kaggle.com/datasets/yamaerenay/100-images-of-top-50-car-brands?select=companies.csv)  
- Total Images: 4,598 across 50 car brands ≈ 100 per brand  
- Challenges:
    - High intra‑class variance (angles, lighting, backgrounds).
    - Severe label noise: non‑car images e.g. “Hudson” folder include photos of people named Hudson instead of cars.

Here's a 4×5 grid of 20 randomly selected car-brand images from the original dataset:
<img src="docs/figures/initial_sample.png" width="600"/>

To download the data:
- Run the helper script (recommended):
```bash
bash scripts/download_data.sh
```
- Or manually download the dataset from Kaggle and extract it into the data/ folder.

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
