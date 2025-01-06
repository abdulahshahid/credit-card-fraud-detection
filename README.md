# Fraud Detection Using Machine Learning and Deep Learning

This project focuses on detecting fraudulent transactions using advanced machine learning (ML) and deep learning (DL) techniques. The system analyzes credit card transaction data to classify transactions as fraudulent or non-fraudulent while addressing challenges like class imbalance and data scalability.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Technologies Used](#technologies-used)
4. [Dataset](#dataset)
5. [Data Preprocessing](#data-preprocessing)
6. [Models Implemented](#models-implemented)
7. [Results](#results)
---

## Introduction

Fraud detection is critical in financial transactions. This project aims to leverage ML and DL models to improve the detection accuracy of fraudulent credit card transactions. It tackles challenges such as class imbalance by implementing synthetic oversampling techniques and integrates both traditional ML classifiers and neural networks for robust performance.

---

## Features

- **Data Exploration:** Identifies patterns and highlights class imbalance in the dataset.
- **Class Balancing:** Applies SMOTE (Synthetic Minority Oversampling Technique) to handle imbalanced data.
- **Machine Learning Models:** Implements Random Forest, SVM, and AdaBoost classifiers.
- **Deep Learning Models:** Develops a Keras-based neural network for improved accuracy.
- **Performance Evaluation:** Utilizes metrics like ROC-AUC, confusion matrix, and precision-recall curves.
- **Visualizations:** Provides insightful plots for data distribution, model performance, and trends.

---

## Technologies Used

- **Python:** Primary language for data manipulation and model development.
- **Libraries:**
  - `pandas`, `numpy` for data handling.
  - `seaborn`, `matplotlib` for visualization.
  - `scikit-learn`, `imblearn` for ML workflows and class balancing.
  - `tensorflow`, `keras` for deep learning implementation.

---

## Dataset

- **Source:** [Dataset from Kaggle](https://www.kaggle.com).
- **Details:** Contains anonymized credit card transaction data with features and labels (fraudulent or not).
- **Key Characteristics:**
  - 31 features including `Time`, `Amount`, and 28 anonymized variables.
  - Highly imbalanced classes.

---

## Data Preprocessing

1. **Missing Values:** Verified absence of null values.
2. **Class Balancing:** Applied SMOTE to address the imbalance.
3. **Scaling:** Standardized features using `StandardScaler`.
4. **Splitting:** Divided data into training and testing sets with an 80-20 split.

---

## Models Implemented

### Machine Learning Models

1. **Random Forest Classifier:**
   - Robust and interpretable ensemble model.
2. **Support Vector Machine (SVM):**
   - Kernel-based method for classification.
3. **AdaBoost Classifier:**
   - Adaptive boosting algorithm for better generalization.

### Deep Learning Model

1. **Neural Network:**
   - Architecture: Fully connected with ReLU activation and Dropout layers.
   - Optimizer: Adam.
   - Loss Function: Binary Crossentropy.

---

## Results

- **Evaluation Metrics:**
  - Accuracy, precision, recall, F1-score.
  - ROC-AUC score and precision-recall curve analysis.
- **Visual Insights:**
  - Confusion matrices to understand predictions.
  - ROC and precision-recall curves for model performance visualization.

---
