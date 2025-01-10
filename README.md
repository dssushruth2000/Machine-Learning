# Machine Learning and Deep Learning Projects

## Overview

This repository contains four distinct projects that explore machine learning and deep learning techniques applied to various datasets. Each project focuses on specific tasks, models, and methodologies, providing a comprehensive learning experience in data science and AI.

## Projects

### 1. **Decision Tree Classifier Evaluation using Scikit-learn**
- **Description**: Evaluates decision tree classifiers using Gini and Entropy criteria on two datasets for binary classification.
- **Key Features**:
  - Grid search for hyperparameter tuning.
  - ROC curve generation and AUC evaluation.
- **Datasets**:
  - OVA_Lung: Gene expression profiles for tissue classification.
  - OVA_Kidney: Gene expression profiles for tissue classification.
- **Tools**: Scikit-learn, Matplotlib
- **Key Results**: Entropy-based models slightly outperform Gini-based ones.
- [Detailed README](./Project_1/README.md)

---

### 2. **Machine Learning Regression Evaluation using Scikit-learn**
- **Description**: Compares KNN, Decision Tree Regressor, and Linear Regression on the socmob dataset for regression tasks.
- **Key Features**:
  - Learning curves for performance visualization.
  - Hyperparameter tuning for KNN and Decision Tree.
- **Dataset**: Socmob dataset (social mobility prediction).
- **Tools**: Scikit-learn, Matplotlib
- **Key Results**: KNN provides the best balance between performance and generalization.
- [Detailed README](./Project_2/README.md)

---

### 3. **Neural Network Models for Regression and Classification**
- **Description**: Builds and evaluates neural network models for regression and classification tasks using the Keras library.
- **Key Features**:
  - Regression: Appliance energy prediction.
  - Classification: Banknote authentication.
- **Datasets**:
  - Energy Usage: Predicting energy consumption.
  - Banknote Authentication: Identifying genuine vs. forged notes.
- **Tools**: TensorFlow (Keras), Scikit-learn, Matplotlib
- **Key Results**: Simpler architectures perform comparably to complex models for classification tasks.
- [Detailed README](./Project_3/README.md)

---

### 4. **Monkey Species Classification using CNNs**
- **Description**: Explores image classification using custom and pre-trained CNN architectures for the Monkey Species dataset.
- **Key Features**:
  - Custom CNN architectures.
  - Fine-tuning a pre-trained EfficientNetV2S model.
  - Error analysis with misclassified images.
- **Dataset**: Monkey Species Dataset (10 classes).
- **Tools**: TensorFlow (Keras), Matplotlib, Scikit-learn
- **Key Results**: Pre-trained models significantly outperform custom architectures.
- [Detailed README](./Project_4/README.md)

## Prerequisites

- Python 3.8 or later
- Libraries:
  - Scikit-learn
  - TensorFlow
  - Pandas
  - NumPy
  - Matplotlib

## Usage

Each project folder contains its own `README.md` file with detailed instructions for running the code and reproducing the results. Run the main script in the respective folder to execute the project:

```bash
python main.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue to discuss your ideas.

---
This repository is a showcase of machine learning and deep learning techniques applied across a variety of datasets and problem domains. Dive into the individual projects to explore more!