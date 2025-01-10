# Neural Network Models for Regression and Classification

## Overview

This project evaluates the performance of neural network models for regression and classification tasks using the Keras library. The project involves building and training four different neural network architectures for each task, analyzing their performances, and discussing the results.

## Repository Contents

- **main.py**: The main Python script that handles data loading, preprocessing, model building, training, evaluation, and visualization. The script is self-contained, and running it will reproduce all results.
- **Report.pdf**: A comprehensive report containing a detailed description of the datasets, model architectures, graphs, tables of results, and discussions.

## Datasets

### Regression Dataset
- **Dataset Name**: Energy Usage (OpenML ID: 550)
- **Description**: Predicts appliance energy consumption based on environmental features such as temperature and humidity. The dataset includes 2178 instances and 4 features.
- **Target Variable**: Appliance energy consumption (continuous variable).

### Classification Dataset
- **Dataset Name**: Banknote Authentication (OpenML ID: 1462)
- **Description**: A binary classification task to identify genuine or forged banknotes based on statistical features extracted from images.
- **Target Variable**: Binary labels (1 for genuine, 2 for forged).
- **Size**: 1372 instances and 4 features.

## Prerequisites

- Python 3.8 or later
- Libraries:
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `tensorflow` (Keras)

## Installation

Install the required dependencies using pip:

```bash
pip install numpy pandas scikit-learn matplotlib tensorflow
```

## Usage

Run the main script to execute all tasks and generate outputs:

```bash
python main.py
```

The script performs the following operations:

- Loads datasets from OpenML.
- Preprocesses the data:
    - One-hot encodes categorical features.
    - Standardizes numerical features.
- Builds and trains four neural network models for:
  - Regression: Optimized for minimum validation mean squared error (MSE).
  - Classification: Optimized for maximum validation accuracy.
- Plots learning curves:
  - Training and validation errors for regression models.
  - Training and validation accuracy for classification models.
- Summarizes results in tabular format.

## Discussion
The experiments highlight the following key points:

- For regression, increasing model complexity (e.g., layers, neurons, dropout) slightly improved performance but did not significantly outperform simpler models.
- For classification, the dataset's characteristics allowed even simple models to achieve near-perfect accuracy.
- Dropout layers in regression models helped prevent overfitting, while consistent activation functions provided stability in classification models. 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

