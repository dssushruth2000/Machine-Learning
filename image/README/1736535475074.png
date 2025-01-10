# Decision Tree Classifier Evaluation using Scikit-learn

## Overview
This project evaluates the performance of decision tree classifiers using the Gini and Entropy criteria on two distinct datasets. The datasets used in this project are from the OpenML repository, specifically for binary classification tasks with all numeric features and each containing over 1000 examples.

## Repository Contents
- **main.py**: This is the main Python script that loads the datasets, performs grid search for hyperparameter tuning, computes the ROC curves, and plots the results. It uses Scikit-learn for model training and evaluation.

- **Report.pdf**: A detailed report that includes a description of the datasets, methodology, results in tabular and graphical forms, and a discussion on the findings. This document is critical for understanding the context and conclusions of the computational experiments.

## Datasets
- **Dataset 1130 (OVA_Lung)**: This dataset is used for classifying tissue types based on gene expression profiles.

- **Dataset 1134 (OVA_Kidney)**: Similar to the previous dataset, it classifies tissue types using gene expression data.

Both datasets contain 10936 numeric features and a binary nominal target representing tissue types.

## Prerequisites
- Python 3
- pandas
- numpy
- scikit-learn
- matplotlib

## Installation
To run this project, you will need to install the required libraries. You can install these using pip:
```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage
To run the evaluation script, execute the following command:

```bash
python assignment1.py
```

The script performs the following operations:

- Loads the data from OpenML.
- Prepares the data for modeling.
- Conducts grid search with 10-fold cross-validation to find the best min_samples_leaf parameter.
- Computes ROC curves and AUC values for both datasets using Gini and Entropy criteria.

## Discussion
The analysis concludes that decision tree classifiers exhibit strong performance on both datasets. Entropy tends to provide slightly better results than Gini, suggesting it may be more suitable for these specific classification tasks. For further details, please refer to the project report.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

