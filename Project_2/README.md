# Machine Learning Regression Evaluation using Scikit-learn

## Overview

This project evaluates the performance of three machine learning regression models—K-Nearest Neighbors (KNN), Decision Tree Regressor, and Linear Regression—on the **social mobility (socmob)** dataset. The evaluation focuses on comparing the models using learning curves and Root Mean Squared Error (RMSE) as the evaluation metric. Hyperparameter tuning is performed for KNN and Decision Tree to optimize their performance.

## Repository Contents

- **main.py**: The main Python script that loads the dataset, preprocesses the data, performs hyperparameter tuning, generates learning curves, and evaluates the models. The script is self-contained and uses Scikit-learn for the entire pipeline.
- **Report.pdf**: A comprehensive report that includes a description of the dataset, methodology, results (graphs and tables), and discussions on the findings. This document provides deeper insights into the experiments and results.

## Dataset

- **Dataset Name**: `socmob` (ID: 44987) from OpenML  
- **Description**: The dataset examines the relationship between fathers' occupations and their sons' current and first occupations, along with attributes like race and family structure.
- **Target Variable**: `counts_for_sons_current_occupation` (numeric)  
- **Features**:  
  - Numerical: `counts_for_sons_current_occupation`, `counts_for_sons_first_occupation`  
  - Categorical: `fathers_occupation`, `sons_occupation`, `family_structure`, `race`  
- **Size**: 6 features, over 1000 examples

## Prerequisites

- Python 3.8 or later
- pandas
- numpy
- scikit-learn
- matplotlib

## Installation

Install the required dependencies using pip:

```bash
pip install numpy pandas scikit-learn matplotlib
```

## Usage
To run the evaluation script, execute the following command:

```bash
python assignment2.py
```

The script performs the following operations:

- Loads the dataset from OpenML.
- Preprocesses the data:
  - One-hot encodes categorical features.
  - Standardizes numerical features.
- Plots learning curves for:
  - KNN models with three different values of k (3, 5, 7).
  - Best-tuned KNN, Decision Tree, and Linear Regression models.
- Displays results in graphical and tabular formats, including RMSE for training and test sets.

## Discussion
The analysis highlights the importance of hyperparameter tuning and model selection for regression tasks. KNN demonstrated the best balance between bias and variance, while Decision Tree required further tuning to mitigate overfitting. Linear Regression's inability to capture non-linear patterns underscores the need for more advanced feature engineering or alternative models for such datasets.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

