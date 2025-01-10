# Monkey Species Classification using CNNs

## Overview

This project explores image classification for the Monkey Species dataset using Convolutional Neural Networks (CNNs). The assignment involves designing and training custom CNN architectures, fine-tuning a pre-trained model, and performing error analysis. The results highlight the effectiveness of different CNN strategies for this classification task.

## Repository Contents

- **main.py**: The main Python script that implements data loading, preprocessing, model training, evaluation, and error analysis. It is self-contained and can reproduce all results when run.
- **removecorrupt.py**: A utility script to clean the dataset by removing corrupt image files.
- **best_model1.keras**: The saved model file of the best-performing custom CNN architecture from Task 1.
- **Monkey Species Data**: Dataset used for this project.
- **Report.pdf**: A comprehensive report detailing the tasks, model architectures, results, and analysis.

## Dataset

### Dataset Name: **Monkey Species Dataset**

### Description
A multi-class classification dataset containing images of 10 monkey species. The dataset is divided into:
- **Training Data**: Used to train the CNN models.
- **Prediction Data**: Used to evaluate model performance.

### Classes
Each image belongs to one of 10 monkey species, identified by their respective labels.

## Prerequisites

- Python 3.8 or later
- Libraries:
  - `tensorflow`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`

## Installation

Install the required dependencies using pip:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

## Usage

Run the main script to execute all tasks and generate outputs:

```bash
python main.py
```

The script performs the following:

- Loads and Preprocesses the Data: Ensures the dataset is ready for model training and evaluation.
- Custom CNN Architectures: Builds and trains two distinct CNN models.
- Fine-Tuning a Pre-Trained Model: Adapts an EfficientNetV2S model for the Monkey Species dataset.
- Model Evaluation: Compares accuracy and confusion matrices of all models.
- Error Analysis: Analyzes 10 misclassified images to identify potential reasons for errors.

## Discussion
This project illustrates the advantages of pre-trained architectures for complex image classification tasks, particularly in handling small datasets with fine-grained distinctions. Custom architectures provided valuable insights into the model design process but were limited by their complexity and capacity.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

