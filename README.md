# Comparing-Optimizers-for-Multiclass-Classification-with-ANN
The notebook compares the performance of three optimizers  (SGD), Adam, and RMSprop—in terms of training speed and accuracy. The dataset is preprocessed with standard scaling and one-hot encoding, and the ANN architecture consists of three hidden layers with ReLU activation and a softmax output layer for the seven-class classification task.
# Comparing Optimizers for Multiclass Classification with ANN on the Covertype Dataset

## Overview
This Jupyter notebook demonstrates the implementation of a feedforward Artificial Neural Network (ANN) for multiclass classification on the [Covertype dataset](https://archive.ics.uci.edu/ml/datasets/covertype). The primary objective is to compare the performance of three popular optimizers—Stochastic Gradient Descent (SGD), Adam, and RMSprop—in terms of training speed and accuracy. The notebook includes data preprocessing, model building, training with early stopping, and visualization of training and validation metrics.

## Dataset
The Covertype dataset contains 581,012 samples with 54 features, representing cartographic variables to predict one of seven forest cover types. The dataset is publicly available and fetched using `sklearn.datasets.fetch_covtype`.

## Requirements
To run this notebook, ensure you have the following Python libraries installed:
- `numpy`
- `scikit-learn`
- `tensorflow`
- `matplotlib`

Notebook Structure
Data Preprocessing:
Loads the Covertype dataset.
Adjusts target labels to 0–6 for compatibility with softmax.
Splits data into training (80%) and testing (20%) sets.
Applies standard scaling to features using StandardScaler.
Converts target labels to one-hot encoding for multiclass classification.
Model Architecture:
Defines a feedforward ANN with:
Input layer matching the feature count.
Three hidden layers (128, 64, 32 units) with ReLU activation.
Output layer with 7 units and softmax activation for multiclass classification.
Optimizer Comparison:
Trains the model with three optimizers: SGD, Adam, and RMSprop.
Uses categorical crossentropy loss and accuracy as the evaluation metric.
Implements early stopping with a patience of 3 epochs to prevent overfitting.
Trains for up to 20 epochs with a batch size of 1024.
Visualization:
Plots training and validation accuracy and loss for each optimizer to compare performance.
Results:
Adam: Converges fastest and achieves the highest validation accuracy (~87.98% at epoch 19).
SGD: Slower convergence, lower accuracy (~75.78% at epoch 20), likely due to lack of momentum.
RMSprop: Good performance with noisy gradients, reaching ~84.55% validation accuracy.
Key Findings
Adam is the most effective optimizer for this task, offering faster convergence and higher accuracy.
SGD underperforms without momentum or careful learning rate tuning.
RMSprop handles noisy gradients well but is slightly outperformed by Adam.
The best optimizer choice depends on the problem, learning rate, and dataset characteristics.
How to Run
Clone or download this repository.
Ensure all required libraries are installed (see Requirements).
Open the notebook (2025_07_10.ipynb) in Jupyter Notebook or JupyterLab.
Run the cells sequentially to preprocess the data, train the models, and visualize results.
Visualizations
The notebook includes plots comparing:

Training and validation accuracy across optimizers.
Training and validation loss across optimizers.
These plots help visualize the convergence speed and performance differences between SGD, Adam, and RMSprop.

Future Improvements
Experiment with learning rate tuning for SGD (e.g., adding momentum or scheduling).
Test additional optimizers like Adagrad or Nadam.
Explore hyperparameter tuning (e.g., batch size, number of layers, or units).
Incorporate cross-validation for more robust performance evaluation.
