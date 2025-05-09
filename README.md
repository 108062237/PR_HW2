# Pattern Recognition - Assignment #2

## Description

Experiments on FLD/LDA (Task 1) and PCA (Task 2) for dimensionality reduction using Wine, Iris, Breast Cancer, and Ionosphere datasets. Evaluates impact on Naive Bayes and Logistic Regression classifiers.

## How to Run

* **Task 1 (FLD/LDA):** `python src/task1.py --config configs/<dataset_name>.yaml`
    * Example: `python src/task1.py --config configs/iris.yaml`
* **Task 2 (PCA):** `python src/task2.py --config configs/<dataset_name>.yaml`
    * Example: `python src/task2.py --config configs/breast_cancer.yaml`

## Key Files & Structure

* `configs/`: Contains dataset YAML configuration files.
* `src/data_loader.py`: Loads data.
* `src/naive_bayes.py`: Naive Bayes classifier from HW1.
* `src/logistic_regression.py`: Logistic Regression classifier from hw1.
* `src/task1.py`: FLD/LDA experiments.
* `src/task2.py`: PCA experiments.

