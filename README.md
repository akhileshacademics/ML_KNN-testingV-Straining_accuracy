Iris Model Evaluation — KNN & Intro to Logistic Regression
Overview

This notebook contains my practical work building and evaluating a machine learning model using the Iris dataset.
I implemented K-Nearest Neighbors (KNN) and experimented with evaluating model performance by splitting the dataset into training and testing subsets.
I also briefly explored Logistic Regression, mostly at a surface level — not the mathematical derivation yet.

What I Learned
📌 Model Training & Dataset Splitting

Training a model using the entire dataset and then testing on the same data gives misleading accuracy, because the model is memorizing instead of learning patterns.

Splitting the data into 80% training and 20–30% testing provides more realistic evaluation and avoids performance illusion.

📌 Choosing the Best K Value in KNN

Testing multiple values of k helps identify the most accurate version of the KNN model.

Very small k values may lead to noisy predictions.

Very large k values may oversimplify the decision boundaries.

📌 Train Accuracy vs Test Accuracy

A model with high training accuracy may still perform poorly on unseen test data.

A large gap between training and testing accuracy usually indicates overfitting.

📌 Overfitting

Making the model unnecessarily complex causes it to perform extremely well on training data but fail on real unseen data.

Overfitting restricts generalization ability, which is the actual goal of machine learning.

Progress and Next Steps
Completed so far:

KNN basics & evaluation

Understanding dataset splitting

Observing variance in model performance

Intro-level exposure to Logistic Regression

Planned next steps:

Understanding the mathematics behind Logistic Regression

Experimenting with cross-validation

Learning regularization techniques to combat overfitting

Tools & Libraries Used

Python

scikit-learn (KNeighborsClassifier, train_test_split)

Jupyter Notebook

Repository Purpose

This repository exists only for the evaluation notebook:

iris_model_eval.ipynb
