# Iris-classification-using-machine-learning

### Introduction
In this repository, aiming to develop an ML Model for classifying iris flowers based on their features using Python, scikit-learn, and TensorFlow.

The aim is to classify iris flowers among three species(setosa, versicolor, or virginica) from measurements of sepals and petal's length and width.

The dataset contains a set of 150 records under 5 attributes - Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species).

The main goal here is to design an ml model that makes useful classification for new flowers or, in other words, one which exihibits good generalization.

This repository contains code and resources for classifying Iris flowers using machine learning techniques. 

The dataset used for training and testing the models is sourced from Kaggle's Iris dataset.

### Dataset
The Iris dataset contains various features of Iris flowers along with their respective species. The features include attributes such as sepal length, sepal width, petal length, and petal width. This dataset is widely used for practicing classification algorithms in machine learning.

The Iris flower dataset is included in the project directory. The dataset contains 150 samples of iris flowers, with 50 samples for each of three iris species. Four features are included for each sample: sepal length, sepal width, petal length, and petal width.

### Requirements
To run the code in this repository, you'll need the following libraries:
Python 3.x,
Pandas,
NumPy,
Matplotlib,
Seaborn,
Scikit-learn.

### Approach
In the Jupyter notebook, we explore the dataset, perform data preprocessing, and then train several machine learning models such as Logistic Regression, Support Vector Machines and Decision Tree Classifier. We evaluate the performance of each model and select the best-performing one for classifying Iris flowers into their respective species.
We found out that Logistic Regression gave the best accuracy and good prediction among other three models.
Later we saved the best model.

### Result
After training and testing our Iris classification model using various machine learning algorithms, we evaluated its performance using standard classification metrics for the Linear Regression Model.

Accuracy: 0.93,
Precision: 0.95,
Recall: 0.93,
F1 Score: 0.93.

Additionally, we visualized the model's performance using a confusion matrix:
From the confusion matrix, we can see how well the model performed in classifying each species of Iris flower. 

We continue to refine our model and explore further optimizations to enhance its performance.

### Conclusion
In conclusion, this project has demonstrated the effectiveness of machine learning algorithms in accurately classifying Iris flowers based on their features. Through meticulous data preprocessing, model training, and evaluation, we have achieved commendable results in distinguishing between the three species of Iris flowers: Setosa, Versicolor, and Virginica.


