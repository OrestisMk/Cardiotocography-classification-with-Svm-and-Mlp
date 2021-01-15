# Comparison-Svm-and-Mlp
This project compares the classification accuracy of SVM and Mlp on cardiotocography dataset. For the purpose of this project ,we added suspicious and pathologic classes and created a new variable as a target value. As a result, the classification problem became binary with 2 classes (normal, suspicious/pathologic). Although, we
combined these two, the classes remain highly imbalance. So, In order to tackle this problem SMOTE imbalance method is used. Also, we tried to make the model more robust by adding noise to the dataset.

The dataset was downloaded from the UCI: https://archive.ics.uci.edu/ml/datasets/cardiotocography
