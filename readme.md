# 💳 Credit Card Fraud Detection Using Machine Learning

## 📋 Overview

This project is a web-based dashboard for detecting credit card fraud using machine learning techniques. The dashboard allows users to upload transaction data, analyze patterns, and make predictions about the likelihood of fraud in individual transactions.

**Live Demo:** [https://creditcard-fraud-detection-1.onrender.com](https://creditcard-fraud-detection-1.onrender.com)

## 📑 Table of Contents
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Dataset Overview](#dataset-overview)
- [Implementation Steps](#implementation-steps)
- [Algorithms Used](#algorithms-used)
- [Sampling Techniques](#sampling-techniques)
- [Results](#results)
- [Installation and Setup](#installation-and-setup)
  - [Local Machine Setup](#local-machine-setup)
- [Requirements](#requirements)
- [Demo Images](#demo-images)


## ✨ Key Features

* **Upload Dataset**: Users can upload a CSV file containing credit card transaction data.
* **Make Predictions**: Enter transaction details to check for fraud risk.
* **Dashboard**: Visualize transaction patterns, train machine learning models, and compare model performance.
* **Advanced Analytics**: Analyze behavioral patterns, detect anomalies, and assess risk scores.
  
## Technologies Used

- **Streamlit**: For building the web application
- **Pandas**: For data manipulation and analysis
- **NumPy**: For numerical operations
- **Matplotlib & Seaborn**: For data visualization
- **Plotly**: For interactive visualizations
- **Scikit-learn**: For machine learning algorithms and metrics
- **XGBoost**: For advanced boosting algorithms
- **imbalanced-learn**: For handling imbalanced datasets
  
## Dataset Overview

- **Transactions**: 284,807
- **Frauds**: 492
- **Fraud Percentage**: 0.172%

## Implementation Steps

1. **Reading, Understanding, and Visualizing the Data**
   - Loading the dataset
   - Exploratory data analysis
   - Correlation analysis
   - Distribution of normal vs fraudulent transactions

2. **Preparing the Data for Modeling**
   - Handling imbalanced data using various sampling techniques
   - Feature scaling
   - Train-test split

3. **Building the Model**
   - Training multiple machine learning algorithms
   - Hyperparameter tuning
   - Cross-validation

4. **Evaluating the Model**
   - Performance metrics (Accuracy, Precision, Recall, F1-Score)
   - ROC Curve and AUC
   - Confusion Matrix



## Algorithms Used

We have utilized a total of **5 algorithms** in our project:

1. Logistic Regression
2. K-Nearest Neighbours
3. Decision Tree
4. Random Forest
5. XgBoost

## Sampling Techniques

We employed **six techniques** for undersampling or oversampling:

1. Random Oversampling
2. SMOTE Oversampling
3. Random Undersampling
4. Tomek Links Undersampling
5. Cluster Centroids Undersampling
6. SMOTE + Tomek Links

## Results

- We compared each of the 5 algorithms across all 7 scenarios (Normal data + Six undersampled or oversampled data) by evaluating their accuracy through the Area Under the Curve (AUC) of the ROC Curve.
- **XgBoost** emerged as the best algorithm in many simulations.
- The performance of classifiers can be significantly improved when sampling methods are used to rebalance the two classes.


## Installation and Setup

### Prerequisites
- Python 3.6+
- Anaconda (recommended) or pip

### Local Machine Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Download the dataset and extract it into the project directory:
   [Download Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)

3. Install Anaconda IDE if not already installed.

4. Navigate to the project directory and launch Jupyter Notebook:
   ```
   jupyter notebook
   ```

5. Open "Credit Card Fraud Detection Using Machine Learning - Local Version.ipynb"

6. Execute all cells in the notebook.


## Demo Images

![Dashboard](/images/dashboard.png)
*Main dashboard showing fraud detection metrics and transaction statistics*

![ROC Curves](/images/roc_curves.png)
*Comparison of ROC curves for different algorithms*

![Confusion Matrix](/images/confusion_matrix.png)
*Confusion matrix showing true/false positives and negatives*

![Class Distribution](/images/class_distribution.png)
*Visualization of the class imbalance between fraudulent and normal transactions*


## Requirements

- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- imbalanced-learn
- xgboost
- streamlit
- plotly

Install all requirements using:
```
pip install -r requirements.txt
```


