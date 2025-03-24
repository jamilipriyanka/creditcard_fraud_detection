# 💳 Credit Card Fraud Detection Using Machine Learning

## 📋 Overview

A machine learning system that detects fraudulent credit card transactions using a European cardholder dataset. The project addresses the challenge of extreme class imbalance (0.172% fraud cases) through various sampling techniques and compares five machine learning algorithms to find the optimal approach for fraud detection.

**Live Demo:** [https://creditcard-fraud-detection-1.onrender.com](https://creditcard-fraud-detection-1.onrender.com)

## 📑 Table of Contents
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Dataset Overview](#dataset-overview)
- [Implementation Steps](#implementation-steps)
- [Demo Images](#demo-images)
- [Algorithms Used](#algorithms-used)
- [Sampling Techniques](#sampling-techniques)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Installation and Setup](#installation-and-setup)
  - [Local Machine Setup](#local-machine-setup)
  - [Google Colab Setup](#google-colab-setup)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Future Improvements](#future-improvements)
- [License](#license)

## Problem Statement

The aim of this project is to predict fraudulent credit card transactions using machine learning models. This is crucial from both the bank's and customer's perspectives. Banks cannot afford to lose their customers' money to fraudsters, as every fraud is a loss to the bank.

The dataset contains transactions made over a period of two days in September 2013 by European credit cardholders. The dataset is highly unbalanced, with the positive class (frauds) accounting for only 0.172% of all transactions. We need to address this data imbalance while building the model and identify the best model by trying various algorithms.

## ✨ Key Features

* **Upload Dataset**: Users can upload a CSV file containing credit card transaction data.
* **Make Predictions**: Enter transaction details to check for fraud risk.
* **Dashboard**: Visualize transaction patterns, train machine learning models, and compare model performance.
* **Advanced Analytics**: Analyze behavioral patterns, detect anomalies, and assess risk scores.
* **Technologies**:
  * **Streamlit**: Interactive web application for user-friendly interface.
  * **Pandas & NumPy**: Robust data processing and numerical operations.
  * **Matplotlib, Seaborn & Plotly**: Dynamic and interactive data visualizations.
  * **Scikit-learn**: Powerful machine learning algorithms and evaluation metrics.
  * **XGBoost**: Advanced gradient boosting for superior model performance.

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

## Demo Images

![Dashboard](/images/dashboard.png)
*Main dashboard showing fraud detection metrics and transaction statistics*

![ROC Curves](/images/roc_curves.png)
*Comparison of ROC curves for different algorithms*

![Confusion Matrix](/images/confusion_matrix.png)
*Confusion matrix showing true/false positives and negatives*

![Class Distribution](/images/class_distribution.png)
*Visualization of the class imbalance between fraudulent and normal transactions*

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

## Technologies Used

- **Streamlit**: For building the web application
- **Pandas**: For data manipulation and analysis
- **NumPy**: For numerical operations
- **Matplotlib & Seaborn**: For data visualization
- **Plotly**: For interactive visualizations
- **Scikit-learn**: For machine learning algorithms and metrics
- **XGBoost**: For advanced boosting algorithms
- **imbalanced-learn**: For handling imbalanced datasets

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

### Google Colab Setup

1. Visit [Google Colab](https://colab.research.google.com/).

2. Log in with your Gmail account.

3. Upload "Credit Card Fraud Detection Using Machine Learning - Colab version.ipynb".

4. Run all cells in the notebook.

## Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv          # Dataset file
│
├── notebooks/
│   ├── local_version.ipynb     # Jupyter notebook for local execution
│   └── colab_version.ipynb     # Notebook optimized for Google Colab
│
├── images/
│   ├── dashboard.png           # Dashboard screenshot
│   ├── roc_curves.png          # ROC curves comparison
│   ├── confusion_matrix.png    # Confusion matrix visualization
│   └── class_distribution.png  # Class imbalance visualization
│
├── results/
│   ├── figures/                # Generated plots and visualizations
│   └── models/                 # Saved model files
│
├── src/
│   ├── data_preprocessing.py   # Data preparation scripts
│   ├── model_training.py       # Model training scripts
│   └── evaluation.py           # Performance evaluation scripts
│
├── requirements.txt            # Required packages
└── README.md                   # Project documentation
```

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

## Future Improvements

- Implement deep learning models (Neural Networks)
- Feature engineering to create more discriminative features
- Ensemble methods combining multiple models
- Real-time fraud detection implementation

## License

This project is licensed under the MIT License - see the LICENSE file for details.
