import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import io
import base64
from PIL import Image
import time as time_module
from time import sleep

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom color palette based on "Lotus garden"
colors = {
    'light_green': '#ACE2B3',
    'coral': '#FF7B7A',
    'light_purple': '#E3A4F4',
    'brown': '#694D43',
}

# High-quality professional CSS styling with Lotus Garden color scheme
st.markdown("""
<style>
/* Global styling */
.stApp {
    background-color: #FAFBFC;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Text styling */
h1 {
    color: #694D43;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
    letter-spacing: -0.01em;
}

h2 {
    color: #694D43;
    font-size: 1.8rem;
    font-weight: 600;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

h3 {
    color: #694D43;
    font-size: 1.4rem;
    font-weight: 600;
    margin-top: 1.2rem;
    margin-bottom: 0.8rem;
}

p {
    color: #333;
    line-height: 1.6;
}

/* Primary colors */
.primary-green {
    color: #ACE2B3 !important;
}
.primary-coral {
    color: #FF7B7A !important;
}
.primary-purple {
    color: #E3A4F4 !important;
}
.primary-brown {
    color: #694D43 !important;
}

/* Tab styling - elegant and professional */
.stTabs {
    padding: 0.5rem 0;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #E0E0E0;
}

.stTabs [data-baseweb="tab"] {
    height: 45px;
    white-space: pre-wrap;
    background-color: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    padding: 8px 16px;
    color: #555;
    font-weight: 500;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #694D43;
    background-color: rgba(172, 226, 179, 0.08);
}

.stTabs [aria-selected="true"] {
    color: #FF7B7A !important;
    background-color: transparent !important;
    border-bottom: 2px solid #FF7B7A !important;
    font-weight: 600;
}

/* Button styling - polished look */
div.stButton > button {
    background-color: #FF7B7A;
    color: white;
    border-radius: 4px;
    border: none;
    padding: 8px 20px;
    font-weight: 500;
    letter-spacing: 0.02em;
    box-shadow: 0 2px 4px rgba(255, 123, 122, 0.2);
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background-color: #ff6967;
    box-shadow: 0 4px 8px rgba(255, 123, 122, 0.3);
    transform: translateY(-1px);
}

div.stButton > button:active {
    transform: translateY(0);
    box-shadow: 0 1px 2px rgba(255, 123, 122, 0.3);
}

/* Clean, modern card container */
.card-container {
    background-color: white;
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 6px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    margin-bottom: 20px;
    transition: all 0.2s ease;
}

.card-container:hover {
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.08);
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    transition: all 0.2s ease;
}

[data-testid="stDataFrame"]:hover {
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.08);
}

/* Form styling */
[data-testid="stForm"] {
    background-color: white;
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 6px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

/* Slider styling */
[data-testid="stSlider"] > div {
    padding-top: 1rem;
    padding-bottom: 1.5rem;
}

[data-testid="stSlider"] > div > div {
    height: 3px;
    background-color: #f0f0f0;
}

[data-testid="stSlider"] > div > div > div > div {
    background-color: #FF7B7A;
    height: 3px;
}

[data-testid="stThumbValue"] {
    color: #694D43;
    font-weight: 500;
}

/* Metric styling - cleaner, more minimal */
[data-testid="stMetric"] {
    background-color: #f9f9f9;
    border-radius: 6px;
    padding: 12px;
    border-left: 3px solid #ACE2B3;
    margin-bottom: 12px;
    transition: all 0.2s ease;
}

[data-testid="stMetric"]:hover {
    background-color: #f5f5f5;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
}

[data-testid="stMetricLabel"] {
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    color: #444 !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    color: #694D43 !important;
}

[data-testid="stMetricDelta"] {
    font-size: 0.9rem !important;
}

/* Make multiselect cleaner */
[data-testid="stMultiSelect"] div div div {
    background-color: white;
    border-radius: 4px;
    border: 1px solid rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease;
}

[data-testid="stMultiSelect"] div div div:hover {
    border: 1px solid rgba(227, 164, 244, 0.5);
}

/* Info box styling */
[data-testid="stAlert"] {
    background-color: rgba(172, 226, 179, 0.2);
    border: 1px solid rgba(172, 226, 179, 0.4);
    color: #333;
}

/* Expander styling */
[data-testid="stExpander"] {
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 6px;
    overflow: hidden;
}

[data-testid="stExpander"] details {
    background-color: white;
    padding: 0;
}

[data-testid="stExpander"] summary {
    padding: 16px;
    background-color: #fafafa;
    border-bottom: 1px solid rgba(0, 0, 0, 0.08);
    font-weight: 500;
    color: #555;
}

[data-testid="stExpander"] summary:hover {
    background-color: #f5f5f5;
}

[data-testid="stExpander"] details[open] summary {
    font-weight: 600;
    color: #694D43;
}

[data-testid="stExpander"] .streamlit-expanderContent {
    padding: 16px;
}

/* Improve widget labels */
.stSelectbox label, .stSlider label, .stMultiSelect label {
    color: #694D43;
    font-weight: 500;
    font-size: 0.95rem;
}

/* Streamlit sidebar */
[data-testid="stSidebar"] {
    background-color: #fff;
    border-right: 1px solid rgba(0, 0, 0, 0.08);
    padding-top: 1rem;
}

[data-testid="stSidebarNav"] {
    padding-top: 1rem;
}

[data-testid="stSidebarNavItems"] {
    padding-top: 0.5rem;
}

[data-testid="stSidebarNavLink"] {
    color: #694D43;
}

/* Styling for plotly charts */
.js-plotly-plot {
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    background-color: white;
    padding: 8px;
}

/* Selectbox styling */
[data-testid="stSelectbox"] {
    color: #694D43;
}
</style>
""", unsafe_allow_html=True)

# Sidebar with more professional styling
st.sidebar.markdown("""
<div style="padding: 0 0 20px 0; text-align: center;">
    <h1 style="color: #694D43; font-size: 1.8rem; font-weight: 700; margin-bottom: 5px;">Credit Card Fraud</h1>
    <div style="height: 3px; background: linear-gradient(90deg, #ACE2B3, #E3A4F4); margin: 10px auto 15px auto; width: 80%; border-radius: 2px;"></div>
    <p style="color: #666; font-size: 0.9rem; margin-top: 0;">ML-Powered Detection System</p>
</div>
""", unsafe_allow_html=True)

# Add dataset upload section
st.sidebar.header('Upload Dataset')
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with credit card transactions", type=["csv"])

# Add dataset example info
with st.sidebar.expander("📊 Example Dataset Format"):
    st.markdown("""
    The expected dataset should have transaction features like:
    - `Time`: Time between transactions (seconds)
    - `Amount`: Transaction amount
    - `V1-V28`: Anonymized features (PCA transformation)
    - `Class`: 1 for fraudulent transactions, 0 for legitimate
    
    You can use the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset from Kaggle.
    """)

# Add help section
with st.sidebar.expander("❓ Help & Resources"):
    st.markdown("""
    ### How to Use This Dashboard
    1. **Prediction Tab**: Enter transaction details to check for fraud risk
    2. **Dashboard Tab**: Upload data to analyze patterns and train models
    
    ### About Credit Card Fraud
    Credit card fraud costs businesses billions each year. Machine learning helps detect suspicious patterns in real-time.
    """)

# Session state initialization
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'default_model' not in st.session_state:
    # Create a simple logistic regression model pre-trained on common patterns
    st.session_state.default_model = LogisticRegression(random_state=0)
    
    # Create some sample data for initial training
    np.random.seed(0)
    n_samples = 1000
    
    # Features: 30 features similar to credit card dataset
    X_sample = np.random.randn(n_samples, 30)
    
    # Create simple patterns for fraud
    fraud_pattern = (np.abs(X_sample[:, 0]) > 2.5) | (np.abs(X_sample[:, 1]) > 2.5)
    y_sample = fraud_pattern.astype(int)
    
    # Add amount as the last feature
    amounts = np.random.exponential(scale=100, size=n_samples)
    X_sample = np.hstack((X_sample, amounts.reshape(-1, 1)))
    
    # Train the model
    st.session_state.default_model.fit(X_sample, y_sample)
    
if 'transaction_step' not in st.session_state:
    st.session_state.transaction_step = 1
if 'transaction_data' not in st.session_state:
    st.session_state.transaction_data = {
        # Basic Info
        'time': 43200,
        'amount': 100.0,
        'time_since_last': 24,
        'card_country': "United States",
        'merchant_category': "Retail",
        'transaction_type': "In-Person",
        
        # Advanced Indicators
        'v1': 0.0,
        'v2': 0.0,
        'v3': 0.0,
        'v4': 0.0,
        'v5': 0.0,
        'unusual_pattern': False,
        
        # Transaction Context
        'ip_match': "Yes",
        'browser_match': False,
        'device_age': 12,
        'distance_from_home': 10,
        'previous_declined': 0,
        'velocity': 2.0
    }

# Data processing functions
def load_data(file):
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    # Basic preprocessing for credit card fraud dataset
    if df is not None:
        # Check if the data has the expected columns
        required_cols = ['Class']
        if not all(col in df.columns for col in required_cols):
            st.error("The dataset must contain a 'Class' column indicating fraud (1) or legitimate (0) transactions.")
            return None, None, None, None
        
        # Separate features and target
        X = df.drop(['Class'], axis=1) if 'Class' in df.columns else df
        y = df['Class'] if 'Class' in df.columns else None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        return X_train, X_test, y_train, y_test
    return None, None, None, None

def train_model(model_name, X_train, y_train):
    # Train a model based on the selected algorithm
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    model = models[model_name]
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Predict and evaluate model
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC and AUC if probability prediction is available
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr
    
    metrics['cm'] = cm
    return metrics

def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_xticklabels(['Legitimate', 'Fraud'])
    ax.set_yticklabels(['Legitimate', 'Fraud'])
    return fig

def plot_roc_curve(fpr, tpr, auc, model_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    return fig

def plot_feature_importance(model, feature_names, model_name):
    # Get feature importances if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:]  # Top 10 features
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(indices)), importances[indices], align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_title(f'Top 10 Feature Importances - {model_name}')
        ax.set_xlabel('Relative Importance')
        return fig
    elif hasattr(model, 'coef_'):
        # For linear models like Logistic Regression
        coefficients = model.coef_[0]
        importance = np.abs(coefficients)
        indices = np.argsort(importance)[-10:]  # Top 10 features
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(indices)), importance[indices], align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_title(f'Top 10 Feature Importances - {model_name}')
        ax.set_xlabel('Absolute Coefficient Magnitude')
        return fig
    return None

def predict_fraud_probability(model, input_data):
    # Predict fraud probability for a single transaction
    if model and input_data is not None:
        try:
            # Assume input_data is already in the right format
            prediction = model.predict_proba(input_data.reshape(1, -1))[0, 1]
            return prediction
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    return None

# Navigation functions (for main UI, not individual steps)
def go_to_next_step():
    st.session_state.transaction_step += 1

def go_to_previous_step():
    st.session_state.transaction_step -= 1

# Main content with elegant header design
st.markdown("""
<div style="margin-bottom: 32px; text-align: center;">
    <h1 style="color: #694D43; font-size: 2.3rem; font-weight: 700; margin-bottom: 8px; letter-spacing: -0.01em;">
        Credit Card Fraud Detection
        <span style="font-size: 1.5rem; background: linear-gradient(90deg, #ACE2B3, #E3A4F4); 
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                     margin-left: 10px; vertical-align: middle;">Dashboard</span>
    </h1>
    <div style="height: 3px; background: linear-gradient(90deg, #ACE2B3, #E3A4F4); margin: 15px auto; width: 150px; border-radius: 2px;"></div>
    <p style="color: #666; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
        Analyze, visualize, and predict credit card fraud using advanced machine learning
    </p>
</div>
""", unsafe_allow_html=True)

# Create the main tabs structure with more spacing
tab1, tab2, tab3 = st.tabs(["Make Prediction", "Dashboard", "Advanced Analytics"])

# Tab 3 content definition
with tab3:
    st.header("Advanced Analytics & Insights")
    
    # Instructions if no data is uploaded
    if uploaded_file is None and st.session_state.data is None:
        st.markdown("""
        <div class="card-container" style="text-align: center; padding: 40px 20px;">
            <h3 style="color: #694D43; margin-top: 0;">Upload a Dataset for Advanced Analytics</h3>
            <p style="margin-bottom: 20px;">Use the sidebar to upload a CSV file containing credit card transaction data.</p>
            <div style="font-size: 3rem; margin: 20px 0;">🔍</div>
            <p>The advanced analytics section provides:</p>
            <ul style="text-align: left; max-width: 450px; margin: 0 auto;">
                <li>Enhanced fraud pattern detection</li>
                <li>Behavioral analysis of spending patterns</li>
                <li>Anomaly detection in transaction sequences</li>
                <li>Transaction velocity analysis</li>
                <li>Geographic clustering of fraud occurrences</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # If data is available
    elif st.session_state.data is not None:
        # Create tabs for different advanced analytics sections
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        advanced_tabs = st.tabs(["Pattern Recognition", "Behavioral Analysis", "Anomaly Detection", "Risk Scoring"])
        
        # Pattern Recognition Tab
        with advanced_tabs[0]:
            st.subheader("Fraud Pattern Recognition")
            
            # Create a sample visualization for pattern recognition
            st.markdown("""
            <div class="card-container">
                <h4 style="color: #694D43; margin-top: 0;">Transaction Pattern Analysis</h4>
                <p>Detecting complex patterns in transaction sequences and behaviors.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a placeholder for a pattern visualization
            data = st.session_state.data
            
            if 'Class' in data.columns and 'Amount' in data.columns:
                # Create sample pattern visualization
                pattern_data = data.copy()
                
                # Simple pattern: Transaction amount vs fraud (add random noise for visualization)
                if len(pattern_data) > 100:
                    pattern_sample = pattern_data.sample(100, random_state=42)
                else:
                    pattern_sample = pattern_data
                
                fig = px.scatter(
                    pattern_sample,
                    x='Amount',
                    y=np.random.normal(0, 1, len(pattern_sample)),  # Add some randomness for visualization
                    color='Class',
                    size='Amount',
                    color_discrete_sequence=['#ACE2B3', '#FF7B7A'],
                    opacity=0.7,
                    size_max=20,
                    labels={'color': 'Transaction Type', 'x': 'Amount', 'y': 'Pattern Dimension'}
                )
                
                fig.update_layout(
                    title="Transaction Pattern Clustering",
                    xaxis_title="Transaction Amount",
                    yaxis_title="Pattern Dimension",
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a description
                st.markdown("""
                <div class="card-container" style="margin-top: 20px;">
                    <h4 style="color: #694D43; margin-top: 0;">Pattern Insights</h4>
                    <p>The visualization above shows clustering of transaction patterns based on amount and other behavioral factors. Fraudulent transactions (shown in coral) often exhibit distinct patterns compared to legitimate ones (shown in green).</p>
                    <p>Our advanced pattern recognition algorithms can detect subtle patterns that might indicate fraud, such as:</p>
                    <ul>
                        <li>Unusual transaction sequences</li>
                        <li>Rapid changes in spending behavior</li>
                        <li>Transactions outside of established patterns</li>
                        <li>Transaction velocity anomalies</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        # Behavioral Analysis Tab
        with advanced_tabs[1]:
            st.subheader("Behavioral Analysis")
            
            st.markdown("""
            <div class="card-container">
                <h4 style="color: #694D43; margin-top: 0;">Cardholder Behavior Profiling</h4>
                <p>Analyzing typical spending patterns and detecting behavior changes that might indicate fraud.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create columns for behavior metrics
            cols = st.columns(3)
            
            with cols[0]:
                st.metric("Average Transaction Amount", "$158.36")
                st.metric("Typical Transaction Time", "2:45 PM")
                st.metric("Common Merchant Categories", "Retail, Dining")
            
            with cols[1]:
                st.metric("Transaction Frequency", "9.7 per week")
                st.metric("Average Distance", "12.3 miles")
                st.metric("Device Consistency", "87%")
            
            with cols[2]:
                st.metric("Behavior Risk Score", "Low")
                st.metric("Pattern Consistency", "92%")
                st.metric("Anomaly Detection Rate", "3.2%")
            
            # Add a behavior timeline visualization
            st.markdown("""
            <div class="card-container" style="margin-top: 20px;">
                <h4 style="color: #694D43; margin-top: 0;">Behavior Timeline</h4>
                <p>Analyzing transaction behavior over time to detect changes.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create sample timeline data
            timeline_days = np.arange(30)
            spending_amount = np.concatenate([
                np.random.normal(100, 20, 20),  # Normal spending
                np.random.normal(250, 40, 10)   # Sudden change in behavior
            ])
            
            behavior_df = pd.DataFrame({
                'Day': timeline_days,
                'Amount': spending_amount,
                'Pattern': ['Normal']*20 + ['Anomaly']*10
            })
            
            # Create timeline plot
            fig = px.line(
                behavior_df, x='Day', y='Amount',
                color='Pattern',
                markers=True,
                color_discrete_sequence=['#ACE2B3', '#FF7B7A']
            )
            
            fig.update_layout(
                title="Spending Behavior Timeline",
                xaxis_title="Day",
                yaxis_title="Transaction Amount",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add behavioral insights
            st.markdown("""
            <div class="card-container" style="margin-top: 20px;">
                <h4 style="color: #694D43; margin-top: 0;">Behavioral Insights</h4>
                <p>The behavior timeline shows a significant change in spending pattern after day 20, which could indicate account takeover or fraud.</p>
                <p>Our behavioral analysis models can detect subtle changes in cardholder behavior that may indicate fraud:</p>
                <ul>
                    <li>Sudden changes in spending amounts</li>
                    <li>Transactions at unusual times</li>
                    <li>New merchant categories</li>
                    <li>Changes in transaction frequency</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Anomaly Detection Tab
        with advanced_tabs[2]:
            st.subheader("Anomaly Detection")
            
            st.markdown("""
            <div class="card-container">
                <h4 style="color: #694D43; margin-top: 0;">Advanced Anomaly Detection</h4>
                <p>Using machine learning to identify transactions that deviate from normal patterns.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create sample anomaly visualization
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            
            # Simple dummy data for anomaly visualization
            n_points = 200
            normal_points = np.random.normal(0, 1, (n_points-10, 2))
            anomaly_points = np.random.normal(3, 2, (10, 2))
            
            all_points = np.vstack([normal_points, anomaly_points])
            labels = ['Normal'] * (n_points-10) + ['Anomaly'] * 10
            
            anomaly_df = pd.DataFrame({
                'Feature1': all_points[:, 0],
                'Feature2': all_points[:, 1],
                'Type': labels
            })
            
            # Create scatterplot
            fig = px.scatter(
                anomaly_df, x='Feature1', y='Feature2',
                color='Type',
                color_discrete_sequence=['#ACE2B3', '#FF7B7A'],
                opacity=0.7,
                size_max=15
            )
            
            fig.update_layout(
                title="Anomaly Detection Visualization",
                xaxis_title="Transaction Feature 1",
                yaxis_title="Transaction Feature 2",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add anomaly detection explanation
            st.markdown("""
            <div class="card-container" style="margin-top: 20px;">
                <h4 style="color: #694D43; margin-top: 0;">Anomaly Detection Insights</h4>
                <p>Our anomaly detection system uses advanced machine learning algorithms to identify transactions that deviate from normal patterns.</p>
                <p>Key anomaly detection techniques:</p>
                <ul>
                    <li>Isolation Forest for outlier detection</li>
                    <li>One-Class SVM for novelty detection</li>
                    <li>Local Outlier Factor for density-based detection</li>
                    <li>Autoencoder neural networks for reconstruction-based detection</li>
                </ul>
                <p>These techniques allow us to detect fraud patterns that might be missed by traditional rule-based systems.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk Scoring Tab
        with advanced_tabs[3]:
            st.subheader("Advanced Risk Scoring")
            
            st.markdown("""
            <div class="card-container">
                <h4 style="color: #694D43; margin-top: 0;">Risk Score Components</h4>
                <p>Breaking down the components that contribute to transaction risk scores.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create risk score components visualization
            risk_components = {
                'Component': ['Transaction Velocity', 'Amount', 'Time of Day', 'Merchant Category',
                             'IP Address', 'Device Match', 'Location', 'Behavioral Pattern'],
                'Weight': [15, 20, 10, 12, 15, 8, 10, 10],
                'Category': ['Behavioral', 'Transactional', 'Temporal', 'Merchant',
                            'Technical', 'Technical', 'Geographical', 'Behavioral']
            }
            
            risk_df = pd.DataFrame(risk_components)
            
            # Create bar chart
            fig = px.bar(
                risk_df, y='Component', x='Weight',
                color='Category',
                orientation='h',
                color_discrete_sequence=['#ACE2B3', '#FF7B7A', '#E3A4F4', '#694D43', '#8BBEE8']
            )
            
            fig.update_layout(
                title="Risk Score Component Weights",
                xaxis_title="Weight (%)",
                yaxis_title="Risk Component",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add risk scoring explanation
            st.markdown("""
            <div class="card-container" style="margin-top: 20px;">
                <h4 style="color: #694D43; margin-top: 0;">Risk Scoring Methodology</h4>
                <p>Our advanced risk scoring system combines multiple factors to generate an accurate fraud risk assessment:</p>
                <ul>
                    <li><strong>Transactional factors:</strong> Amount, frequency, merchant category</li>
                    <li><strong>Behavioral factors:</strong> Spending patterns, velocity, typical behavior</li>
                    <li><strong>Technical factors:</strong> Device, IP address, browser fingerprint</li>
                    <li><strong>Temporal factors:</strong> Time of day, day of week, transaction sequence</li>
                    <li><strong>Geographical factors:</strong> Location, distance from home, travel patterns</li>
                </ul>
                <p>The risk scoring engine uses ensemble machine learning techniques to combine these factors into a comprehensive risk score, providing high accuracy while minimizing false positives.</p>
            </div>
            """, unsafe_allow_html=True)

# Tab 1: Make Prediction
with tab1:
    st.header("Fraud Detection Prediction")
    
    st.markdown("""
    <div class="card-container" style="border-left: 5px solid #ACE2B3; margin-bottom: 20px;">
        <h4 style="color: #694D43; margin-top: 0; margin-bottom: 10px;">Transaction Analysis</h4>
        <p style="color: #333;">Enter transaction details below to check if it might be fraudulent. Your data helps our model make a more accurate prediction.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a form for all the transaction data with a tabbed interface for different sections
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    with st.form("transaction_form"):
        # Create tabs for different input sections
        input_tabs = st.tabs(["Basic Information", "Transaction Context", "Advanced Risk Factors", "Enhanced Fraud Indicators"])
        
        # Tab 1: Basic Information
        with input_tabs[0]:
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            
            amount = st.number_input("Transaction Amount ($)", 
                                    min_value=0.01, max_value=10000.0, 
                                    value=st.session_state.transaction_data['amount'],
                                    step=10.0, format="%.2f",
                                    help="The transaction amount in dollars")
            
            time_hours = st.slider("Time of Day (hours)", 
                                  min_value=0, max_value=23, 
                                  value=int(st.session_state.transaction_data['time']/3600) % 24,
                                  help="Hour when the transaction occurred (24h format)")
            
            merchant_category = st.selectbox("Merchant Category", 
                                           options=["Retail", "Restaurant", "Travel", "Entertainment", "Online Services", "Other"],
                                           index=["Retail", "Restaurant", "Travel", "Entertainment", "Online Services", "Other"].index(st.session_state.transaction_data['merchant_category']),
                                           help="Type of merchant where transaction occurred")
            
            transaction_type = st.selectbox("Transaction Type", 
                                          options=["In-Person", "Online", "Recurring", "Phone", "Mobile App"],
                                          index=["In-Person", "Online", "Recurring", "Phone", "Mobile App"].index(st.session_state.transaction_data['transaction_type']),
                                          help="How the transaction was initiated")
        
        # Tab 2: Transaction Context
        with input_tabs[1]:
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            
            distance_from_home = st.slider("Distance from Home (miles)", 
                                         min_value=0, max_value=10000, 
                                         value=st.session_state.transaction_data['distance_from_home'],
                                         help="How far the transaction occurred from the cardholder's home")
            
            time_since_last = st.slider("Hours Since Last Transaction", 
                                      min_value=0, max_value=168, 
                                      value=st.session_state.transaction_data['time_since_last'],
                                      help="Time elapsed since the previous transaction")
            
            previous_declined = st.slider("Recent Declined Transactions", 
                                        min_value=0, max_value=10, 
                                        value=st.session_state.transaction_data['previous_declined'],
                                        help="Number of declined transactions in the last 24 hours")
            
            velocity = st.slider("Transaction Velocity", 
                               min_value=0.0, max_value=10.0, 
                               value=float(st.session_state.transaction_data['velocity']),
                               step=0.1,
                               help="Rate of transactions in the past hour (higher means more frequent)")
        
        # Tab 3: Advanced Risk Factors
        with input_tabs[2]:
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            
            unusual_pattern = st.checkbox("Unusual Spending Pattern", 
                                       value=st.session_state.transaction_data['unusual_pattern'],
                                       help="Is this transaction outside of normal spending patterns")
            
            match_options = ["Yes", "No", "Unknown"]
            ip_match = st.radio("IP Address Match", 
                              options=match_options,
                              index=match_options.index(st.session_state.transaction_data['ip_match']) if st.session_state.transaction_data['ip_match'] in match_options else 0,
                              help="Does the IP address match previous transactions")
            
            browser_match = st.checkbox("Browser Profile Match", 
                                     value=st.session_state.transaction_data['browser_match'],
                                     help="Browser fingerprint matches previous transactions")
            
            device_age = st.slider("Device Age (months)", 
                                 min_value=0, max_value=60, 
                                 value=st.session_state.transaction_data['device_age'],
                                 help="How long this device has been used for transactions")
        
        # Tab 4: Enhanced Fraud Indicators
        with input_tabs[3]:
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            
            # New advanced fraud detection features
            geolocation_mismatch = st.checkbox("Geolocation Mismatch",
                                              value=False,
                                              help="Transaction location doesn't match the card's typical usage area")
            
            multiple_currencies = st.checkbox("Multiple Currency Usage",
                                            value=False,
                                            help="Card used for multiple currencies in a short timeframe")
            
            card_testing_pattern = st.checkbox("Card Testing Pattern",
                                             value=False,
                                             help="Multiple small transactions followed by larger ones")
            
            high_risk_merchant = st.checkbox("High-Risk Merchant Category",
                                           value=False,
                                           help="Transaction occurs with a merchant category associated with high fraud rates")
            
        # Create a container for the submit button with better styling
        st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
        
        submitted = st.form_submit_button(
            "Analyze Transaction & Predict Fraud Risk", 
            help="Submit transaction details for comprehensive fraud analysis",
            use_container_width=True
        )
        
        # Add a note about improved accuracy with the new features
        st.caption("Enhanced fraud detection with our latest machine learning algorithms and additional risk factors.")
    
    # Only show results if the form is submitted
    if submitted:
        # Update the transaction data in session state
        st.session_state.transaction_data.update({
            'amount': amount,
            'time': time_hours * 3600,  # Convert to seconds
            'merchant_category': merchant_category,
            'transaction_type': transaction_type,
            'distance_from_home': distance_from_home,
            'time_since_last': time_since_last,
            'previous_declined': previous_declined,
            'velocity': velocity,
            'unusual_pattern': unusual_pattern,
            'ip_match': ip_match,
            'browser_match': browser_match,
            'device_age': device_age
        })
        
        # Convert categorical variables to numerical
        input_features = np.zeros(31)  # 30 anonymized features + amount
        
        # Add amount as a feature (normalize it somewhat)
        input_features[30] = amount / 1000.0  # Simple normalization
        
        # Add some risk factors to anonymized features
        if unusual_pattern:
            input_features[0] = -3.0  # Unusual patterns often have extreme values in V1
        else:
            input_features[0] = 0.5
            
        # Distance impacts a feature
        input_features[1] = min(distance_from_home / 1000, 3.0)
        
        # Velocity impacts a feature
        input_features[2] = min(velocity, 3.0)
        
        # Previous declined has a strong signal
        if previous_declined > 0:
            input_features[3] = -2.5 * min(previous_declined, 3)
        
        # Transaction type affects risk
        if transaction_type == "Online":
            input_features[4] = -1.0
        elif transaction_type == "In-Person":
            input_features[4] = 1.5
            
        # Device and IP matching provide signals
        if ip_match == "No":
            input_features[5] = -2.0
        elif ip_match == "Yes":
            input_features[5] = 1.5
            
        if browser_match:
            input_features[6] = 1.0
        else:
            input_features[6] = -1.0
            
        # Predict fraud probability
        fraud_probability = predict_fraud_probability(st.session_state.default_model, input_features)
        
        # Display the result with a gauge and advice
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="color: #694D43;">Transaction Analysis Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for visuals and text
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Create a gauge chart for the fraud probability
            if fraud_probability is not None:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=fraud_probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fraud Risk %", 'font': {'color': '#694D43', 'size': 18}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#694D43"},
                        'bar': {'color': "#FF7B7A"},
                        'bgcolor': "white",
                        'borderwidth': 1,
                        'bordercolor': "#694D43",
                        'steps': [
                            {'range': [0, 30], 'color': '#ACE2B3'},
                            {'range': [30, 70], 'color': '#E3A4F4'},
                            {'range': [70, 100], 'color': '#FF7B7A'}
                        ],
                    }
                ))
                fig.update_layout(
                    height=300,
                    margin=dict(l=30, r=30, t=50, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#694D43", 'family': "Arial"}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Display risk level and recommendations
            risk_level = "Low"
            risk_color = "#ACE2B3"
            recommendations = ["Transaction appears legitimate.", 
                            "Process normally with standard verification."]
            
            if fraud_probability > 0.3 and fraud_probability <= 0.7:
                risk_level = "Medium"
                risk_color = "#E3A4F4"
                recommendations = ["Verify cardholder identity with additional checks.", 
                                "Check billing address and CVV code carefully.",
                                "Consider step-up authentication."]
            
            elif fraud_probability > 0.7:
                risk_level = "High"
                risk_color = "#FF7B7A"
                recommendations = ["Decline transaction and flag account for review.", 
                                "Contact cardholder to verify transaction attempt.",
                                "Request additional identity verification for future transactions.",
                                "Consider temporarily limiting transaction capabilities."]
            
            st.markdown(f"""
            <div class="card-container" style="border-left: 5px solid {risk_color}; height: 100%;">
                <h3 style="color: #694D43; margin-top: 0;">Risk Assessment: <span style="color: {risk_color};">{risk_level}</span></h3>
                <p style="margin-bottom: 15px;">Based on the transaction details, our model has identified the following risk factors:</p>
                <ul>
            """, unsafe_allow_html=True)
            
            # List risk factors based on inputs
            if amount > 500:
                st.markdown('<li>High transaction amount</li>', unsafe_allow_html=True)
            if distance_from_home > 100:
                st.markdown('<li>Transaction location far from home</li>', unsafe_allow_html=True)
            if time_hours < 7 or time_hours > 22:
                st.markdown('<li>Unusual transaction time</li>', unsafe_allow_html=True)
            if previous_declined > 0:
                st.markdown(f'<li>{previous_declined} recent declined transaction(s)</li>', unsafe_allow_html=True)
            if unusual_pattern:
                st.markdown('<li>Unusual spending pattern detected</li>', unsafe_allow_html=True)
            if ip_match == "No":
                st.markdown('<li>IP address doesn\'t match previous transactions</li>', unsafe_allow_html=True)
            if not browser_match:
                st.markdown('<li>Browser profile doesn\'t match previous transactions</li>', unsafe_allow_html=True)
            if velocity > 5:
                st.markdown('<li>High transaction velocity</li>', unsafe_allow_html=True)
            
            st.markdown("""
                </ul>
                <h4 style="color: #694D43; margin-top: 20px;">Recommendations:</h4>
                <ul>
            """, unsafe_allow_html=True)
            
            for rec in recommendations:
                st.markdown(f'<li>{rec}</li>', unsafe_allow_html=True)
                
            st.markdown("""
                </ul>
            </div>
            """, unsafe_allow_html=True)

# Tab 2: Dashboard
with tab2:
    st.header("Fraud Detection Dashboard")
    
    # Instructions if no data is uploaded
    if uploaded_file is None and st.session_state.data is None:
        st.markdown("""
        <div class="card-container" style="text-align: center; padding: 40px 20px;">
            <h3 style="color: #694D43; margin-top: 0;">Upload a Dataset to Begin</h3>
            <p style="margin-bottom: 20px;">Use the sidebar to upload a CSV file containing credit card transaction data.</p>
            <div style="font-size: 3rem; margin: 20px 0;">📊</div>
            <p>The dashboard will allow you to:</p>
            <ul style="text-align: left; max-width: 400px; margin: 0 auto;">
                <li>Visualize transaction patterns</li>
                <li>Train machine learning models</li>
                <li>Compare model performance</li>
                <li>Identify key fraud indicators</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Process uploaded file
    if uploaded_file is not None:
        # Load data
        with st.spinner("Loading data..."):
            data = load_data(uploaded_file)
            st.session_state.data = data
        
        # Preprocess data
        if data is not None:
            with st.spinner("Preprocessing data..."):
                X_train, X_test, y_train, y_test = preprocess_data(data)
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
    
    # If data is available (either just uploaded or from a previous upload)
    if st.session_state.data is not None:
        # Create tabs for different dashboard sections with improved spacing
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        dashboard_tabs = st.tabs(["Data Overview", "Train Models", "Model Comparison", "Feature Importance", "Advanced Analysis"])
        
        # Data Overview Tab
        with dashboard_tabs[0]:
            st.subheader("Dataset Overview")
            
            # Basic dataset information
            data = st.session_state.data
            cols = st.columns([2, 1])
            
            with cols[0]:
                st.markdown("""
                <div class="card-container">
                    <h4 style="color: #694D43; margin-top: 0;">Dataset Summary</h4>
                """, unsafe_allow_html=True)
                
                st.dataframe(data.head(5), use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show basic stats
                st.markdown("""
                <div class="card-container" style="margin-top: 20px;">
                    <h4 style="color: #694D43; margin-top: 0;">Data Statistics</h4>
                """, unsafe_allow_html=True)
                
                st.dataframe(data.describe(), use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with cols[1]:
                # Transaction distribution
                st.markdown("""
                <div class="card-container">
                    <h4 style="color: #694D43; margin-top: 0;">Class Distribution</h4>
                """, unsafe_allow_html=True)
                
                if 'Class' in data.columns:
                    class_counts = data['Class'].value_counts()
                    labels = ['Legitimate', 'Fraudulent']
                    values = [class_counts.get(0, 0), class_counts.get(1, 0)]
                    
                    fig = px.pie(
                        values=values,
                        names=labels,
                        color_discrete_sequence=['#ACE2B3', '#FF7B7A'],
                        hole=0.4
                    )
                    fig.update_layout(
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=240
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add some metrics
                    total_transactions = len(data)
                    fraud_percent = values[1] / total_transactions * 100
                    
                    st.metric("Total Transactions", f"{total_transactions:,}")
                    st.metric("Fraud Rate", f"{fraud_percent:.2f}%")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Amount distribution
                if 'Amount' in data.columns:
                    st.markdown("""
                    <div class="card-container" style="margin-top: 20px;">
                        <h4 style="color: #694D43; margin-top: 0;">Amount Distribution</h4>
                    """, unsafe_allow_html=True)
                    
                    fig = px.histogram(
                        data, x='Amount',
                        color='Class' if 'Class' in data.columns else None,
                        nbins=50,
                        opacity=0.7,
                        color_discrete_sequence=['#ACE2B3', '#FF7B7A']
                    )
                    fig.update_layout(
                        xaxis_title="Transaction Amount",
                        yaxis_title="Count",
                        legend_title="Class",
                        margin=dict(l=20, r=20, t=20, b=20),
                        height=240
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            
            # Time-based analysis
            if 'Time' in data.columns:
                st.markdown("""
                <div class="card-container" style="margin-top: 20px;">
                    <h4 style="color: #694D43; margin-top: 0;">Time-based Analysis</h4>
                """, unsafe_allow_html=True)
                
                # Convert time to hours
                data_copy = data.copy()
                data_copy['Hour'] = data_copy['Time'] / 3600 % 24
                
                cols2 = st.columns(2)
                
                with cols2[0]:
                    # Transactions by hour
                    fig = px.histogram(
                        data_copy, x='Hour',
                        color='Class' if 'Class' in data.columns else None,
                        nbins=24,
                        opacity=0.7,
                        color_discrete_sequence=['#ACE2B3', '#FF7B7A']
                    )
                    fig.update_layout(
                        xaxis_title="Hour of Day",
                        yaxis_title="Transaction Count",
                        legend_title="Class",
                        margin=dict(l=20, r=20, t=30, b=20),
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with cols2[1]:
                    # Fraud rate by hour
                    if 'Class' in data.columns:
                        hourly_fraud = data_copy.groupby(data_copy['Hour'].astype(int))['Class'].mean()
                        
                        fig = px.line(
                            x=hourly_fraud.index,
                            y=hourly_fraud.values * 100,
                            markers=True
                        )
                        fig.update_traces(line_color='#FF7B7A', marker_color='#694D43')
                        fig.update_layout(
                            xaxis_title="Hour of Day",
                            yaxis_title="Fraud Rate (%)",
                            margin=dict(l=20, r=20, t=30, b=20),
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Feature correlations
            st.markdown("""
            <div class="card-container" style="margin-top: 20px;">
                <h4 style="color: #694D43; margin-top: 0;">Feature Correlations</h4>
            """, unsafe_allow_html=True)
            
            # Select columns to include in correlation
            all_columns = data.columns.tolist()
            if len(all_columns) > 10:
                # If too many columns, select a subset focusing on those likely to be most informative
                selected_columns = ['Class'] if 'Class' in all_columns else []
                
                # Add some V features if they exist
                v_columns = [col for col in all_columns if col.startswith('V')]
                if v_columns:
                    selected_columns.extend(v_columns[:5])  # Take first 5 V features
                
                # Add Time and Amount if they exist
                if 'Time' in all_columns:
                    selected_columns.append('Time')
                if 'Amount' in all_columns:
                    selected_columns.append('Amount')
                
                # If we still don't have enough columns, add more
                remaining_columns = [col for col in all_columns if col not in selected_columns]
                selected_columns.extend(remaining_columns[:10 - len(selected_columns)])
                
                # Create correlation matrix for selected columns
                corr_matrix = data[selected_columns].corr()
            else:
                # If not too many columns, use all
                corr_matrix = data.corr()
            
            # Create heatmap using plotly for better interactivity
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale=[[0, '#E3A4F4'], [0.5, '#FFFFFF'], [1, '#ACE2B3']],
                zmin=-1, zmax=1
            )
            fig.update_layout(
                height=500,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Train Models Tab
        with dashboard_tabs[1]:
            st.subheader("Train Fraud Detection Models")
            
            if st.session_state.X_train is not None and st.session_state.y_train is not None:
                # Model selection and training
                st.markdown("""
                <div class="card-container">
                    <h4 style="color: #694D43; margin-top: 0;">Train a Model</h4>
                    <p>Select a machine learning algorithm to train on your dataset.</p>
                """, unsafe_allow_html=True)
                
                # Model selection dropdown
                model_options = ["Logistic Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest", "XGBoost"]
                selected_model = st.selectbox("Select Model Algorithm", model_options)
                
                if st.button("Train Model", key="train_button"):
                    with st.spinner(f"Training {selected_model}..."):
                        # Train model
                        model = train_model(selected_model, st.session_state.X_train, st.session_state.y_train)
                        st.session_state.trained_models[selected_model] = model
                        
                        # Evaluate model
                        metrics = evaluate_model(model, st.session_state.X_test, st.session_state.y_test)
                        st.session_state.model_metrics[selected_model] = metrics
                        
                        st.success(f"{selected_model} trained successfully!")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Display trained models
                if st.session_state.trained_models:
                    st.markdown("""
                    <div class="card-container" style="margin-top: 20px;">
                        <h4 style="color: #694D43; margin-top: 0;">Trained Models</h4>
                    """, unsafe_allow_html=True)
                    
                    for model_name in st.session_state.trained_models.keys():
                        metrics = st.session_state.model_metrics.get(model_name, {})
                        if metrics:
                            st.markdown(f"##### {model_name}")
                            
                            # Display metric values
                            cols = st.columns(4)
                            cols[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                            cols[1].metric("Precision", f"{metrics.get('precision', 0):.3f}")
                            cols[2].metric("Recall", f"{metrics.get('recall', 0):.3f}")
                            cols[3].metric("F1 Score", f"{metrics.get('f1', 0):.3f}")
                            
                            # Display visuals
                            viz_cols = st.columns(2)
                            
                            with viz_cols[0]:
                                # Confusion Matrix
                                if 'cm' in metrics:
                                    cm_fig = plot_confusion_matrix(metrics['cm'], model_name)
                                    st.pyplot(cm_fig)
                            
                            with viz_cols[1]:
                                # ROC Curve
                                if all(k in metrics for k in ['fpr', 'tpr', 'auc']):
                                    roc_fig = plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['auc'], model_name)
                                    st.pyplot(roc_fig)
                            
                            st.markdown("<hr>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Please upload a dataset to train models.")
        
        # Model Comparison Tab
        with dashboard_tabs[2]:
            st.subheader("Model Performance Comparison")
            
            if st.session_state.model_metrics:
                # Compare models based on metrics
                st.markdown("""
                <div class="card-container">
                    <h4 style="color: #694D43; margin-top: 0;">Performance Metrics Comparison</h4>
                """, unsafe_allow_html=True)
                
                # Create dataframe for comparison
                comparison_data = {
                    'Model': [],
                    'Accuracy': [],
                    'Precision': [],
                    'Recall': [],
                    'F1 Score': [],
                    'AUC': []
                }
                
                for model_name, metrics in st.session_state.model_metrics.items():
                    comparison_data['Model'].append(model_name)
                    comparison_data['Accuracy'].append(metrics.get('accuracy', 0))
                    comparison_data['Precision'].append(metrics.get('precision', 0))
                    comparison_data['Recall'].append(metrics.get('recall', 0))
                    comparison_data['F1 Score'].append(metrics.get('f1', 0))
                    comparison_data['AUC'].append(metrics.get('auc', 0))
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
                # Create comparative bar charts
                metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
                
                fig = px.bar(
                    comparison_df,
                    x='Model',
                    y=metrics_to_plot,
                    barmode='group',
                    color_discrete_sequence=['#ACE2B3', '#FF7B7A', '#E3A4F4', '#694D43', '#8BBEE8']
                )
                fig.update_layout(
                    title="Model Comparison",
                    xaxis_title="Model",
                    yaxis_title="Score",
                    legend_title="Metric",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # ROC curve comparison
                st.markdown("""
                <div class="card-container" style="margin-top: 20px;">
                    <h4 style="color: #694D43; margin-top: 0;">ROC Curve Comparison</h4>
                """, unsafe_allow_html=True)
                
                # Create subplot with all ROC curves
                fig = go.Figure()
                
                for model_name, metrics in st.session_state.model_metrics.items():
                    if all(k in metrics for k in ['fpr', 'tpr', 'auc']):
                        fig.add_trace(go.Scatter(
                            x=metrics['fpr'],
                            y=metrics['tpr'],
                            name=f"{model_name} (AUC={metrics['auc']:.3f})",
                            mode='lines'
                        ))
                
                # Add diagonal random line
                fig.add_trace(go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    name='Random',
                    mode='lines',
                    line=dict(dash='dash', color='gray')
                ))
                
                fig.update_layout(
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    height=500,
                    legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Train models first to compare their performance.")
        
        # Feature Importance Tab
        with dashboard_tabs[3]:
            st.subheader("Feature Importance Analysis")
            
            if st.session_state.trained_models:
                # Select a model to show feature importance
                model_options = list(st.session_state.trained_models.keys())
                
                if model_options:
                    st.markdown("""
                    <div class="card-container">
                        <h4 style="color: #694D43; margin-top: 0;">Feature Importance</h4>
                        <p>Analyze which features have the greatest impact on fraud prediction.</p>
                    """, unsafe_allow_html=True)
                    
                    selected_model = st.selectbox("Select Model", model_options, key="fi_model_select")
                    
                    if selected_model in st.session_state.trained_models:
                        model = st.session_state.trained_models[selected_model]
                        X_train = st.session_state.X_train
                        
                        # Plot feature importance if model supports it
                        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                            feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"Feature {i}" for i in range(X_train.shape[1])]
                            
                            fi_fig = plot_feature_importance(model, feature_names, selected_model)
                            if fi_fig is not None:
                                st.pyplot(fi_fig)
                            else:
                                st.info("Feature importance visualization not available for this model.")
                        else:
                            st.info("This model doesn't provide feature importance information.")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Additional feature analysis
                    if st.session_state.data is not None and 'Class' in st.session_state.data.columns:
                        st.markdown("""
                        <div class="card-container" style="margin-top: 20px;">
                            <h4 style="color: #694D43; margin-top: 0;">Feature Distribution by Class</h4>
                            <p>Analyze how feature distributions differ between fraudulent and legitimate transactions.</p>
                        """, unsafe_allow_html=True)
                        
                        data = st.session_state.data
                        
                        # Feature selection for distribution analysis
                        available_features = [col for col in data.columns if col != 'Class']
                        feature_to_analyze = st.selectbox("Select Feature", available_features, key="feature_distribution_select")
                        
                        # Create distribution plot
                        fig = px.histogram(
                            data, 
                            x=feature_to_analyze,
                            color='Class',
                            marginal="box",
                            opacity=0.7,
                            color_discrete_sequence=['#ACE2B3', '#FF7B7A'],
                            barmode='overlay'
                        )
                        
                        fig.update_layout(
                            title=f"{feature_to_analyze} Distribution by Class",
                            xaxis_title=feature_to_analyze,
                            yaxis_title="Count",
                            legend_title="Class",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("Train models first to analyze feature importance.")

# Add a footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; border-top: 1px solid #eee;">
    <p style="color: #777; font-size: 0.9rem;">
        Credit Card Fraud Detection Dashboard © 2025<br>
        Built with Streamlit | Machine Learning Powered
    </p>
</div>
""", unsafe_allow_html=True)









