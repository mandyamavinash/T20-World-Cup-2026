"""
ML Assignment 2 - Model Training Script
Trains 6 classification models and saves them along with evaluation metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)
import joblib
import os
import json

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load dataset
print("Loading dataset...")
# For this implementation, we'll use a publicly available version
df = pd.read_csv('Match_Train.csv')

# Drop non-predictive columns
df = df.drop(['Match_ID', 'Date'], axis=1)

# Separate features and target
X = df.drop('Winner', axis=1)
y = df['Winner']

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, 'model/label_encoder.pkl')

# One-hot encode categorical features
categorical_cols = ['Venue', 'Team_A', 'Team_B', 'Stage', 
                    'Pitch_Type', 'Toss_Winner', 'Toss_Decision']

X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# After one-hot encoding and before train-test split
training_cols = X_encoded.columns.tolist()

# Save training columns for later alignment
import joblib
joblib.dump(training_cols, "model/training_columns.pkl")


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'model/scaler.pkl')
        
# Initialize models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0)
}

# Dictionary to store all metrics
all_metrics = {}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Use scaled data for models that need it
    if model_name in ['Logistic Regression', 'KNN', 'Naive Bayes']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics (using encoded labels)
    accuracy = accuracy_score(y_test, y_pred)
    
    # For multi-class, use average='weighted' or 'macro'
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # AUC for multi-class (one-vs-rest)
    try:
        if len(np.unique(y_test)) > 2:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        else:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    except:
        auc = 0.0
    
    # Store metrics
    all_metrics[model_name] = {
        'accuracy': float(accuracy),
        'auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mcc': float(mcc),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    # Save model with proper filename mapping
    filename_map = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'KNN': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    model_filename = f"model/{filename_map[model_name]}"
    joblib.dump(model, model_filename)
    
    print(f"{model_name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
        
        # Save all metrics to JSON
    with open('model/metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    # Save test data for Streamlit app (using original indices)
    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    X_test_df.to_csv('model/test_data.csv', index=False)
    # Save both encoded and original labels
    pd.Series(y_test).to_csv('model/test_labels.csv', index=False)
    pd.Series(label_encoder.inverse_transform(y_test)).to_csv('model/test_labels_original.csv', index=False)
    
    print("\n" + "="*50)
    print("All models trained and saved successfully!")
    print("="*50)
    
    # Print summary table
    print("\nModel Performance Summary:")
    print("-" * 100)
    print(f"{'Model':<25} {'Accuracy':<12} {'AUC':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'MCC':<12}")
    print("-" * 100)
    for model_name, metrics in all_metrics.items():
        print(f"{model_name:<25} {metrics['accuracy']:<12.4f} {metrics['auc']:<12.4f} "
                f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
                f"{metrics['f1']:<12.4f} {metrics['mcc']:<12.4f}")
    print("-" * 100)

