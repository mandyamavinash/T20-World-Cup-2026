"""
ML Assignment 2 - Streamlit Web Application
Interactive app to demonstrate classification models
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="ML Classification Models Comparison",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🍷 Wine Quality Classification - ML Models Comparison</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📋 Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["📊 Model Comparison", "🔮 Predict on New Data", "📈 Dataset Info"]
)

# Load models and metrics
@st.cache_data
def load_models():
    """Load all trained models"""
    models = {}
    model_files = {
        'Logistic Regression': 'model/logistic_regression.pkl',
        'Decision Tree': 'model/decision_tree.pkl',
        'KNN': 'model/knn.pkl',
        'Naive Bayes': 'model/naive_bayes.pkl',
        'Random Forest': 'model/random_forest.pkl',
        'XGBoost': 'model/xgboost.pkl'
    }
    
    scaler = None
    scaler_path = 'model/scaler.pkl'
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.warning(f"Could not load scaler: {str(e)}")
    
    for model_name, filepath in model_files.items():
        if os.path.exists(filepath):
            try:
                models[model_name] = joblib.load(filepath)
            except Exception as e:
                st.warning(f"Could not load {model_name}: {str(e)}")
    
    return models, scaler

@st.cache_data
def load_metrics():
    """Load saved metrics"""
    metrics_path = 'model/metrics.json'
    if os.path.exists(metrics_path):
        try:
            import json
            with open(metrics_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load metrics: {str(e)}")
            return None
    return None

# Load data
try:
    models, scaler = load_models()
    metrics = load_metrics()
except Exception as e:
    st.error(f"Error loading models/metrics: {str(e)}")
    models, scaler = {}, None
    metrics = None

# Page 1: Model Comparison
if page == "📊 Model Comparison":
    st.header("Model Performance Comparison")
    
    # Check if models/metrics exist
    if not metrics and not models:
        st.error("""
        ⚠️ **Models and metrics not found!**
        
        Please ensure you have:
        1. Trained the models by running: `python train_models.py`
        2. The `model/` directory contains:
           - `metrics.json`
           - `*.pkl` files for all 6 models
           - `scaler.pkl`
        
        If you're running this on Streamlit Cloud, make sure the model files are committed to GitHub.
        """)
    elif metrics:
        # Display metrics table
        st.subheader("Evaluation Metrics Table")
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, model_metrics in metrics.items():
            comparison_data.append({
                'ML Model Name': model_name,
                'Accuracy': f"{model_metrics['accuracy']:.4f}",
                'AUC': f"{model_metrics['auc']:.4f}",
                'Precision': f"{model_metrics['precision']:.4f}",
                'Recall': f"{model_metrics['recall']:.4f}",
                'F1': f"{model_metrics['f1']:.4f}",
                'MCC': f"{model_metrics['mcc']:.4f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, width='stretch')
        
        # Model selection for detailed view
        st.subheader("Select Model for Detailed Analysis")
        selected_model = st.selectbox(
            "Choose a model",
            list(metrics.keys())
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Metrics")
            selected_metrics = metrics[selected_model]
            st.metric("Accuracy", f"{selected_metrics['accuracy']:.4f}")
            st.metric("AUC Score", f"{selected_metrics['auc']:.4f}")
            st.metric("Precision", f"{selected_metrics['precision']:.4f}")
            st.metric("Recall", f"{selected_metrics['recall']:.4f}")
            st.metric("F1 Score", f"{selected_metrics['f1']:.4f}")
            st.metric("MCC Score", f"{selected_metrics['mcc']:.4f}")
        
        with col2:
            st.markdown("### 📈 Confusion Matrix")
            cm = np.array(selected_metrics['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap with better formatting
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                       cbar_kws={'label': 'Count'}, linewidths=0.5)
            ax.set_xlabel('Predicted Quality', fontsize=12, fontweight='bold')
            ax.set_ylabel('Actual Quality', fontsize=12, fontweight='bold')
            ax.set_title(f'Confusion Matrix - {selected_model}', fontsize=14, fontweight='bold')
            
            # Set class labels if available
            if cm.shape[0] <= 10:  # Only if reasonable number of classes
                class_labels = [f'Q{i}' for i in range(cm.shape[0])]
                ax.set_xticklabels(class_labels, rotation=0)
                ax.set_yticklabels(class_labels, rotation=0)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Calculate and display accuracy from confusion matrix
            cm_accuracy = np.trace(cm) / np.sum(cm)
            st.caption(f"Accuracy from Confusion Matrix: {cm_accuracy:.4f}")
        
        # Classification Report
        st.markdown("### 📋 Classification Report")
        report = selected_metrics['classification_report']
        if isinstance(report, dict):
            # Convert to DataFrame for better display
            report_df = pd.DataFrame(report).transpose()
            # Format numeric columns
            numeric_cols = report_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                report_df[col] = report_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            st.dataframe(report_df, width='stretch')
            
            # Show summary metrics
            if 'weighted avg' in report_df.index:
                st.markdown("**Weighted Average Metrics:**")
                weighted_avg = report_df.loc['weighted avg']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Precision (Weighted)", weighted_avg.get('precision', 'N/A'))
                with col2:
                    st.metric("Recall (Weighted)", weighted_avg.get('recall', 'N/A'))
                with col3:
                    st.metric("F1-Score (Weighted)", weighted_avg.get('f1-score', 'N/A'))
    else:
        st.warning("⚠️ No metrics found. Please train the models first using train_models.py")

# Page 2: Predict on New Data
elif page == "🔮 Predict on New Data":
    st.header("Upload Test Data and Make Predictions")
    
    st.info("💡 Upload a CSV file with test data. The file should have the same features as the training data (excluding the target column).")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload test data CSV file"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded data
            try:
                test_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(test_data)} rows, {len(test_data.columns)} columns")
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(test_data.head(10), width='stretch')
                
                # Show data info
                with st.expander("📊 Data Information"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Shape:**", test_data.shape)
                        st.write("**Columns:**", list(test_data.columns))
                    with col2:
                        st.write("**Data Types:**")
                        st.write(test_data.dtypes)
                        if test_data.isnull().any().any():
                            st.warning("⚠️ Missing values detected")
                            st.write("Missing values per column:")
                            st.write(test_data.isnull().sum())
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")
                st.info("Please ensure the file is a valid CSV format.")
                st.stop()
            
            # Model selection
            if models:
                st.subheader("Select Model for Prediction")
                selected_model = st.selectbox(
                    "Choose a model",
                    list(models.keys())
                )
                
                if st.button("🔮 Make Predictions"):
                    try:
                        model = models[selected_model]
                        
                        # Validate data columns
                        expected_columns = ['fixed acidity', 'volatile acidity', 'citric acid', 
                                         'residual sugar', 'chlorides', 'free sulfur dioxide',
                                         'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
                        
                        missing_cols = set(expected_columns) - set(test_data.columns)
                        if missing_cols:
                            st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
                            st.info(f"Required columns: {', '.join(expected_columns)}")
                            st.stop()
                        
                        # Check for extra columns (warn but continue)
                        extra_cols = set(test_data.columns) - set(expected_columns)
                        if extra_cols:
                            st.warning(f"⚠️ Extra columns detected (will be ignored): {', '.join(extra_cols)}")
                        
                        # Select only required columns in correct order
                        test_data_clean = test_data[expected_columns].copy()
                        
                        # Check for missing values
                        if test_data_clean.isnull().any().any():
                            st.warning("⚠️ Missing values detected. Filling with column means.")
                            test_data_clean = test_data_clean.fillna(test_data_clean.mean())
                        
                        # Check if model needs scaled data
                        needs_scaling = selected_model in ['Logistic Regression', 'KNN', 'Naive Bayes']
                        
                        if needs_scaling and scaler:
                            X_test = scaler.transform(test_data_clean)
                        else:
                            X_test = test_data_clean.values
                        
                        # Make predictions
                        with st.spinner('Making predictions...'):
                            predictions = model.predict(X_test)
                            prediction_proba = model.predict_proba(X_test)
                        
                        # Load label encoder if exists to convert predictions back to original labels
                        label_encoder = None
                        if os.path.exists('model/label_encoder.pkl'):
                            try:
                                label_encoder = joblib.load('model/label_encoder.pkl')
                                predictions_original = label_encoder.inverse_transform(predictions)
                            except:
                                predictions_original = predictions
                        else:
                            predictions_original = predictions
                        
                        # Display results
                        st.subheader("Predictions")
                        st.success(f"✅ Successfully predicted {len(predictions)} samples")
                        
                        results_df = pd.DataFrame({
                            'Sample': range(1, len(predictions) + 1),
                            'Predicted Quality': predictions_original,
                            'Confidence': np.max(prediction_proba, axis=1)
                        })
                        
                        # Format confidence as percentage
                        results_df['Confidence'] = results_df['Confidence'].apply(lambda x: f"{x*100:.2f}%")
                        
                        st.dataframe(results_df, width='stretch', hide_index=True)
                        
                        # Show prediction statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Predictions", len(predictions))
                        with col2:
                            st.metric("Most Common Quality", int(pd.Series(predictions_original).mode()[0]))
                        with col3:
                            avg_confidence = np.max(prediction_proba, axis=1).mean()
                            st.metric("Avg Confidence", f"{avg_confidence*100:.2f}%")
                        
                        # Download predictions
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name='predictions.csv',
                            mime='text/csv'
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error making predictions: {str(e)}")
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())
            else:
                st.warning("⚠️ No models found. Please train the models first.")
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    else:
        # Show example format
        st.subheader("Expected Data Format")
        st.info("""
        The CSV file should contain the following columns (for Wine Quality dataset):
        - fixed acidity
        - volatile acidity
        - citric acid
        - residual sugar
        - chlorides
        - free sulfur dioxide
        - total sulfur dioxide
        - density
        - pH
        - sulphates
        - alcohol
        """)

# Page 3: Dataset Info
elif page == "📈 Dataset Info":
    st.header("Dataset Information")
    
    # Dataset Overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Wine Quality Dataset
        
        **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
        
        **Description:**
        This dataset contains chemical properties of red wine samples and their quality ratings.
        The quality is rated on a scale from 3 to 8 based on sensory data.
        
        **Citation:**
        P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
        Modeling wine preferences by data mining from physicochemical properties.
        In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
        """)
    
    with col2:
        st.info("""
        **Quick Stats:**
        - 📊 Instances: 1,599
        - 🔢 Features: 11
        - 🎯 Classes: 6
        - 📈 Type: Classification
        """)
    
    # Feature Information
    st.subheader("🔍 Feature Information")
    
    features_df = pd.DataFrame({
        'Feature': [
            'Fixed Acidity',
            'Volatile Acidity',
            'Citric Acid',
            'Residual Sugar',
            'Chlorides',
            'Free Sulfur Dioxide',
            'Total Sulfur Dioxide',
            'Density',
            'pH',
            'Sulphates',
            'Alcohol'
        ],
        'Unit': [
            'g/dm³',
            'g/dm³',
            'g/dm³',
            'g/dm³',
            'g/dm³',
            'mg/dm³',
            'mg/dm³',
            'g/cm³',
            'scale',
            'g/dm³',
            '% vol'
        ],
        'Description': [
            'Non-volatile acids',
            'Acetic acid amount',
            'Adds freshness',
            'Sugar remaining after fermentation',
            'Salt content',
            'Prevents microbial growth',
            'Free + bound forms',
            'Density of wine',
            'Acidity/alkalinity level',
            'Wine additive (potassium sulphate)',
            'Alcohol percentage'
        ]
    })
    
    st.dataframe(features_df, width='stretch', hide_index=True)
    
    # Dataset Exploration
    st.subheader("📊 Dataset Exploration")
    
    # Try to load the actual dataset
    dataset_loaded = False
    wine_data = None
    
    # Try multiple possible locations
    possible_paths = [
        'winequality-red.csv',
        'temp_files/winequality-red.csv',
        'model/test_data.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                # Try with semicolon delimiter first (common in Wine Quality dataset)
                wine_data = pd.read_csv(path, sep=';')
                dataset_loaded = True
                st.success(f"✅ Loaded dataset from: {path}")
                break
            except:
                try:
                    # Fallback to comma delimiter
                    wine_data = pd.read_csv(path)
                    dataset_loaded = True
                    st.success(f"✅ Loaded dataset from: {path}")
                    break
                except:
                    continue
    
    if dataset_loaded and wine_data is not None:
        # Dataset shape
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", wine_data.shape[0])
        with col2:
            st.metric("Total Columns", wine_data.shape[1])
        with col3:
            st.metric("Memory Usage", f"{wine_data.memory_usage(deep=True).sum() / 1024:.2f} KB")
        with col4:
            st.metric("Missing Values", wine_data.isnull().sum().sum())
        
        # Data Preview Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 First Rows", 
            "📋 Last Rows", 
            "📊 Statistics", 
            "📈 Distribution",
            "🔗 Correlations"
        ])
        
        with tab1:
            st.write("**First 10 rows of the dataset:**")
            st.dataframe(wine_data.head(10), width='stretch')
        
        with tab2:
            st.write("**Last 10 rows of the dataset:**")
            st.dataframe(wine_data.tail(10), width='stretch')
        
        with tab3:
            st.write("**Statistical Summary:**")
            st.dataframe(wine_data.describe(), width='stretch')
            
            st.write("**Data Types:**")
            dtype_df = pd.DataFrame({
                'Column': wine_data.columns,
                'Data Type': wine_data.dtypes.values,
                'Non-Null Count': wine_data.count().values,
                'Null Count': wine_data.isnull().sum().values
            })
            st.dataframe(dtype_df, width='stretch', hide_index=True)
        
        with tab4:
            st.write("**Target Variable Distribution:**")
            if 'quality' in wine_data.columns:
                quality_counts = wine_data['quality'].value_counts().sort_index()
                
                fig, ax = plt.subplots(figsize=(10, 5))
                quality_counts.plot(kind='bar', ax=ax, color='steelblue')
                ax.set_xlabel('Quality Rating')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Wine Quality Ratings')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Quality Distribution:**")
                    st.dataframe(
                        pd.DataFrame({
                            'Quality': quality_counts.index,
                            'Count': quality_counts.values,
                            'Percentage': (quality_counts.values / len(wine_data) * 100).round(2)
                        }),
                        width='stretch',
                        hide_index=True
                    )
                with col2:
                    st.metric("Most Common Quality", quality_counts.idxmax())
                    st.metric("Average Quality", f"{wine_data['quality'].mean():.2f}")
                    st.metric("Quality Std Dev", f"{wine_data['quality'].std():.2f}")
        
        with tab5:
            st.write("**Feature Correlation Matrix:**")
            
            # Calculate correlation matrix
            corr_matrix = wine_data.corr()
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title('Feature Correlation Heatmap')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            if 'quality' in wine_data.columns:
                st.write("**Correlation with Quality (Target):**")
                quality_corr = corr_matrix['quality'].drop('quality').sort_values(ascending=False)
                corr_df = pd.DataFrame({
                    'Feature': quality_corr.index,
                    'Correlation': quality_corr.values
                })
                st.dataframe(corr_df, width='stretch', hide_index=True)
    else:
        st.warning("⚠️ Dataset file not found. Please ensure 'winequality-red.csv' is in the project directory.")
        st.info("""
        You can download the dataset from:
        - [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
        - [Kaggle - Red Wine Quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)
        """)
    
    # Models Information
    st.subheader("🤖 Implemented Models")
    
    models_info = pd.DataFrame({
        'Model': [
            'Logistic Regression',
            'Decision Tree',
            'K-Nearest Neighbors (KNN)',
            'Naive Bayes (Gaussian)',
            'Random Forest',
            'XGBoost'
        ],
        'Type': [
            'Linear',
            'Tree-based',
            'Instance-based',
            'Probabilistic',
            'Ensemble',
            'Ensemble (Boosting)'
        ],
        'Key Characteristics': [
            'Fast, interpretable, works well with linear relationships',
            'Non-linear, interpretable, prone to overfitting',
            'Non-parametric, sensitive to feature scaling',
            'Fast, works well with independent features',
            'Reduces overfitting, handles non-linear relationships',
            'High performance, handles complex patterns'
        ]
    })
    
    st.dataframe(models_info, width='stretch', hide_index=True)
    
    # Evaluation Metrics
    st.subheader("📏 Evaluation Metrics")
    
    metrics_info = pd.DataFrame({
        'Metric': [
            'Accuracy',
            'AUC Score',
            'Precision',
            'Recall',
            'F1 Score',
            'MCC'
        ],
        'Description': [
            'Overall correctness of predictions',
            'Area under ROC curve - model discrimination ability',
            'Accuracy of positive predictions',
            'Coverage of actual positive cases',
            'Harmonic mean of precision and recall',
            'Matthews Correlation Coefficient - balanced measure'
        ],
        'Range': [
            '0 to 1 (higher is better)',
            '0 to 1 (higher is better)',
            '0 to 1 (higher is better)',
            '0 to 1 (higher is better)',
            '0 to 1 (higher is better)',
            '-1 to 1 (higher is better)'
        ]
    })
    
    st.dataframe(metrics_info, width='stretch', hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>ML Assignment 2 - Classification Models Comparison</div>",
    unsafe_allow_html=True
)

