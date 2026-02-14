# ML Classification Models Comparison

## Problem Statement

This project implements and compares six different machine learning classification models to predict match winner based on realistic correlations between team strength, venue conditions, match dynamics, and match outcomes. The goal is to build an end-to-end ML pipeline that includes model training, evaluation, and deployment through an interactive Streamlit web application.

This dataset contains synthetically generated match-level data for the ICC Men’s T20 World Cup 2026. It is designed specifically for machine learning and sports analytics, with realistic correlations between team strength, venue conditions, match dynamics, and match outcomes.

Each row represents a single T20 match and includes feature-engineered variables such as ICC rankings, recent form, head-to-head history, pitch type, toss results, and a probabilistically simulated winner.

⚠️ This dataset is synthetic and created for educational and analytical purposes only. It does not represent real tournament outcomes. 

The assignment requires:
- Implementation of 6 classification models
- Comprehensive evaluation using multiple metrics
- Interactive web application for model comparison and prediction
- Deployment on Streamlit Community Cloud

## Dataset Description

**Dataset:** T20 World Cup 2026 Match Dataset from Kaggle

**Source:** [T20 World Cup 2026 Match Dataset from Kaggle](https://www.kaggle.com/api/v1/datasets/download/vishardmehta/t20-world-cup-2026-match-dataset)

**Description:**
The dataset contains synthetically generated match-level data for the ICC Men’s T20 World Cup 2026. It is designed specifically for machine learning and sports analytics, with realistic correlations between team strength, venue conditions, match dynamics, and match outcomes.

**Features (11 input features):**
1. **Team_A_Ranking** - ICC T20 ranking (lower = stronger team)
2. **Team_B_Ranking** - ICC T20 ranking
3. **Team_A_Form** - Recent performance score (%)
4. **Team_B_Form** - Recent performance score (%)
5. **HeadToHead_A_Wins** - Historical wins of Team A vs Team B
6. **HeadToHead_B_Wins** - Historical wins of Team B vs Team A
7. **Venue_HomeAdvantage_A** - 1 if Team A plays in home country
8. **Venue_HomeAdvantage_B** - 1 if Team B plays in home country
9. **Avg_T20_Score_Venue** - Average total score at the venu
10. **Team_A_Tech_Index** - Synthetic technical strength index
11. **Team_B_Tech_Index** - Synthetic technical strength index
12. **Match_Total** - Simulated total runs scored in the match

**Target Variable:**
- **Winner** - Match winner (Team_A or Team_B)

**Dataset Statistics:**
- **Total Instances:** 570
- **Features:** 12
- **Classes:** 2 (Team A and Team B)
- **Missing Values:** None

## Models Used

### Comparison Table with Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.5906 | 0.7555 | 0.5695 | 0.5906 | 0.5673 | 0.3250 |
| Decision Tree | 0.6062 | 0.6974 | 0.6097 | 0.6062 | 0.6066 | 0.3944 |
| kNN | 0.6094 | 0.7476 | 0.5841 | 0.6094 | 0.5959 | 0.3733 |
| Naive Bayes | 0.5625 | 0.7377 | 0.5745 | 0.5625 | 0.5681 | 0.3299 |
| Random Forest (Ensemble) | 0.6750 | 0.8375 | 0.6504 | 0.6750 | 0.6599 | 0.4764 |
| XGBoost (Ensemble) | 0.6531 | 0.8153 | 0.6480 | 0.6531 | 0.6434 | 0.4453 |

### Observations about Model Performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Logistic Regression achieved 59.06% accuracy with 75.55% AUC. While it provides a linear decision boundary and is fast to train, its performance is limited by the non-linear relationships in the wine quality data. The model struggles with the multi-class classification task, particularly for minority classes (quality 3, 4, 8). |
| Decision Tree | Decision Tree achieved 60.62% accuracy with 69.74% AUC. The model can capture non-linear relationships but shows signs of overfitting. It performs better than Logistic Regression but still struggles with class imbalance, especially for rare quality levels. |
| kNN | K-Nearest Neighbors achieved 60.94% accuracy with 74.76% AUC, performing slightly better than Decision Tree. The model benefits from feature scaling and captures local patterns effectively. However, it's computationally expensive and sensitive to the choice of k parameter. |
| Naive Bayes | Naive Bayes achieved the lowest accuracy (56.25%) but a reasonable AUC (73.77%). The model's assumption of feature independence doesn't hold well for wine quality data, where chemical properties are correlated. Despite this, it provides a fast baseline model. |
| Random Forest (Ensemble) | Random Forest achieved the best performance (67.50% accuracy, 83.75% AUC). By combining multiple decision trees, it effectively reduces overfitting and handles non-linear relationships. The ensemble approach provides robust predictions and better generalization compared to individual models. |
| XGBoost (Ensemble) | XGBoost achieved the second-best performance (65.31% accuracy, 81.53% AUC). It effectively captures complex patterns and feature interactions. While slightly lower than Random Forest in this case, it shows strong performance with good generalization. The model benefits from gradient boosting's ability to correct errors iteratively. |

## Project Structure

```
ml-classification-comparison/
├── streamlit_app.py          # Main Streamlit application
├── train_models.py           # Script to train all models
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── model/                    # Directory for saved models
│   ├── *.pkl                # Trained model files
│   ├── scaler.pkl           # Feature scaler
│   ├── metrics.json         # Evaluation metrics
│   ├── test_data.csv        # Test dataset
│   └── test_labels.csv      # Test labels
└── .github/
    └── workflows/
        └── deploy.yml       # GitHub Actions workflow
```

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ml-classification-comparison
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   ```bash
   python download_dataset.py
   ```
   Alternatively, download manually from [T20 World Cup 2026 Match Dataset from Kaggle](https://www.kaggle.com/api/v1/datasets/download/vishardmehta/t20-world-cup-2026-match-dataset) and save as `Match_Train.csv` in the project root.

4. **Train the models**
   ```bash
   python train_models.py
   ```
   This will:
   - Load and preprocess the dataset
   - Train all 6 models
   - Calculate evaluation metrics
   - Save models and metrics to the `model/` directory

5. **Run the Streamlit app locally**
   ```bash
   streamlit run streamlit_app.py
   ```

## Usage

### Streamlit Application Features

1. **Model Comparison Page**
   - View comparison table of all models
   - Select a model to see detailed metrics
   - View confusion matrix and classification report

2. **Predict on New Data Page**
   - Upload CSV file with test data
   - Select a model for prediction
   - View predictions and download results

3. **Dataset Info Page**
   - Learn about the dataset
   - View feature statistics

## Deployment

### Streamlit Community Cloud

**Quick Deployment Steps:**

1. **Push your code to GitHub** (already done ✅)
   ```bash
   git push origin main
   ```

2. **Go to Streamlit Cloud**
   - Visit: https://streamlit.io/cloud
   - Sign in with your GitHub account

3. **Create New App**
   - Click "New App" button
   - Select repository: `ml-classification-comparison`
   - Choose branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

4. **Wait for Deployment**
   - Deployment takes 2-5 minutes
   - Monitor the deployment logs
   - Your app will be live at: `https://ml-classification-comparison.streamlit.app`

**📖 Detailed Guide:** See [STREAMLIT_DEPLOYMENT.md](STREAMLIT_DEPLOYMENT.md) for complete step-by-step instructions and troubleshooting.

**✅ Deployment Checklist:** See [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) to ensure everything is ready.

### Automatic Updates

- Streamlit Cloud automatically redeploys when you push to the `main` branch
- Just push your changes: `git push origin main`
- Wait 2-5 minutes for automatic redeployment

## Evaluation Metrics

All models are evaluated using the following metrics:

- **Accuracy**: Overall correctness of predictions
- **AUC Score**: Area Under the ROC Curve (one-vs-rest for multi-class)
- **Precision**: Ratio of true positives to all predicted positives
- **Recall**: Ratio of true positives to all actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **MCC Score**: Matthews Correlation Coefficient, a balanced measure for multi-class classification

## Technologies Used

- **Python 3.8+**
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning library
- **XGBoost** - Gradient boosting framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Joblib** - Model serialization

## Author

Avinash Mandyam


## License

This project is for educational purposes as part of BITS Pilani ML Assignment 2.

## Acknowledgments

- Kaggle Repository for the T20 World Cup 2026 dataset
- BITS Pilani for the assignment framework

