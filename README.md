# 🏏 ML Classification Models Comparison – T20 World Cup 2026

## 📌 Problem Statement
This project implements and compares six different machine learning classification models to predict match winners based on realistic correlations between team strength, venue conditions, match dynamics, and match outcomes.  

The goal is to build an **end-to-end ML pipeline** that includes:
- Model training  
- Comprehensive evaluation  
- Deployment through an interactive **Streamlit web application**  

⚠️ **Note:** The dataset is synthetic and created for educational purposes only. It does not represent real tournament outcomes.

---

## 📊 Dataset Description
**Dataset:** [T20 World Cup 2026 Match Dataset (Kaggle)](https://www.kaggle.com/api/v1/datasets/download/vishardmehta/t20-world-cup-2026-match-dataset)  

**Features (11 input features):**
1. Team_A_Ranking – ICC T20 ranking (lower = stronger team)  
2. Team_B_Ranking – ICC T20 ranking  
3. Team_A_Form – Recent performance score (%)  
4. Team_B_Form – Recent performance score (%)  
5. HeadToHead_A_Wins – Historical wins of Team A vs Team B  
6. HeadToHead_B_Wins – Historical wins of Team B vs Team A  
7. Venue_HomeAdvantage_A – 1 if Team A plays in home country  
8. Venue_HomeAdvantage_B – 1 if Team B plays in home country  
9. Avg_T20_Score_Venue – Average total score at the venue  
10. Team_A_Tech_Index – Synthetic technical strength index  
11. Team_B_Tech_Index – Synthetic technical strength index  
12. Match_Total – Simulated total runs scored in the match  

**Target Variable:**  
- **Winner** – Match winner (Team_A or Team_B)  

**Dataset Stats:**  
- Total Instances: 570  
- Features: 12  
- Classes: 2 (Team A, Team B)  
- Missing Values: None  

---

## 🤖 Models Implemented
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (KNN)  
4. Naive Bayes (Gaussian/Multinomial)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

---

## 📈 Model Comparison

### Validation Performance
| Model                | Accuracy | AUC    | Precision | Recall | F1    | MCC   |
|-----------------------|----------|--------|-----------|--------|-------|-------|
| Logistic Regression   | 0.8250   | 0.8956 | 0.8264    | 0.8250 | 0.8252| 0.6504|
| Decision Tree         | 0.7667   | 0.7656 | 0.7667    | 0.7667 | 0.7667| 0.5312|
| KNN                   | 0.7833   | 0.8418 | 0.7836    | 0.7833 | 0.7826| 0.5641|
| Naive Bayes           | 0.7667   | 0.8803 | 0.7907    | 0.7667 | 0.7646| 0.5592|
| Random Forest         | 0.8417   | 0.8972 | 0.8416    | 0.8417 | 0.8416| 0.6817|
| XGBoost               | 0.8083   | 0.8477 | 0.8097    | 0.8083 | 0.8085| 0.6170|

### Test Performance
| Model                | Accuracy | AUC    | Precision | Recall | F1    | MCC   |
|-----------------------|----------|--------|-----------|--------|-------|-------|
| Logistic Regression   | 0.8667   | 0.8914 | 0.8667    | 0.8667 | 0.8667| 0.7285|
| Decision Tree         | 0.4333   | 0.5000 | 0.1878    | 0.4333 | 0.2620| 0.0000|
| KNN                   | 0.8667   | 0.9163 | 0.8756    | 0.8667 | 0.8673| 0.7399|
| Naive Bayes           | 0.7333   | 0.8778 | 0.7599    | 0.7333 | 0.7333| 0.4932|
| Random Forest         | 0.5667   | 0.7896 | 0.5405    | 0.5667 | 0.4607| 0.0360|
| XGBoost               | 0.5667   | 0.5023 | 0.3211    | 0.5667 | 0.4099| 0.0000|

---

## 🔎 Observations
- **Logistic Regression**: Stable, interpretable, consistently strong (Accuracy ~0.83–0.87, MCC ~0.65–0.73).  
- **Decision Tree**: Very unstable, prone to overfitting.  
- **KNN**: Excellent on test (~0.87 Accuracy, MCC ~0.74) but weaker on validation.  
- **Naive Bayes**: Moderate but consistent baseline.  
- **Random Forest**: Strong validation (~0.84 Accuracy, MCC ~0.68) but poor test (~0.57 Accuracy). Needs tuning.  
- **XGBoost**: Similar to Random Forest – decent validation, weak test. Requires hyperparameter tuning.  

---

## 📂 Project Structure

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

## Author

Avinash Mandyam


## License

This project is for educational purposes as part of BITS Pilani ML Assignment 2.

## Acknowledgments

- Kaggle Repository for the T20 World Cup 2026 dataset
- BITS Pilani for the assignment framework