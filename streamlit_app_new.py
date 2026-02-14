import streamlit as st
import pandas as pd
import joblib

MODEL_DIR = "model"

# Load scaler, label encoder, training columns
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")
training_cols = joblib.load(f"{MODEL_DIR}/training_columns.pkl")

# Load pretrained models
models = {
    "LogisticRegression": joblib.load(f"{MODEL_DIR}/logistic_regression.pkl"),
    "DecisionTree": joblib.load(f"{MODEL_DIR}/decision_tree.pkl"),
    "KNN": joblib.load(f"{MODEL_DIR}/knn.pkl"),
    "NaiveBayes": joblib.load(f"{MODEL_DIR}/naive_bayes.pkl"),
    "RandomForest": joblib.load(f"{MODEL_DIR}/random_forest.pkl"),
    "XGBoost": joblib.load(f"{MODEL_DIR}/xgboost.pkl"),
}

# Model selector with default
selected_model_name = st.selectbox(
    "Choose a model for prediction:",
    list(models.keys()),
    index=list(models.keys()).index("LogisticRegression")  # Default to LogisticRegression
)
selected_model = models[selected_model_name]

st.title("🏏 T20 World Cup Match Outcome Predictor (Pretrained Models)")

option = st.radio("Choose Prediction Mode:", ["Manual Entry", "Upload Match_Test.csv"])

# -------------------------------
# Manual Entry Mode
# -------------------------------
if option == "Manual Entry":
    st.subheader("Enter Match Details")

    venue = st.selectbox("Venue", ["Chennai","Mumbai","Kolkata","Delhi","Ahmedabad","Colombo"])
    team_a = st.text_input("Team A")
    team_b = st.text_input("Team B")
    stage = st.selectbox("Stage", ["Group","Super8","Final"])
    team_a_rank = st.number_input("Team A Ranking", min_value=1, max_value=20)
    team_b_rank = st.number_input("Team B Ranking", min_value=1, max_value=20)
    team_a_form = st.number_input("Team A Form", min_value=0.0)
    team_b_form = st.number_input("Team B Form", min_value=0.0)
    pitch_type = st.selectbox("Pitch Type", ["Flat","Spin-Friendly","Pace-Friendly"])
    avg_score = st.number_input("Avg T20 Score Venue", min_value=100, max_value=220)
    toss_winner = st.selectbox("Toss Winner", ["Team_A","Team_B"])
    toss_decision = st.selectbox("Toss Decision", ["Bat","Field"])
    team_a_tech = st.number_input("Team A Tech Index", min_value=0.0)
    team_b_tech = st.number_input("Team B Tech Index", min_value=0.0)
    match_total = st.number_input("Match Total", min_value=100, max_value=250)

    if st.button("Predict Winner"):
        input_dict = {
            "Venue": venue,
            "Team_A": team_a,
            "Team_B": team_b,
            "Stage": stage,
            "Team_A_Ranking": team_a_rank,
            "Team_B_Ranking": team_b_rank,
            "Team_A_Form": team_a_form,
            "Team_B_Form": team_b_form,
            "Pitch_Type": pitch_type,
            "Avg_T20_Score_Venue": avg_score,
            "Toss_Winner": toss_winner,
            "Toss_Decision": toss_decision,
            "Team_A_Tech_Index": team_a_tech,
            "Team_B_Tech_Index": team_b_tech,
            "Match_Total": match_total
        }

        input_df = pd.DataFrame([input_dict])
        input_df = pd.get_dummies(input_df, columns=['Venue','Team_A','Team_B','Stage',
                                                     'Pitch_Type','Toss_Winner','Toss_Decision'], drop_first=True)
        input_df = input_df.reindex(columns=training_cols, fill_value=0)
        input_scaled = scaler.transform(input_df)

        st.subheader("Predictions")
        for name, model in models.items():
            pred = model.predict(input_scaled)
            winner = label_encoder.inverse_transform(pred)[0]
            st.write(f"**{name}** predicts: 🏆 {winner}")

# -------------------------------
# Batch Prediction Mode (Match_Test.csv)
# -------------------------------
else:
    st.subheader("Upload Match_Test.csv for Batch Predictions")
    uploaded_file = st.file_uploader("Upload Match_Test.csv", type="csv")

    if uploaded_file:
        df_test = pd.read_csv(uploaded_file)
        st.write("Test Dataset Preview:", df_test.head())

        # Drop ID/date if present
        if "Match_ID" in df_test.columns and "Date" in df_test.columns:
            df_test = df_test.drop(["Match_ID","Date"], axis=1)

        # Drop Winner if present (since we want to predict it)
        if "Winner" in df_test.columns:
            df_test = df_test.drop("Winner", axis=1)

        # One-hot encode categorical features
        categorical_cols = ['Venue','Team_A','Team_B','Stage','Pitch_Type','Toss_Winner','Toss_Decision']
        df_test = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)

        # Align with training columns
        df_test = df_test.reindex(columns=training_cols, fill_value=0)

        # Scale
        df_test_scaled = scaler.transform(df_test)

        st.subheader("Batch Predictions")
        for name, model in models.items():
            preds = model.predict(df_test_scaled)
            winners = label_encoder.inverse_transform(preds)
            df_test[f"{name}_Prediction"] = winners

        st.write("Predictions Table:", df_test)
        st.download_button("Download Predictions CSV", df_test.to_csv(index=False), "Match_Test_Predictions.csv")