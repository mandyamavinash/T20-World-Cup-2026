import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report
)

MODEL_DIR = "model"

# Load preprocessing artifacts
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
    stage = st.selectbox("Stage", ["Group","Super8","SemiFinal","Final"])
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
        input_proc = pd.get_dummies(input_df, columns=['Venue','Team_A','Team_B','Stage',
                                                     'Pitch_Type','Toss_Winner','Toss_Decision'], drop_first=True)
        input_proc = input_proc.reindex(columns=training_cols, fill_value=0)
        input_scaled = scaler.transform(input_proc)

        st.subheader("Predictions")
        for name, model in models.items():
            try:
                pred = model.predict(input_scaled)
                winner = label_encoder.inverse_transform(pred)[0]
                st.write(f"**{name}** predicts: 🏆 {winner}")

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_scaled)[0]
                    prob_df = pd.DataFrame({"Team": label_encoder.classes_, "Probability": probs})
                    st.write(f"### {name} Probability Distribution")
                    st.bar_chart(prob_df.set_index("Team"))
            except Exception as e:
                st.error(f"{name} could not make prediction: {e}")

# -------------------------------
# Batch Prediction Mode (Match_Test.csv)
# -------------------------------
else:
    st.subheader("Upload Match_Test.csv for Batch Predictions")

    GITHUB_URL = "https://raw.githubusercontent.com/mandyamavinash/T20-World-Cup-2026/master/Match_Test.csv"
    if st.button("Fetch Sample Test Data from GitHub"):
        response = requests.get(GITHUB_URL)
        if response.status_code == 200:
            st.download_button(
                label="Download Match_Test.csv",
                data=response.content,
                file_name="Match_Test.csv",
                mime="text/csv"
            )
        else:
            st.error("Could not fetch file from GitHub. Please check the URL.")

    uploaded_file = st.file_uploader("Upload Match_Test.csv", type=["csv"])

    if uploaded_file:
        try:
            df_raw = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not parse file: {e}")
            df_raw = None

        if df_raw is not None:
            st.write("Test Dataset Preview:", df_raw.head())

            # Keep original for display
            df_display = df_raw.copy()

            # Separate target if available
            y_true = None
            if "Winner" in df_raw.columns:
                y_true = df_raw["Winner"]
            
            # Drop unwanted columns for preprocessing
            drop_cols = ["Match_ID", "Date", "Winner"]
            df_proc = df_raw.drop(columns=[c for c in drop_cols if c in df_raw.columns])

            # Encode categorical
            expected_categorical_cols = ['Venue','Team_A','Team_B','Stage',
                                         'Pitch_Type','Toss_Winner','Toss_Decision']
            categorical_cols = [col for col in expected_categorical_cols if col in df_proc.columns]
            df_proc = pd.get_dummies(df_proc, columns=categorical_cols, drop_first=True)
            df_proc = df_proc.reindex(columns=training_cols, fill_value=0)
            df_proc = df_proc.apply(pd.to_numeric, errors='coerce')

            if df_proc.isnull().values.any():
                st.error("Test data contains NaN values after preprocessing.")
            else:
                df_proc_scaled = scaler.transform(df_proc)

                st.subheader("Batch Predictions")
                model_metrics = {}

                for name, model in models.items():
                    try:
                        preds = model.predict(df_proc_scaled)
                        winners = label_encoder.inverse_transform(preds)
                        df_display[f"{name}_Prediction"] = winners  # attach to original

                        if y_true is not None:
                            acc = accuracy_score(y_true, winners)
                            precision = precision_score(y_true, winners, average="weighted", zero_division=0)
                            recall = recall_score(y_true, winners, average="weighted", zero_division=0)
                            f1 = f1_score(y_true, winners, average="weighted", zero_division=0)
                            mcc = matthews_corrcoef(y_true, winners)

                            try:
                                if len(label_encoder.classes_) == 2 and hasattr(model, "predict_proba"):
                                    auc = roc_auc_score(y_true, model.predict_proba(df_proc_scaled)[:,1])
                                else:
                                    auc = None
                            except Exception:
                                auc = None

                            model_metrics[name] = {
                                "Accuracy": acc,
                                "AUC": auc,
                                "Precision": precision,
                                "Recall": recall,
                                "F1-score": f1,
                                "MCC": mcc
                            }

                            st.write(f"### {name} Evaluation Metrics")
                            st.write(f"Accuracy: {acc:.2f}")
                            if auc is not None:
                                st.write(f"AUC Score: {auc:.2f}")
                            st.write(f"Precision: {precision:.2f}")
                            st.write(f"Recall: {recall:.2f}")
                            st.write(f"F1-score: {f1:.2f}")
                            st.write(f"MCC: {mcc:.2f}")

                            st.write(f"### {name} Confusion Matrix")
                            cm = confusion_matrix(y_true, winners, labels=label_encoder.classes_)
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                        xticklabels=label_encoder.classes_,
                                        yticklabels=label_encoder.classes_, ax=ax)
                            st.pyplot(fig)

                            st.write(f"### {name} Classification Report")
                            st.text(classification_report(y_true, winners, zero_division=0))

                    except Exception as e:
                        st.error(f"{name} could not make predictions: {e}")

            # ✅ Show predictions table with original match info
            st.write("Predictions Table:", df_display)
            st.download_button("Download Predictions CSV", df_display.to_csv(index=False), "Match_Test_Predictions.csv")

            # ✅ Show metrics summary if ground truth exists
            if y_true is not None and model_metrics:
                st.subheader("Model Performance Comparison")
                metrics_df = pd.DataFrame(model_metrics).T.reset_index().rename(columns={"index":"Model"})
                st.write(metrics_df)
                st.download_button("Download Metrics CSV", metrics_df.to_csv(index=False), "Model_Metrics.csv")

                # Comparative charts
                fig, ax = plt.subplots(figsize=(8,5))
                sns.barplot(x="Model", y="Accuracy", data=metrics_df, ax=ax, palette="viridis")
                ax.set_ylim(0, 1)
                ax.set_title("Accuracy Across Models")
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=(8,5))
                sns.barplot(x="Model", y="F1-score", data=metrics_df, ax=ax, palette="magma")
                ax.set_ylim(0, 1)
                ax.set_title("F1-score Across Models")
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=(8,5))
                sns.barplot(x="Model", y="MCC", data=metrics_df, ax=ax, palette="coolwarm")
                ax.set_ylim(-1, 1)
                ax.set_title("MCC Across Models")
                st.pyplot(fig)
            else:
                st.info("No 'Winner' column found — showing predictions only, metrics unavailable.")