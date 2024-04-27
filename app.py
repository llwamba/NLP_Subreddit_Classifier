# This streamlit app run only with scikit-learn version 1.2.2

import streamlit as st
import joblib

# load the saved models
cvect_logreg_model = joblib.load("./NLP_Model/cvect_logreg_model.pkl")
tfidf_logreg_model = joblib.load("./NLP_Model/tfidf_logreg_model.pkl")
cvect_nb_model = joblib.load("./NLP_Model/cvect_nb_model.pkl")
tfidf_nb_model = joblib.load("./NLP_Model/tfidf_nb_model.pkl")
rf_cvect_model = joblib.load("./NLP_Model/rf_cvect_model.pkl")
rf_tfidf_model = joblib.load("./NLP_Model/rf_tfidf_model.pkl")
xgb_cvect_model = joblib.load("./NLP_Model/xgb_cvect_model.pkl")
xgb_tfidf_model = joblib.load("./NLP_Model/xgb_tfidf_model.pkl")

# prediction function
def predict(model, title_text, self_text):
    combined_text = title_text + " " + self_text
    prediction = model.predict([combined_text])
    return prediction[0]

# Streamlit UI
st.title("Text Classification App")

# user to input title text
title_input = st.text_area("Enter title text to classify", "")

# user to input self-text
self_text_input = st.text_area("Enter self-text to classify", "")

# Model selection
selected_model = st.selectbox("Select Model", ["CountVectorizer + Logistic Regression", "TfidfVectorizer + Logistic Regression",
                                                "CountVectorizer + Multinomial Naive Bayes", "TfidfVectorizer + Multinomial Naive Bayes",
                                                "RandomForestClassifier + CountVectorizer", "RandomForestClassifier + TfidfVectorizer",
                                                "XGBoost + CountVectorizer", "XGBoost + TfidfVectorizer"])

# button to make predictions
if st.button("Predict"):
    if title_input and self_text_input:
        if selected_model == "CountVectorizer + Logistic Regression":
            prediction = predict(cvect_logreg_model, title_input, self_text_input)
        elif selected_model == "TfidfVectorizer + Logistic Regression":
            prediction = predict(tfidf_logreg_model, title_input, self_text_input)
        elif selected_model == "CountVectorizer + Multinomial Naive Bayes":
            prediction = predict(cvect_nb_model, title_input, self_text_input)
        elif selected_model == "TfidfVectorizer + Multinomial Naive Bayes":
            prediction = predict(tfidf_nb_model, title_input, self_text_input)
        elif selected_model == "RandomForestClassifier + CountVectorizer":
            prediction = predict(rf_cvect_model, title_input, self_text_input)
        elif selected_model == "RandomForestClassifier + TfidfVectorizer":
            prediction = predict(rf_tfidf_model, title_input, self_text_input)
        elif selected_model == "XGBoost + CountVectorizer":
            prediction = predict(xgb_cvect_model, title_input, self_text_input)
        elif selected_model == "XGBoost + TfidfVectorizer":
            prediction = predict(xgb_tfidf_model, title_input, self_text_input)

        class_names = ["Music", "Movies"]
        st.write(f"Predicted class: {class_names[prediction]}")
        
    else:
        st.warning("Please enter both title and self-text for classification.")

st.sidebar.write("By Lionel Lwamba")