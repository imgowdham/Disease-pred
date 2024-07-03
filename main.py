import streamlit as st
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("Training.csv")
tr = pd.read_csv("Testing.csv")

# Replace disease names with numerical labels
disease = {
    'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
    'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes': 7, 'Gastroenteritis': 8, 'Bronchial Asthma': 9,
    'Hypertension': 10, 'Migraine': 11, 'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13,
    'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
    'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24,
    'Tuberculosis': 25, 'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28,
    'Heart attack': 29, 'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32, 'Hypoglycemia': 33,
    'Osteoarthristis': 34, 'Arthritis': 35, '(vertigo) Paroymsal Positional Vertigo': 36, 'Acne': 37,
    'Urinary tract infection': 38, 'Psoriasis': 39, 'Impetigo': 40
}
df.replace({'prognosis': disease}, inplace=True)
tr.replace({'prognosis': disease}, inplace=True)

# Define features and target variable
features = df.drop(columns=["prognosis"])
target = df["prognosis"]

# Model training
clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(features, target)

clf_forest = RandomForestClassifier()
clf_forest.fit(features, target)

gnb = GaussianNB()
gnb.fit(features, target)

# Streamlit app
st.title("Disease Predictor using Machine Learning")
st.sidebar.title("Patient Information")

symptom1 = st.sidebar.selectbox("Symptom 1", features.columns)
symptom2 = st.sidebar.selectbox("Symptom 2", features.columns)
symptom3 = st.sidebar.selectbox("Symptom 3", features.columns)
symptom4 = st.sidebar.selectbox("Symptom 4", features.columns)
symptom5 = st.sidebar.selectbox("Symptom 5", features.columns)

symptoms = [symptom1, symptom2, symptom3, symptom4, symptom5]

# Function to predict disease
def predict_disease(model, symptoms):
    input_data = pd.DataFrame([symptoms], columns=features.columns)
    prediction = model.predict(input_data)
    return prediction[0]

# Button to predict using Decision Tree
if st.sidebar.button("Predict using Decision Tree"):
    prediction = predict_disease(clf_tree, symptoms)
    st.sidebar.text(f"Predicted Disease: {list(disease.keys())[prediction]}")

# Button to predict using Random Forest
if st.sidebar.button("Predict using Random Forest"):
    prediction = predict_disease(clf_forest, symptoms)
    st.sidebar.text(f"Predicted Disease: {list(disease.keys())[prediction]}")

# Button to predict using Naive Bayes
if st.sidebar.button("Predict using Naive Bayes"):
    prediction = predict_disease(gnb, symptoms)
    st.sidebar.text(f"Predicted Disease: {list(disease.keys())[prediction]}")
