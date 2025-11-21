import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained Decision Tree model
model = pickle.load(open('titanic_model.pkl', 'rb'))

# Custom dark theme styling
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
            color: #fafafa;
        }
        h1, h2, h3 {
            color: #00b4d8;
        }
        .stButton>button {
            background-color: #00b4d8;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🚢 Titanic Survival Prediction App")
st.caption("💡 Powered by Decision Tree Classifier | Built with Streamlit")

with st.expander("About this App"):
    st.write("""
    This app predicts **Titanic passenger survival** using a trained **Decision Tree model**.
    It takes passenger details like class, age, and fare, and estimates whether a passenger would have survived.
    """)

# Sidebar inputs
st.sidebar.header("🧭 Passenger Details")

pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
age = st.sidebar.slider("Age", 0, 100, 25)
sibsp = st.sidebar.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.sidebar.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.sidebar.number_input("Fare Paid", 0.0, 600.0, 32.0)
embarked = st.sidebar.selectbox("Port of Embarkation", ["C", "Q", "S"])
title = st.sidebar.selectbox("Passenger Title", ["Mr", "Mrs", "Miss", "Master", "Rare"])
deck = st.sidebar.selectbox("Cabin Deck", ["A", "B", "C", "D", "E", "F", "G", "N", "T"])

# Derived features
family_size = sibsp + parch + 1
is_alone = 1 if family_size == 1 else 0

# Feature dictionary
features = {
    'Pclass': pclass,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'FamilySize': family_size,
    'IsAlone': is_alone,
    'Sex_female': 1 if sex == 'Female' else 0,
    'Sex_male': 1 if sex == 'Male' else 0,
    'Emb_C': 1 if embarked == 'C' else 0,
    'Emb_Q': 1 if embarked == 'Q' else 0,
    'Emb_S': 1 if embarked == 'S' else 0,
    'Title_Master': 1 if title == 'Master' else 0,
    'Title_Miss': 1 if title == 'Miss' else 0,
    'Title_Mr': 1 if title == 'Mr' else 0,
    'Title_Mrs': 1 if title == 'Mrs' else 0,
    'Title_Rare': 1 if title == 'Rare' else 0,
    'Deck_A': 1 if deck == 'A' else 0,
    'Deck_B': 1 if deck == 'B' else 0,
    'Deck_C': 1 if deck == 'C' else 0,
    'Deck_D': 1 if deck == 'D' else 0,
    'Deck_E': 1 if deck == 'E' else 0,
    'Deck_F': 1 if deck == 'F' else 0,
    'Deck_G': 1 if deck == 'G' else 0,
    'Deck_N': 1 if deck == 'N' else 0,
    'Deck_T': 1 if deck == 'T' else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([features])
input_df = input_df[[
    'Pclass','Age','SibSp','Parch','Fare','FamilySize','IsAlone',
    'Sex_female','Sex_male','Emb_C','Emb_Q','Emb_S',
    'Title_Master','Title_Miss','Title_Mr','Title_Mrs','Title_Rare',
    'Deck_A','Deck_B','Deck_C','Deck_D','Deck_E','Deck_F','Deck_G','Deck_N','Deck_T'
]]

# Prediction button
if st.button("Predict Survival"):
    prediction = model.predict(input_df)[0]
    
    st.subheader("🎯 Prediction Result:")
    if prediction == 1:
        st.success("🟢 Passenger Likely Survived!")
    else:
        st.error("🔴 Passenger Likely Did Not Survive")
