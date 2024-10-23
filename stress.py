import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


# Load the data
def load_data():
    data = pd.read_excel('C:/Users/engidad/OneDrive - World Health Organization/Desktop/Modeling/Data/feature_Final.xlsx')
    
    # Mapping categorical responses to binary
    mapping = {
        'Strongly Disagree': 0,
        'Disagree': 0,
        'Neutral': 0,
        'Agree': 1,
        'Strongly Agree': 1
    }
    
    for col in data.columns[:-1]: 
        data[col] = data[col].map(mapping)
    
    # Encode the target variable
    le = LabelEncoder()
    data['I feel stressed now due to one or more reasons of above'] = le.fit_transform(data['I feel stressed now due to one or more reasons of above'])
    
    return data

# Train the model based on classifier selection
def train_model(Classifier_name, params):
    data = load_data()
    X = data.drop('I feel stressed now due to one or more reasons of above', axis=1)
    y = data['I feel stressed now due to one or more reasons of above']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

   
   
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if Classifier_name == "Random Forest (with PCA)":
        pca = PCA(n_components=8)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)

    # Get classifier
    clf = get_classifier(Classifier_name, params)
    
    # Train the model
    clf.fit(X_train_scaled, y_train)
    
    # Return trained model and scaler
    return clf, scaler

# Classifier selection based on parameters
def get_classifier(Classifier_name, params):
    if Classifier_name == "KNearest Neighbors":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif Classifier_name == "Support Vector Machine":
        clf = SVC(C=params["C"], probability=True)
    elif Classifier_name == "Random Forest" or Classifier_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], 
                                     max_depth=params["max_depth"], 
                                     random_state=42)
    elif Classifier_name == "Logistic Regression":
        clf = LogisticRegression(C=params["C"], random_state=42)
    elif Classifier_name == "Decision Tree":
        clf = DecisionTreeClassifier(max_depth=params["max_depth"], random_state=42)
    elif Classifier_name == "Naive Bayes":
        clf = GaussianNB()
    return clf

# Select hyperparameters for classifiers
def add_parameter_ui(Classifier_name):
    params = dict()
    if Classifier_name == "KNearest Neighbors":
        K = st.slider("K (Number of Neighbors)", 1, 15)
        params["K"] = K
    elif Classifier_name == "Support Vector Machine":
        C = st.slider("C (Regularization parameter)", 0.01, 10.0)
        params["C"] = C
    elif Classifier_name == "Random Forest":
        max_depth = st.slider("Max Depth", 2, 20)
        n_estimators = st.slider("Number of Trees", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    elif Classifier_name == "Logistic Regression":
        C = st.slider("C (Inverse of Regularization Strength)", 0.01, 10.0)
        params["C"] = C
    elif Classifier_name == "Decision Tree":
        max_depth = st.slider("Max Depth", 2, 20)
        params["max_depth"] = max_depth
    return params

# Prediction function to show stress level and confidence
def predict_stress(model, scaler, user_inputs):
    user_inputs_scaled = scaler.transform([user_inputs])
    
    # Get the prediction (0 or 1)
    prediction = model.predict(user_inputs_scaled)[0]
    
    # Get the prediction probabilities
    prediction_prob = model.predict_proba(user_inputs_scaled)[0]

    def stress_bar(prediction_prob):
        st.write(f"### Stress Level: {'High Stress' if prediction_prob[1] > 0.5 else 'Low Stress'}")
    st.progress(prediction_prob[1])  # Show high-stress probability
    
       
    # Determine if "High Stress" or "Low Stress" and return the confidence level
    if prediction == 1:
        return f"High Stress with {prediction_prob[1] * 100:.2f}% confidence"
    else:
        return f"Low Stress with {prediction_prob[0] * 100:.2f}% confidence"

# Main Streamlit app

st.markdown("""
    <style>
    .header {
        padding: 10px;
        text-align: left;
        font-size: 25px;
        color: white;
    }
    </style>
    <div class="header">
        <h3>Workplace Stress Prediction App</h3>
    </div>
    """, unsafe_allow_html=True)


def main():

    Classifier_name = st.selectbox("### STEP 2: Select Machine Learning Classifier", 
                                           ("Random Forest", "Support Vector Machine", 
                                            "KNearest Neighbors", 
                                            "Naive Bayes", "Logistic Regression", 
                                            "Decision Tree"))

    # Add parameter selection UI based on classifier
    params = add_parameter_ui(Classifier_name)
    
    # Load user input features (binary choices for yes/no questions)
    st.sidebar.write("### STEP 1: Please Enter Your Responses for the Following Questions:")
    feature_1 = st.sidebar.selectbox('Do you feel more stressed due to traumatic events you saw before', ['No', 'Yes'])
    feature_2 = st.sidebar.selectbox('Do you feel more stressed due to too many meetings and travel', ['No', 'Yes'])
    feature_3 = st.sidebar.selectbox('Do you feel more stressed due to the conflicts/social unrest issue in your duty station', ['No', 'Yes'])
    feature_4 = st.sidebar.selectbox('Do you feel more stressed due to job insecurity', ['No', 'Yes'])
    feature_5 = st.sidebar.selectbox('Do you feel more stressed due to your separation from loved ones', ['No', 'Yes'])
    feature_6 = st.sidebar.selectbox('Do you feel more stressed due to the high workload and work commitment', ['No', 'Yes'])
    feature_7 = st.sidebar.selectbox('Do you feel more stressed due to your duty station unfavorable climate/environment', ['No', 'Yes'])
    feature_8 = st.sidebar.selectbox('Do you feel more stressed due to your health conditions?', ['No', 'Yes'])

    # Convert Yes/No to 1/0
    user_inputs = [1 if feature == 'Yes' else 0 for feature in [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]]

    # Train the model when button clicked
    if st.button("Predict Stress Level"):
        model, scaler = train_model(Classifier_name, params)
        prediction = predict_stress(model, scaler, user_inputs)
        st.write(f"### Prediction: {prediction}")

if __name__ == '__main__':
    main()



# Custom Footer


st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        <p>  Damtew Engida | © Cumbria Universtity and Robert Kennedy College</p>
    </div>
    """, unsafe_allow_html=True)
