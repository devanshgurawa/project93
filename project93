# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)


def prediction(model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
    island_dict = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}
    island_num = island_dict[island]
    sex_dict = {'Male': 0, 'Female': 1}
    
    sex_num = sex_dict[sex]
    input_data = [[island_num, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex_num]]
    
    species = model.predict(input_data)[0]
    species_dict = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
    species_name = species_dict[species]
    return species_name


st.title('Penguin Species Prediction App')

st.sidebar.title('Input Features')
bill_length = st.sidebar.slider('Bill Length (mm)', float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()), float(df['bill_length_mm'].mean()))
bill_depth = st.sidebar.slider('Bill Depth (mm)', float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()), float(df['bill_depth_mm'].mean()))
flipper_length = st.sidebar.slider('Flipper Length (mm)', float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()), float(df['flipper_length_mm'].mean()))
body_mass = st.sidebar.slider('Body Mass (g)', float(df['body_mass_g'].min()), float(df['body_mass_g'].max()), float(df['body_mass_g'].mean()))
sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
classifier = st.sidebar.selectbox('Classifier', ('Random Forest', 'Logistic Regression', 'Support Vector Machine'))
predict_button = st.sidebar.button('Predict')


model_dict = {'Random Forest': rf_clf, 'Logistic Regression':log_reg , 'Support Vector Machine': svc_model}

# Call the prediction function when the Predict button is clicked
if predict_button:
    # Get the selected model object
    model = model_dict[classifier]
    # Use the prediction function to predict the species
    species = prediction(model, island_num, bill_length, bill_depth, flipper_length, body_mass, sex_num)
    # Print the predicted species name on the screen
    st.write('Predicted Species: ', species)
    # Get the accuracy score of the model on the test set
    y_pred = model.predict(X_test)
    acc_score = accuracy_score(y_test, y_pred)
    # Print the accuracy score on the screen
    st.write('Accuracy Score: ', acc_score)
