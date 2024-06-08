import pickle
import pandas as pd

# model deployment
import streamlit as st

# read model and holdout data
model = pickle.load(open('gb_sm.pkl', 'rb'))
X_holdout = pd.read_csv('holdout.csv', index_col=0)
holdout_transactions = X_holdout.index.to_list()

Title_bg = """
<div style="background:black;padding:10px">
<h2 style="color:white;text-align:center;"> Student Drop Out Prediction </h2>
</div>
"""
st.title("Student Status")
html_temp = """
<div style="background:#025246 ;padding:10px">
<h2 style="color:white;text-align:center;"> Student Drop Out Prediction </h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)

#adding a selectbox
choice = st.selectbox(
    "Select Student ID:",
    options = holdout_transactions)


def predict_if_drop(transaction_id):
    transaction = X_holdout.loc[transaction_id].values.reshape(1, -1)
    prediction_num = model.predict(transaction)[0]
    pred_map = {0: 'Enrolled/Graduate', 1: 'Dropped'}
    prediction = pred_map[prediction_num]
    return prediction

if st.button("Predict"):
    output = predict_if_drop(choice)

    if output == 'Enrolled/Graduate':
        st.error('This Student is Enrolled or has Graduated', icon="✅")
        st.subheader(f"Curricular units 2nd sem (approved): {X_holdout['Curricular units 2nd sem (approved)'].loc[choice]}")
        st.subheader(f"Curricular units 2nd sem (grade): {X_holdout['Curricular units 2nd sem (grade)'].loc[choice]}")
        st.subheader(f"Curricular units 1st sem (approved): {X_holdout['Curricular units 1st sem (approved)'].loc[choice]}")
        st.subheader(f"Age at enrollment: {X_holdout['Age at enrollment'].loc[choice]}")
        st.subheader(f"Curricular units 1st sem (enrolled): {X_holdout['Curricular units 1st sem (enrolled)'].loc[choice]}")
    elif output == 'Dropped':
        st.success('This Student Dropped', icon="🚨")
        st.subheader(f"Curricular units 2nd sem (approved): {X_holdout['Curricular units 2nd sem (approved)'].loc[choice]}")
        st.subheader(f"Curricular units 2nd sem (grade): {X_holdout['Curricular units 2nd sem (grade)'].loc[choice]}")
        st.subheader(f"Curricular units 1st sem (approved): {X_holdout['Curricular units 1st sem (approved)'].loc[choice]}")
        st.subheader(f"Age at enrollment: {X_holdout['Age at enrollment'].loc[choice]}")
        st.subheader(f"Curricular units 1st sem (enrolled): {X_holdout['Curricular units 1st sem (enrolled)'].loc[choice]}")
