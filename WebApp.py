import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, confusion_matrix, classification_report, matthews_corrcoef, roc_curve, 
                            roc_auc_score,accuracy_score, recall_score, precision_score, precision_recall_curve,
                            cohen_kappa_score, log_loss )
from sklearn.preprocessing import LabelEncoder


# Create a function for the first page
def Tester():
    st.write("""
    # Simple Home Credit Default Risk Prediction App using machine learning

    """)

    st.sidebar.header('User Input Parameters')
    st.sidebar.markdown("""
    [Example of input file](https://www.kaggle.com/c/home-credit-default-risk/data)""")
    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            CODE_GENDER = st.sidebar.selectbox('CODE_GENDER', ['M', 'F', 'XNA'])
            CNT_CHILDREN = st.sidebar.slider('CNT_CHILDREN', 0, 20, 10)
            NAME_TYPE_SUITE = st.sidebar.selectbox('NAME_TYPE_SUITE', ['Unaccompanied', 'Family', 'Spouse, partner', 'Children', 'Other_B', 'Other_A', 'Group of people'])
            NAME_INCOME_TYPE = st.sidebar.selectbox('NAME_INCOME_TYPE', ['Working', 'State servant', 'Commercial associate', 'Pensioner', 'Unemployed', 'Student', 'Businessman', 'Maternity leave'])
            NAME_EDUCATION_TYPE = st.sidebar.selectbox('NAME_EDUCATION_TYPE', ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'])
            NAME_HOUSING_TYPE = st.sidebar.selectbox('NAME_HOUSING_TYPE', ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment', 'Co-op apartment'])
            DAYS_BIRTH = st.sidebar.slider('DAYS_BIRTH', -25229, -7489, -15000)
            DAYS_REGISTRATION = st.sidebar.slider('DAYS_REGISTRATION', -24672, 0, -2000)
            DAYS_ID_PUBLISH = st.sidebar.slider('DAYS_ID_PUBLISH', -7197, 0, -3000)
            FLAG_EMP_PHONE = st.sidebar.selectbox('FLAG_EMP_PHONE', [0, 1])
            FLAG_WORK_PHONE = st.sidebar.selectbox('FLAG_WORK_PHONE', [0, 1])
            CNT_FAM_MEMBERS = st.sidebar.slider('CNT_FAM_MEMBERS', 1, 20, 10)
            REGION_RATING_CLIENT = st.sidebar.slider('REGION_RATING_CLIENT', 1, 3, 2)
            REGION_RATING_CLIENT_W_CITY = st.sidebar.slider('REGION_RATING_CLIENT_W_CITY', 1, 3, 2)
            REG_REGION_NOT_LIVE_REGION = st.sidebar.selectbox('REG_REGION_NOT_LIVE_REGION', [0, 1])
            REG_REGION_NOT_WORK_REGION = st.sidebar.selectbox('REG_REGION_NOT_WORK_REGION', [0, 1])
            REG_CITY_NOT_LIVE_CITY = st.sidebar.selectbox('REG_CITY_NOT_LIVE_CITY', [0, 1])
            REG_CITY_NOT_WORK_CITY = st.sidebar.selectbox('REG_CITY_NOT_WORK_CITY', [0, 1])
            LIVE_CITY_NOT_WORK_CITY = st.sidebar.selectbox('LIVE_CITY_NOT_WORK_CITY', [0, 1])
            OBS_30_CNT_SOCIAL_CIRCLE = st.sidebar.slider('OBS_30_CNT_SOCIAL_CIRCLE', 0, 348, 100)
            DEF_30_CNT_SOCIAL_CIRCLE = st.sidebar.slider('DEF_30_CNT_SOCIAL_CIRCLE', 0, 34, 15)
            OBS_60_CNT_SOCIAL_CIRCLE = st.sidebar.slider('OBS_60_CNT_SOCIAL_CIRCLE', 0, 344, 150)
            DEF_60_CNT_SOCIAL_CIRCLE = st.sidebar.slider('DEF_60_CNT_SOCIAL_CIRCLE', 0, 24, 15)
            DAYS_LAST_PHONE_CHANGE = st.sidebar.slider('DAYS_LAST_PHONE_CHANGE', -5000, 0, -1000)
            FLAG_DOCUMENT_3 = st.sidebar.selectbox('FLAG_DOCUMENT_3', [0, 1])
            AMT_REQ_CREDIT_BUREAU_YEAR = st.sidebar.slider('AMT_REQ_CREDIT_BUREAU_YEAR', 0, 30, 15)
            data = {
                'CODE_GENDER': CODE_GENDER,
                'CNT_CHILDREN': CNT_CHILDREN,
                'NAME_TYPE_SUITE': NAME_TYPE_SUITE,
                'NAME_INCOME_TYPE': NAME_INCOME_TYPE,
                'NAME_EDUCATION_TYPE': NAME_EDUCATION_TYPE,
                'NAME_HOUSING_TYPE': NAME_HOUSING_TYPE,
                'DAYS_BIRTH': DAYS_BIRTH,
                'DAYS_REGISTRATION': DAYS_REGISTRATION,
                'DAYS_ID_PUBLISH': DAYS_ID_PUBLISH,
                'FLAG_EMP_PHONE': FLAG_EMP_PHONE,
                'FLAG_WORK_PHONE': FLAG_WORK_PHONE,
                'CNT_FAM_MEMBERS': CNT_FAM_MEMBERS,
                'REGION_RATING_CLIENT': REGION_RATING_CLIENT,
                'REGION_RATING_CLIENT_W_CITY': REGION_RATING_CLIENT_W_CITY,
                'REG_REGION_NOT_LIVE_REGION': REG_REGION_NOT_LIVE_REGION,
                'REG_REGION_NOT_WORK_REGION': REG_REGION_NOT_WORK_REGION,
                'REG_CITY_NOT_LIVE_CITY': REG_CITY_NOT_LIVE_CITY,
                'REG_CITY_NOT_WORK_CITY': REG_CITY_NOT_WORK_CITY,
                'LIVE_CITY_NOT_WORK_CITY': LIVE_CITY_NOT_WORK_CITY,
                'OBS_30_CNT_SOCIAL_CIRCLE': OBS_30_CNT_SOCIAL_CIRCLE,
                'DEF_30_CNT_SOCIAL_CIRCLE': DEF_30_CNT_SOCIAL_CIRCLE,
                'OBS_60_CNT_SOCIAL_CIRCLE': OBS_60_CNT_SOCIAL_CIRCLE,
                'DEF_60_CNT_SOCIAL_CIRCLE': DEF_60_CNT_SOCIAL_CIRCLE,
                'DAYS_LAST_PHONE_CHANGE': DAYS_LAST_PHONE_CHANGE,
                'FLAG_DOCUMENT_3': FLAG_DOCUMENT_3,
                'AMT_REQ_CREDIT_BUREAU_YEAR': AMT_REQ_CREDIT_BUREAU_YEAR,
            }
            #return data
            features = pd.DataFrame(data, index=[0])
            return features
        df = user_input_features()

    st.subheader('User Input parameters')

    if uploaded_file is not None:
        st.write(df)
    else:
        st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown in left sidebar).')
        st.write(df)



    st.subheader('Random Forest Classifier Algorithms Results (Predictions) ')
    # RUS METHOD 

    ## Reads in saved classification model
    load_clf_rus = pickle.load(open('RandomForestClassifier.pkl', 'rb'))
    ## st.write('Load the saved classification model (RF-done)')
    for col_obj in df.select_dtypes("object").columns: 
        lb=LabelEncoder()
        df[col_obj]=lb.fit_transform(df[col_obj]) 
    ## Apply model to make predictions
    prediction_rus = load_clf_rus.predict(df)
    prediction_proba_rus = load_clf_rus.predict_proba(df)

    # Create three columns
    col1, col2 = st.columns(2)
    with col1:    
        st.markdown("<h3 style='color: green;'>Prediction </h3>", unsafe_allow_html=True)
        Classes = np.array(['Negative :)','positive  !!!'])
        st.write(Classes[prediction_rus])

    with col2:

        st.markdown("<h3 style='color: cyan;'>Prediction Probability</h3>", unsafe_allow_html=True)
        st.write(prediction_proba_rus)


# Create a sidebar with navigation
st.sidebar.title("Menu")
selection = st.sidebar.selectbox("Go to", ["Tester"])

# Conditional rendering based on user selection
if selection == "Tester":
    Tester()