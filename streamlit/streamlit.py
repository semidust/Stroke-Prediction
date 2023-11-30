import streamlit as st
import numpy as np
import joblib
import requests
from io import BytesIO

# load model
model_url = 'https://raw.githubusercontent.com/semidust/Stroke-Prediction/main/trained_model.sav'
response = requests.get(model_url)

loaded_model = joblib.load(BytesIO(response.content), 'rb')

def model_predict(input_data):
  input_data_array = np.array(input_data)
  input_reshape = input_data_array.reshape(1,-1)
  
  prediction = loaded_model.predict(input_reshape)

  return prediction

st.title('Stroke Prediction')

# input form
col1, col2 = st.columns(spec=[1, 1], gap='medium')

with col1:
  gender_input = st.selectbox(
    label=':blue[Gender]',
    options=('Female', 'Male')
  )

  age = st.number_input(
    label=':blue[Age]', 
    min_value=0
  )

  hypertension_input = st.selectbox(
    label=':blue[Hypertension]',
    options=('Yes', 'No')
  )

  heartdisease_input = st.selectbox(
    label=':blue[Heart Disease]',
    options=('Yes', 'No')
  )

  married_input = st.selectbox(
    label=':blue[Ever Married]',
    options=('Yes', 'No')
  )

with col2:
  work_input = st.selectbox(
    label=':blue[Work Type]',
    options=('Government', 'Self-employed', 'Private', 'Children', 'Never Worked')
  )

  residence_input = st.selectbox(
    label=':blue[Residence Type]',
    options=('Rural Area', 'Urban Area')
  )

  avg_glucose = st.number_input(
    label=':blue[Average Glucose Level]'
  )

  bmi = st.number_input(
    label=':blue[BMI]'
  )

  smoking_input = st.selectbox(
    label=':blue[Smoking Status]',
    options=('Formerly Smoked', 'Never Smoked', 'Smokes', 'Unknown')
  )

# change input value
# gender_input
if gender_input == 'Female':
  gender = 0
elif gender_input == 'Male':
  gender = 1

# hypertension_input 
if hypertension_input == 'No':
  hypertension = 0
elif hypertension_input == 'Yes':
  hypertension = 1

# heartdisease_input
if heartdisease_input == 'No':
  heart_disease = 0
elif heartdisease_input == 'Yes':
  heart_disease = 1

# married_input
if married_input == 'No':
  married = 0
elif married_input == 'Yes':
  married = 1

# work_input
if work_input == 'Government':
  work_type = 0
elif work_input == 'Never Worked':
  work_type = 1
elif work_input == 'Private':
  work_type = 2
elif work_input == 'Self-employed':
  work_type = 3
elif work_input == 'Children':
  work_type = 4  

# residence_input
if residence_input == 'Rural Area':
  residence = 0
elif residence_input == 'Urban Area':
  residence = 1

# smoking_input
if smoking_input == 'Unknown':
  smoking_status = 0
elif smoking_input == 'Formerly Smoked':
  smoking_status = 1
elif smoking_input == 'Never Smoked':
  smoking_status = 2
elif smoking_input == 'Smokes':
  smoking_status = 3

input_data = (int(gender), int(age), int(hypertension), int(heart_disease), int(married), int(work_type), int(residence), float(avg_glucose), float(bmi), int(smoking_status))

st.text('')

if st.button('Predict Stroke Risk'):
  prediction = model_predict(input_data)

  st.header('Prediction:')

  if(prediction[0] == 1):
    st.write(':red[The patient is at a high risk of having a stroke.]')
  else:
    st.write(':green[The patient is at a low risk of having a stroke.]')

  st.divider()
  st.write(
    '''
    This prediction is meant to make people aware of factors related to strokes, but it's not meant to accurately predict strokes for individuals. 
    It's best to consult a healthcare professional for health concerns.
    '''
  )

st.text('')
st.caption("Copyright © 2023 [Sammytha](https://github.com/semidust)")




                            

