import pandas as pd
import streamlit as st
from PIL import Image

################### Display stuff
from inference import predict

st.title('My 1st ML app')

# Upload display image
uploaded_file = st.file_uploader('Choose an image')
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

# One line text
user_input = st.text_input('label goes here', 'one liner')
print(user_input)

# Multi line text
user_input = st.text_area('Your feedback on the model predictions?', 'Empty feedback')
print(user_input)

################## model stuff
#  Upload data for all the patients
csv_file = st.file_uploader('Choose a CSV file')
if csv_file:
    st.write('filename : ', csv_file.name)
    patient_diabetes_informations_df = pd.read_csv(csv_file, sep='\t')
    patient_diabetes_informations_df = patient_diabetes_informations_df.drop(['Y'], axis=1)
    st.write(patient_diabetes_informations_df)

# execute model
if st.button('Predict diabetes progression'):
    if csv_file is not None:
        predictions = predict(patient_diabetes_informations_df)
        st.success(f'Prediction:\n {predictions}')
    else:
        st.warning('You need to upload a csv file before')
