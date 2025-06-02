import streamlit as st
import pandas as pd

st.title('Zaidan Machine Learning App')
st.info('this is machine learning app')

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/california_housing_test.csv')
df

with st.expander('Data visualitation'):
st.scatter_chart(data=df, x='median_income', y='median_house_value')

