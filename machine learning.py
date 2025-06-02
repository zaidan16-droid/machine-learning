import streamlit as st

st.title('Zaidan Machine Learning App')
st.info('this is machine learning app')

df = pd.read_csv('https://raw.githubusercontent.com/zaidan16-droid/machine-learning/refs/heads/main/BostonHousing.csv')
df
