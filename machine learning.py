import streamlit as st
import pandas as pd

st.title('Zaidan Machine Learning App')
st.info('this is machine learning app')

df = pd.read_csv('https://raw.githubusercontent.com/zaidan16-droid/machine-learning/refs/heads/main/BostonHousing.csv')
df

with st.expander
