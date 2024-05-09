import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

df = pd.read_csv('digest_descriptives_merged_simple.csv')

numeric_columns = df.select_dtypes(include=['number'])
z_scores = zscore(numeric_columns)
threshold = 3
outlier_indices = (abs(z_scores) > threshold).any(axis=1)
df_filtered = df[~outlier_indices]

tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])

with tab1:
  fig = px.box(
    df_filtered, 
    x='section_chapter', 
    y='gunning_fog',
  )
  fig.update_traces(
    hovertemplate="%{x} <br>Gunning Fog Index: %{y}"
  )
  fig.update_layout(
    xaxis_title=''
  )
  st.plotly_chart(fig, theme=None, use_container_width=True)
  
  

