import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

df = pd.read_csv('digest_descriptives_merged_simple.csv')
df1 = pd.read_csv('digest_descriptives_merged.csv')

def remove_outliers(df):
  numeric_columns = df.select_dtypes(include=['number'])
  z_scores = zscore(numeric_columns)
  threshold = 3
  outlier_indices = (abs(z_scores) > threshold).any(axis=1)
  df_filtered = df[~outlier_indices]
  return df_filtered

tab1, tab2, tab3 = st.tabs(["Gunning Fog", "Flesh Kincaid", "Similarity"])

with tab1:
  col1, col2 = st.columns(2)
  with col1:
    df_filtered = remove_outliers(df)
    fig = px.box(df_filtered, x='section_chapter', y='gunning_fog')
    fig.update_layout(xaxis_title='')
    fig.add_shape(
      type="line",
      x0=-1,
      x1=len(df['section_chapter'].unique()),
      y0=9, 
      y1=9, 
      line=dict(color="red", width=2)
    )
    fig.add_shape(
      type="line",
      x0=-1,
      x1=len(df['section_chapter'].unique()),
      y0=12, 
      y1=12, 
      line=dict(color="red", width=2)
    )
    st.plotly_chart(fig, theme=None, use_container_width=True)
  with col2:
    df1_filtered = remove_outliers(df1)
    fig = px.box(df1_filtered, x='section_chapter', y='gunning_fog')
    fig.update_layout(xaxis_title='')
    fig.add_shape(
      type="line",
      x0=-1,
      x1=len(df['section_chapter'].unique()),
      y0=9, 
      y1=9, 
      line=dict(color="red", width=2)
    )
    fig.add_shape(
      type="line",
      x0=-1,
      x1=len(df['section_chapter'].unique()),
      y0=12, 
      y1=12, 
      line=dict(color="red", width=2)
    )
    st.plotly_chart(fig, theme=None, use_container_width=True)

with tab2:
  col1, col2 = st.columns(2)
  with col1:
    df_filtered = remove_outliers(df)
    fig = px.box(df_filtered, x='section_chapter', y='flesch_kincaid_grade')
    fig.update_layout(xaxis_title='')
    fig.add_shape(
      type="line",
      x0=-1,
      x1=len(df['section_chapter'].unique()),
      y0=9, 
      y1=9, 
      line=dict(color="red", width=2)
    )
    fig.add_shape(
      type="line",
      x0=-1,
      x1=len(df['section_chapter'].unique()),
      y0=12, 
      y1=12, 
      line=dict(color="red", width=2)
    )
    st.plotly_chart(fig, theme=None, use_container_width=True)
  with col2:
    df1_filtered = remove_outliers(df1)
    fig = px.box(df1_filtered, x='section_chapter', y='flesch_kincaid_grade')
    fig.update_layout(xaxis_title='')
    fig.add_shape(
      type="line",
      x0=-1,
      x1=len(df['section_chapter'].unique()),
      y0=9, 
      y1=9, 
      line=dict(color="red", width=2)
    )
    fig.add_shape(
      type="line",
      x0=-1,
      x1=len(df['section_chapter'].unique()),
      y0=12, 
      y1=12, 
      line=dict(color="red", width=2)
    )
    st.plotly_chart(fig, theme=None, use_container_width=True)

with tab3:
  col1, col2 = st.columns(2)
  
    
    
  
  

