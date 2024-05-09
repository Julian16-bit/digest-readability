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
df2 = df = pd.read_csv('digest_descriptives_merged_simple_withscores.csv')

def remove_outliers(df):
  numeric_columns = df.select_dtypes(include=['number'])
  z_scores = zscore(numeric_columns)
  threshold = 3
  outlier_indices = (abs(z_scores) > threshold).any(axis=1)
  df_filtered = df[~outlier_indices]
  return df_filtered

tab1, tab2, tab3, tab4 = st.tabs(["Gunning Fog", "Flesh Kincaid", "Similarity", "Info"])

with tab1:
  st.markdown("<h1 style='text-align: center; margin-bottom: 50px'>Gunning Fox Index Comparison</h1>", unsafe_allow_html=True)
  col1, col2 = st.columns(2)
  with col1:
    df_filtered = remove_outliers(df)
    fig = px.box(df_filtered, x='section_chapter', y='gunning_fog', title="Simplified Version")
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
    fig = px.box(df1_filtered, x='section_chapter', y='gunning_fog', title="Original Version")
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
  st.markdown("<h1 style='text-align: center; margin-bottom: 50px'>Flesh Kincaid Grade Comparison</h1>", unsafe_allow_html=True)
  col1, col2 = st.columns(2)
  with col1:
    df_filtered = remove_outliers(df)
    fig = px.box(df_filtered, x='section_chapter', y='flesch_kincaid_grade', title="Simplified Version")
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
    fig = px.box(df1_filtered, x='section_chapter', y='flesch_kincaid_grade', title="Original Version")
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
  st.markdown("<h1 style='text-align: center; margin-bottom: 50px'>Semantic Similarity: Original vs Simplified</h1>", unsafe_allow_html=True)
  col1, col2 = st.columns(2)
  with col1:
    fig = px.scatter(df2, x=df2.index, y='score', title="Individual Score")
    st.plotly_chart(fig, theme=None, use_container_width=True)
  with col2:
    scores_bychapter = df2.groupby('section_chapter')['score'].mean().reset_index()
    fig = px.bar(scores_bychapter, x='section_chapter', y='score', title="Mean Score by Chapter")
    fig.update_layout(xaxis_title='')
    st.plotly_chart(fig, theme=None, use_container_width=True)

with tab4:
  st.text('The simplified text was generated using gpt-3.5-turbo. \n\nThe prompt: \n"Your task is to rewrite the provided text so that it can easily be understood by a high school student. \nMake sure you include all the same information. Do not start your response with Text: or Answer: and do not use bullet points or any special formatting."')   
    
  
  

