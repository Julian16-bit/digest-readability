import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
from scipy.stats import zscore

df = pd.read_csv('digest_descriptives_merged_simple.csv')

tab1, tab2 = st.tabs(["Tab 1)", "Tab 2"])

with tab1:
  fig = px.box(df, x='section_chapter', y='gunning_fog')
  fig.update_layout(xaxis_title='')
  st.plotly_chart(fig, theme=None, use_container_width=False)
