import streamlit as st
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd

df = pd.read_csv('digest_descriptives_merged_simple.csv')

fig = px.box(df, x='section_chapter', y='gunning_fog')
fig.show()
                 
