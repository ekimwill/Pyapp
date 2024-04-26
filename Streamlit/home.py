import streamlit as st
import pandas as pd
import plotly.express as px
from utils import *


st.set_page_config(page_title="Solar Radiation and Temperature Analysis",
                    page_icon=":sunny:",
                    layout="wide"
                    )

# Load data
@st.cache_data
def load_data(path:str):
    data = pd.read_csv(path)
    return data


with st.sidebar:
    uploadfiles = st.file_uploader("Upload files", type=["csv", "xlsx"], accept_multiple_files=True)

    if not uploadfiles:
        st.info("Please upload files.")
        st.stop()

dfs = [load_data(uploadfile) for uploadfile in uploadfiles]

tabs = st.sidebar.radio("Select Analysis", ["Data Information", "Visualize Correlation", 
                                            "Calculate Z Score", "Time Series Analysis",
                                            "Wind Speed and Wind Direction", "Temperature Analysis",
                                            "Box Plots", "Scatter Plots"])

if tabs == "Data Information":
    for i, df in enumerate(dfs):
        st.subheader(f"File {i+1} - Data Information")
        st.write(data_info(df))
elif tabs == "Visualize Correlation":
    for i, df in enumerate(dfs):
        st.subheader(f"File {i+1} - Visualize Correlation")
        numeric_cols = df.select_dtypes(include=['float64', 'int64'])
        st.pyplot(visualize_correlation(numeric_cols))
elif tabs == "Calculate Z Score":
    for i, df in enumerate(dfs):
        st.subheader(f"File {i+1} - Calculate Z Score")
        st.write(Calculate_Z_Score(df))
elif tabs == "Time Series Analysis":
    for i, df in enumerate(dfs):
        st.subheader(f"File {i+1} - Time Series Analysis")
        TimeStamp(df)
elif tabs == "Wind Speed and Wind Direction":
    for i, df in enumerate(dfs):
        st.subheader(f"File {i+1} - Wind Speed and Wind Direction")
        st.write(wind_speed_and_wind_direction_over_time(df))
elif tabs == "Temperature Analysis":
    for i, df in enumerate(dfs):
        st.subheader(f"File {i+1} - Temperature Analysis")
        generate_histograms(df)
elif tabs == "Box Plots":
    for i, df in enumerate(dfs):
        st.subheader(f"File {i+1} - Box Plots")
        generate_box_plots(df)
elif tabs == "Scatter Plots":
    for i, df in enumerate(dfs):
        st.subheader(f"File {i+1} - Scatter Plots")
        generate_scatter_plots(df)
