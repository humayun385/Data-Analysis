# eda_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

st.set_page_config(page_title="Superb EDA App", layout="wide")
st.title("📊 Superb EDA Tool")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Basic info
    st.subheader("Dataset Information")
    buffer = StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Descriptive statistics
    st.subheader("Summary Statistics")
    st.write(df.describe(include='all'))

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

    # Distribution plots
    st.subheader("Feature Distributions")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        col_choice = st.selectbox("Select a numeric column", numeric_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[col_choice].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    # Boxplots
    st.subheader("Boxplot")
    if numeric_cols:
        col_choice = st.selectbox("Select column for boxplot", numeric_cols, key="box")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col_choice], ax=ax)
        st.pyplot(fig)

    # Pair plot for small datasets
    st.subheader("Pair Plot (first 5 numeric columns)")
    if len(numeric_cols) > 1:
        sns.set(style="ticks")
        subset_cols = numeric_cols[:5]
        pair_fig = sns.pairplot(df[subset_cols].dropna())
        st.pyplot(pair_fig)

    # Target variable analysis
    st.subheader("Target Variable Analysis")
    target = st.selectbox("Select a target variable (optional)", df.columns.insert(0, "None"))
    if target != "None":
        if target in numeric_cols:
            st.write("Histogram of Target Variable")
            fig, ax = plt.subplots()
            sns.histplot(df[target].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.write("Count Plot of Target Variable")
            fig, ax = plt.subplots()
            sns.countplot(x=target, data=df, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

else:
    st.info("👆 Upload a CSV file to begin EDA.")
