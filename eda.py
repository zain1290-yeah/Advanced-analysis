import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("📊 Exploratory Data Analysis (EDA) App")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Dataset preview
    st.subheader("👀 Dataset Preview")
    st.write(df.head())

    # Dataset shape
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Summary statistics
    st.subheader("📈 Summary Statistics")
    st.write(df.describe(include="all"))

    # Missing values
    st.subheader("❌ Missing Values")
    st.write(df.isnull().sum())

    # Correlation heatmap (for numeric columns)
    st.subheader("🔗 Correlation Heatmap")
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numeric columns for correlation heatmap.")

    # Column selection for univariate analysis
    st.subheader("📊 Univariate Analysis")
    column = st.selectbox("Select a column for analysis", df.columns)

    if pd.api.types.is_numeric_dtype(df[column]):
        st.write(f"Summary of {column}:")
        st.write(df[column].describe())

        # Histogram
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, bins=20, ax=ax)
        ax.set_title(f"Histogram of {column}")
        st.pyplot(fig)

        # Boxplot for outliers
        fig, ax = plt.subplots()
        sns.boxplot(x=df[column], ax=ax)
        ax.set_title(f"Boxplot of {column}")
        st.pyplot(fig)

    else:
        st.write(f"Value counts of {column}:")
        st.write(df[column].value_counts())

        # Bar plot
        fig, ax = plt.subplots()
        sns.countplot(x=df[column], order=df[column].value_counts().index, ax=ax)
        ax.set_title(f"Count Plot of {column}")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Bivariate analysis (two-column comparison)
    st.subheader("🔍 Bivariate Analysis")
    col1 = st.selectbox("Select first column", df.columns, key="col1")
    col2 = st.selectbox("Select second column", df.columns, key="col2")

    if pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2]):
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[col1], y=df[col2], ax=ax)
        ax.set_title(f"Scatter Plot: {col1} vs {col2}")
        st.pyplot(fig)
    else:
        st.write("Bivariate analysis requires both columns to be numeric.")
