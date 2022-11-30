import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from numpy import array
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from collections import OrderedDict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt
from sklearn.utils.validation import joblib


st.write("PROJECT DATA MINING")
st.title("Hepatitis C Prediction System")
st.write("Arifatul Maghfiroh - 200411100201")
st.write("Grade: Penambangan Data C")
dataset, preporcessing, modeling, implementation = st.tabs(["Dataset", "Prepocessing", "Modeling", "Implementation"])

with dataset:
    st.write("""# Data Hepatitis""")
    st.write("Dataset yang digunakan adalah Hepatitis-C Prediction dataset yang diambil dari https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset")
    st.write("Total datanya adalah 615 dengan data training 80% (4088) dan data testing 20% (1022)")
    
    df = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(df)


with preporcessing:
    st.write("""# Upload File""")
    df[[ "Category" "Age", "Sex", "ALB", "ALP", "ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"]].agg(['min','max'])

    df.hepatitis.value_counts()
    df = df.drop(columns=["id"])

    X = df.drop(columns="hepatitis")
    y = df.hepatitis



    
