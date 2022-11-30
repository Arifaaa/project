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
st.write("Penambangan Data C")
dataset, preporcessing, modeling, implementation = st.tabs(["Dataset", "Prepocessing", "Modeling", "Implementation"])

with dataset:
    st.write("""# Data Hepatitis""")
    st.write("Dataset yang digunakan adalah Hepatitis-C Prediction dataset yang diambil dari https://www.kaggle.com/datasets/fedesoriano/hepatitis-c-dataset")
    st.write("Total datanya adalah 615 dengan data training 80% (492) dan data testing 20% (123)")
    
    df = pd.read_csv("https://raw.githubusercontent.com/Arifaaa/dataset/main/HepatitisCdata.csv")
    st.dataframe(df)


with preporcessing:
    st.write("""# Preprocessing""")
    df[["Unnamed: 0","Category" "Age", "Sex", "ALB", "ALP", "ALT", "AST", "BIL", "CHE", "CHOL", "CREA", "GGT", "PROT"]].agg(['min','max'])
    X = df.drop(labels = ["Unnamed: 0"],axis = 1)
    y = df['["Unnamed: 0"]']
    
    li = list(data["Category"])
    li2 = []
    for i in range(len(li)) :
        if li[i] == "0=Blood Donor":
                li2.append(0)
        elif li[i] == "0s=suspect Blood Donor":
            li2.append(1)
        elif li[i] == "1=Hepatitis":
            li2.append(2)
        else :
            li2.append(3)
    df["NewCategory"]= li2
    df =df.drop(["Category"], axis =1)
    df.head(615)
    
    # split data
    X = df.iloc[:,0:12].values
    y = df.iloc[:, -1].values
    
    # One Hot Encoding
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough")
    X = np.array(ct.fit_transform(X))
    
    
    #split data training dan data test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    ss=StandardScaler()
    X_test= ss.fit_transform(X_test)
    X_train = ss.fit_transform(X_train)



    
