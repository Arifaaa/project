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
       
    df= df.drop(["Unnamed: 0"], axis=1)
    df['Category'].loc[df['Category'].isin(["1=Hepatitis","2=Fibrosis", "3=Cirrhosis"])] = 1
    df['Category'].loc[df['Category'].isin(["0=Blood Donor", "0s=suspect Blood Donor"])] = 0
    df['Sex'].loc[df['Sex']=='m']=1
    df['Sex'].loc[df['Sex']=='f']=0
    
    data = pd.get_dummies(df, columns = ['Sex'],drop_first=True)
        
    X = data.drop(['Category'],axis=1)
    y = data["Category"]
    
    X=df.iloc[:,0:10].values 
    y=df.iloc[:,10].values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform
    
    #minmax scaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    st.write("Hasil Preprocesing : ", scaled)
    
    #data train dan data set
    

with modeling:
    st.write("""# Modeling """)
    
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")
    
    X=df.iloc[:,0:10].values 
    y=df.iloc[:,10].values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform 
    
    import time
    start_time = time.time()
    
    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    preddt = dt.predict(X_test)
    dtmean = cross_val_score(estimator=dt, X=X_train, y=y_train, cv =3)
    hasil = dtmean.mean()
    print(f"Decision Trees Accuracy : {dtmean.mean()}\n")
    print("Processing Time:  %s seconds " % (time.time() - start_time))
    label =["Akurat", "Error"]
    value = [hasil,1-hasil]
    explode = [0,0.2]
