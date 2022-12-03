import streamlit as st
import joblib
import time
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from numpy import array
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from collections import OrderedDict
from sklearn.datasets import make_classification
from sklearn.svm import SVC
import altair as alt

#display
st.set_page_config(page_title="Arifatul Maghfiroh", page_icon='logo.png')

@st.cache()
def progress():
    with st.spinner('Wait for it...'):
        time.sleep(5)

st.title("UAS DATA MINING C")
st.write("Arifatul Maghfiroh - 200411100201")

upload_data, preporcessing, modeling, implementation = st.tabs(["Fruits Data", "Prepocessing", "Modeling", "Implementation"])


with upload_data:
    progress()
    url = "https://www.kaggle.com/datasets/mjamilmoughal/fruits-with-colors-dataset"
    st.markdown(
        f'[Dataset BMI]({url})')
    data = pd.read_table("https://raw.githubusercontent.com/Arifaaa/dataset/main/fruit_data_with_colors.txt")
    st.dataframe(data)


with preporcessing:
    progress()
    data['fruit_name'].value_counts()
    X=data.drop(columns=['fruit_name','fruit_subtype'],axis=1)

    X
    
    x = data[["mass","width","height","color_score"]]
    y = data["fruit_label"].values
    
    st.write("""# Normalisasi MinMaxScaler""")
    "### Mengubah skala nilai terkecil dan terbesar dari dataset ke skala tertentu.pada dataset ini skala terkecil = 0, skala terbesar= 1"
    
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    x_scaled= scaler.fit_transform(x)
    x_scaled

with modeling:
    progress()
    x_train, x_test,y_train,y_test= train_test_split(x,y,random_state=0)    
    x_train_scaled, x_test_scaled,y_train_scaled,y_test_scaled= train_test_split(x_scaled,y,random_state=0)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    # from sklearn.feature_extraction.text import CountVectorizer
    # cv = CountVectorizer()
    # x_train = cv.fit_transform(x_train)
    # x_test = cv.fit_transform(x_test)
    st.write("""# Modeling """)
    st.subheader("Berikut ini adalah pilihan untuk Modeling")
    st.write("Pilih Model yang Anda inginkan untuk Cek Akurasi")
    naive = st.checkbox('Naive Bayes')
    kn = st.checkbox('K-Nearest Neighbor')
    des = st.checkbox('Decision Tree')
    mod = st.button("Modeling")
    
    # NB
    GaussianNB(priors=None)

    # Fitting Naive Bayes Classification to the Training set with linear kernel
    nvklasifikasi = GaussianNB()
    nvklasifikasi = nvklasifikasi.fit(x_train, y_train)

    # Predicting the Test set results
    y_pred = nvklasifikasi.predict(x_test)
    
    y_compare = np.vstack((y_test,y_pred)).T
    nvklasifikasi.predict_proba(x_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))
    # akurasi = 10

    # KNN 
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(x_train,y_train)
    y_pred=knn.predict(x_test)

    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))

    # DT

    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    # prediction
    dt.score(x_test, y_test)
    y_pred = dt.predict(x_test)
    #Accuracy
    akurasiii = round(100 * accuracy_score(y_test,y_pred))

    if naive :
        if mod :
            st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(akurasi))
    if kn :
        if mod:
            st.write("Model KNN accuracy score : {0:0.2f}" . format(skor_akurasi))
    if des :
        if mod :
            st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(akurasiii))
    
    eval = st.button("Evaluasi semua model")
    if eval :
        # st.snow()
        source = pd.DataFrame({
            'Nilai Akurasi' : [akurasi,skor_akurasi,akurasiii],
            'Nama Model' : ['Naive Bayes','KNN','Decision Tree']
        })

        bar_chart = alt.Chart(source).mark_bar().encode(
            y = 'Nilai Akurasi',
            x = 'Nama Model'
        )

        st.altair_chart(bar_chart,use_container_width=True)
with implementation:
    st.write("# Implementation")
    mass = st.number_input('Masukkan berat buah')

    # width
    width = st.number_input('Masukkan lebar buah')
    

    # height
    width = st.number_input('Masukkan tinggi buah')

    #color_score
    color_score = st.number_input('Masukkan nilai warna buah')
    def submit():
        # input
        inputs = np.array([[mass,width,height,color_score]])
        le = joblib.load("le.save")
        model1 = joblib.load("knn.joblib")
        y_pred3 = model1.predict(inputs)
        st.write(f"Berdasarkan data yang Anda masukkan, maka anda dinyatakan : {le.inverse_transform(y_pred3)[0]}")

    all = st.button("Submit")
    if all :
        st.balloons()
        submit()
