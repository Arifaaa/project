import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import altair as alt
import time

#display
st.set_page_config(page_title="Arifatul Maghfiroh", page_icon='logo1.png')

@st.cache()
def progress():
    with st.spinner('Wait for it...'):
        time.sleep(5)

st.title("UAS DATA MINING C")
st.write("Arifatul Maghfiroh - 200411100201")

upload_data, preporcessing, modeling, implementation = st.tabs(["Fruits Data", "Preprocessing", "Modeling", "Implementation"])


with upload_data:
    progress()
    url = "https://www.kaggle.com/datasets/mjamilmoughal/fruits-with-colors-dataset"
    st.markdown(
        f'[Dataset Fruits With Colors]({url})')
    data = pd.read_table("https://raw.githubusercontent.com/Arifaaa/dataset/main/fruit_data_with_colors.txt")
    st.dataframe(data)


with preporcessing:
    progress()
    st.subheader("""Normalisasi Data""")
    data = data.drop(columns=['fruit_label','fruit_subtype'])

    X = data[["mass","width","height","color_score"]]
    y = data["fruit_name"].values
    data
    X    
    
    data_min = X.min()
    data_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)
    
    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(data.fruit_name).columns.values.tolist()
    dumies = np.array(dumies)
    
    labels = pd.DataFrame({
        '1' : [dumies[0]],
        '2' : [dumies[1]],
        '3' : [dumies[2]],
        '4' : [dumies[3]],
    })
    
    st.write(labels)

with modeling:
    progress()
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        # NB
        GaussianNB(priors=None)

        # Fitting Naive Bayes Classification to the Training set with linear kernel
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        # Predicting the Test set results
        y_pred = gaussian.predict(test)
    
        y_compare = np.vstack((test_label,y_pred)).T
        gaussian.predict_proba(test)
        gaussian_akurasi = round(100 * accuracy_score(test_label, y_pred))
        # akurasi = 10

        

        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)
        
        
with implementation:
    with st.form("my_form"):
        st.subheader("Implementasi")
        mass = st.number_input('Berat')
        width = st.number_input('Lebar')
        height = st.number_input('Tinggi')
        color_score = st.number_input('Skor Warna')
        model = st.selectbox('Model',
                ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Predict")
        if prediksi:
            inputs = np.array([
                mass,
                width,
                height,
                color_score
            ])
            
            data_min = X.min()
            data_max = X.max()
            input_norm = ((inputs - data_min) / (data_max - data_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt

            input_pred = mod.predict(input_norm)
            data_scaled = scaler.transform(inputs)
            y_imp = rf.predict(data_scaled)
            st.success(f'Data Predict = {label[y_imp[0]]}')
