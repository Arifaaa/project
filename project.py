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
    data['fruit_name'].value_counts()
    X=data.drop(columns=['fruit_name','fruit_subtype'],axis=1)

    X
    
    x = data[["mass","width","height","color_score"]]
    y = data["fruit_label"].values
    
    st.write("""# Normalisasi MinMaxScaler""")
    
    data_min = x.min()
    data_max = x.max()
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    x_scaled= scaler.fit_transform(x)
    x_scaled
    
    dumies = pd.get_dummies(data.fruit_name).columns.values.tolist()
    dumies = np.array(dumies)
    
    labels = pd.DataFrame({
        '1' : [dumies[0]],
        '2' : [dumies[1]],
        '3' : [dumies[2]],
        '4' : [dumies[3]]
    })
    st.write(labels)

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
    progress()
    with st.form("my_form"):
        st.subheader("Implementasi")
        mass = st.number_input('Masukkan berat buah (mass) : ')
        width = st.number_input('Masukkan lebar buah (width) : ')
        height = st.number_input('Masukkan tinggi buah (height) : ')
        color_score = st.number_input('Masukkan skor warna (color_score) : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                mass,
                width,
                height,
                color_score
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            if model == 'K-NN':
                mod = knn 
            if model == 'Decision Tree':
                mod = dt

            input_pred = mod.predict(input_norm)


            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :', model)

            st.write(input_pred)
