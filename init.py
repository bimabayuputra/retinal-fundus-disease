import PIL.Image
import streamlit as st
import tensorflow as tf
import PIL
import numpy as np
import pandas as pd
from tensorflow import keras
import os

st.title("SISTEM DETEKSI PENYAKIT MATA")

if "model_initiate" not in st.session_state:
    st.session_state.DenseNet201 = keras.models.load_model('Model/DenseNet201/keras.keras')
    st.session_state.DenseNet121 = keras.models.load_model('Model/DenseNet121/keras.keras')
    st.session_state.InceptionV3 = keras.models.load_model('Model/InceptionV3/keras.keras')
    st.session_state.ResNet101V2 = keras.models.load_model('Model/ResNet101V2/keras.keras')
    st.session_state.ResNet152V2 = keras.models.load_model('Model/ResNet152V2/keras.keras')
    st.session_state.model_initiate = True

label = pd.read_csv('label.csv', index_col=0)
classes_name = label.iloc[:, 0:].columns

def prediksi_label(prediksi, filename):
    _prd = []
    for _ in range(len(classes_name)):
        _prd.append(0)
    for j in range(len(prediksi[0])):
        if prediksi[0][j] > 0.4: _prd[j] = 1

    _prd_ = np.array(_prd)
    index_label = int(filename[:-4])

    true_lab = label.iloc[[index_label]].values
    _true_label = np.array(true_lab[0])
    compare_arr = np.append([_prd_], [_true_label], axis=0)
    compare_label = pd.DataFrame(compare_arr, columns=classes_name, index=['Predicted','Actual'])
    return compare_label

BASE_MODEL_PATH = "Model/"
db_arr = os.listdir(BASE_MODEL_PATH)

db_option = st.selectbox(
    "Select Database",
    (db_arr),
)

uploaded_file = st.file_uploader("**Citra Fundus**", type= ['png', 'jpg'], accept_multiple_files=False )
if uploaded_file is not None:
    uploaded_file.seek(0)

    image = tf.keras.preprocessing.image.load_img(uploaded_file)
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image/255
    img = PIL.Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    if db_option == 'DenseNet201':
        prediksi = st.session_state.DenseNet201.predict(np.expand_dims(image, axis=0))
    elif db_option == 'DenseNet121':
        prediksi = st.session_state.DenseNet121.predict(np.expand_dims(image, axis=0))
    elif db_option == 'InceptionV3':
        prediksi = st.session_state.InceptionV3.predict(np.expand_dims(image, axis=0))
    elif db_option == 'ResNet101V2':
        prediksi = st.session_state.ResNet101V2.predict(np.expand_dims(image, axis=0))
    elif db_option == 'ResNet152V2':
        prediksi = st.session_state.ResNet152V2.predict(np.expand_dims(image, axis=0))

    with col1:
        st.image(img, width=255)

    with col2:
        st.dataframe(prediksi_label(prediksi,uploaded_file.name).style.highlight_between(left=1, right=1))
        

