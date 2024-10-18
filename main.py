from PIL import Image
import streamlit as st
import gdown
import os
import io
import numpy as np
from tensorflow.keras.models import load_model


def driveLink(url):
  # url ссылка на файл для чтения
  #url = 'https://drive.google.com/file/d/1Hr9IjGSZttmG6OHmqOjwqafQvlSA0UwA/view?usp=sharing'
  # вытаскиваем id файла из ссылки
  file_id = url.split('/')[-2]
  # формируем новую ссылку
  dwn_url = 'https://drive.google.com/uc?id=' + file_id
  return dwn_url

model_file = 'model.keras'
if not os.path.isfile(model_file):
  gdown.download(
    driveLink('https://drive.google.com/file/d/1GASPW3lwl_kn2C_cV-RWiRyUzXpE5YV7/view?usp=sharing'),
    output=model_file
    )



#loading model
if "model" not in st.session_state:
  st.session_state.model = None
if st.session_state.model is None:
  #загрузка модели целиком
  st.session_state.model = load_model(model_file)
  # Вывод сводки
  st.session_state.model.summary()

# Defining File Uploader Function in a variable
image_file = st.file_uploader("Upload image",type=["png","jpg"])

if image_file is not None:
  #st.write(image_file)
  # To View Uploaded Image
  image_data = image_file.read()
  img = Image.open(io.BytesIO(image_data))
  st.image(img,
    #width=400,
    )
  np_img = np.array([img.getdata()])#,dtype='float32')
  #st.write('Shape:',np_img.shape)
  np_img = np_img.reshape((1,28,28,1))
  #st.write('Shape:',np_img.shape)
  result = st.session_state.model.predict(np_img)
  st.write('Result', result) 
  st.write('Result Num', result.argmax()) 
