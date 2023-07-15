import streamlit as st
import tensorflow as tf
import requests
import os
from streamlit_lottie import st_lottie
from PIL import Image, ImageOps
import numpy as np
import openai

#openai api
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPEN_AI_KEY')
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]



st.set_page_config(page_title="Flower Guide", page_icon="🌼",layout='wide')

@st.cache_resource
def load_model():
  model=tf.keras.models.load_model('f_model.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()
##Header Section
st.markdown(
    """
    <div style='text-align: center;'>
        <h1>Welcome to Flower Guide Shop</h1>
    </div>
    """,
    unsafe_allow_html=True
)

##lottie
def load_lottieurl(url):
    r= requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()
##LOAD GIF
lotties1= load_lottieurl("https://lottie.host/edbc3a2b-4316-46cc-ba6c-c661ab5a93c4/eXIG0ApB3D.json")


def import_and_predict(image_data, model):
      class_names = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']
      size = (224,224)
      image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
      new_image = tf.keras.preprocessing.image.img_to_array(image)


      # Expand dimensions to match the batch size
      new_image = tf.expand_dims(new_image, axis=0)

      # Make the prediction
      predictions = model.predict(new_image)

      # Get the predicted class label
      predicted_class = tf.argmax(predictions[0]).numpy()
      predicted_name = class_names[predicted_class]
      class_scores = np.max(predictions[0])
      
      return predicted_name,class_scores

with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        file = st.file_uploader("Please upload an image", type=["jpg", "png"])
        if st.button('Submit'):
          if file is None:
              st.error("Please upload an image file!")
          else:
            st.success('Upload Success!', icon="✅")
            with st.spinner('Retrieving Details...'):
              image = Image.open(file)
              st.image(image, use_column_width=True)
              prediction,score = import_and_predict(image, model)
              disp=f"This image most likely belongs to {prediction} with a {100 * score :.2f}% confidence."
              st.write(disp)
              prompt=f"""Give me a perfect guide of keywords only for the following items from the flower_name:
              - Scientific name
              - Sun needs 
              - Soil needs
              - Blooms in
              - Features
              Make sure the headings are in bold letters.
              flower_name : '''{prediction}'''
                    """
              response = get_completion(prompt)
              st.markdown(response)
    with col2:
       st_lottie(lotties1,height=300, key="coding")