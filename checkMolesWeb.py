# MTP UOC Lucia Manzano, 2022

import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = image[np.newaxis,...]

        prediction = model.predict(img_reshape)
        
        return prediction

model = tf.keras.models.load_model('general_model.h5')

st.write("""
         # Mind The Mole!
         """
         )
st.write("""
         ## Automatic melanoma diagnosis
         """
         )

st.write("This is the final Master's Thesis for the Universitat Oberta de Catalunya (UOC). Developed by Lucia Reyes Manzano Gomez, 2022. Please contact lucia.manzano @ uoc.edu")

st.write("Please write your phototype. If unknown, leave this field blank.")

image_photo = Image.open('phototypes.png')
st.image(image_photo, use_column_width=True)

phototype_input = st.selectbox("Select phototype: ",
                     ['Unknown', 'I', 'II', 'III', 'IV', 'V', 'VI'])

file = st.file_uploader("Please upload an image of a mole. For better results, please take a picture as close to the mole as possible with high quality.", type=["jpg", "jpeg"])
#
if file is None:
    st.write("You haven't uploaded an image file yet.")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    #image = cv2.resize(image,(224,224))     # resize image to match model's expected sizing
    #image = image.reshape(1,224,224,3)

    if phototype_input == 'I' or phototype_input == 'II' or phototype_input == 'III':
        model = tf.keras.models.load_model('specific_model_I_II_III.h5')
    elif phototype_input == 'IV' or phototype_input == 'V' or phototype_input == 'VI':
        st.write("Estimated diagnosis: MELANOMA.")
        model = tf.keras.models.load_model('specific_model_IV_V_VI.h5')
    else:
        model = model

    prediction = import_and_predict(image, model)
    
    if phototype_input == 'Unknown':
        st.write("No phototype has been selected.")
    else:
        st.write("You have selected phototype ", phototype_input)

    if prediction >= 0:
        st.success("Estimated diagnosis: BENINGNANT.")
        st.write("Your mole is benign. Please confirm this diagnosis with your doctor.")
    elif prediction < 0:
        st.error("Estimated diagnosis: MELANOMA.")
        st.write("Your mole is malignant and it could be a melanoma. Please confirm this diagnosis with your doctor.")
    else:
        st.warning("Error. Please try again or contact the administrator.")
    
    st.write("Please consider that this information is approximated.")
    st.write("This website only detect melanomas but there are more types of skin cancer.")
    st.write("Visit your doctor to do a skin checkup for confirmation.")
