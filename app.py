import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import tensorflow as tf
import numpy as np


st.set_page_config(page_title="Digit Recognizer",layout="centered")
st.title("Handwritten Digit Recognizer")

col1 , col2 = st.columns(2)
with col1:
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        return_image_data=True,
        key="canvas"
    )

model = load_model("models/mnist_digit_model.keras")

def preprocess_canvas(image_data):
    image = Image.fromarray(image_data.astype('uint8')).convert('L')
    image =  image.resize((28,28))
    array = np.array(image)
    array = 255-array
    array = array.astype('float32') / 255.0
    array = array.reshape((1,784))
    return array

if st.button("Predict"):
    if canvas_result.image_data is not None:
        raw = np.array(canvas_result.image_data)
        if raw[:,:,:3].min() == 255 :
            st.write("Please draw a digit first.")
        else:
            processed = preprocess_canvas(canvas_result.image_data)
            logits = model.predict(processed)
            probabilities = tf.nn.softmax(logits).numpy()[0]
            predicted_digit = np.argmax(probabilities)
            confidence = probabilities[predicted_digit]

            with col2:
                st.write(f"### Predicted Digit: {predicted_digit}")
                st.write(f"Confidence: {confidence:.2%}")
                st.bar_chart(probabilities)

            if confidence>0.9:
                st.balloons()
    else:
        st.write("Please draw a digit first.")
