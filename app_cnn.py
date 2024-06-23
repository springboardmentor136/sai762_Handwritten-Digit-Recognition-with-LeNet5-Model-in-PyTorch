import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load the saved model
model = tf.keras.models.load_model('cnn_model_with_augmentation.h5')

# Streamlit app
st.title('Handwritten Digit Recognition')
uploaded_file = st.file_uploader("Choose an image...", type="png")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert image colors (optional, depending on the input image)
    image = image.resize((28, 28))
    image = np.array(image).astype('float32') / 255.0
    image = image.reshape(1, 28, 28, 1)
    
    st.image(image.reshape(28, 28), caption='Uploaded Image', use_column_width=True)
    
    # Predict the digit
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction, axis=1)[0]
    
    st.write(f'Predicted Digit: {predicted_digit}')