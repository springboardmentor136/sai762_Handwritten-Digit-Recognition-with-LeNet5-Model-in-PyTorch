import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('mlp_baseline_model.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    # Convert the image to grayscale
    image = ImageOps.grayscale(image)
    # Invert the image (to make the background black and digit white)
    image = ImageOps.invert(image)
    # Resize the image to 28x28 pixels
    image = image.resize((28, 28))
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Normalize the pixel values to [0, 1]
    image_array = image_array.astype('float32') / 255.0
    # Flatten the array
    image_array = image_array.flatten()
    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Streamlit app
st.title('MNIST Digit Recognizer')
st.write('Upload an image of a handwritten digit and the model will predict the digit.')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    image_array = preprocess_image(image)
    
    # Make a prediction
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction, axis=1)[0]
    
    # Display the prediction
    st.write(f'Predicted Digit: {predicted_digit}')
