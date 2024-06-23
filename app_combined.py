import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load the trained models
model_mlp = tf.keras.models.load_model('mlp_baseline_model.h5')
model_cnn = tf.keras.models.load_model('cnn_model_with_augmentation.h5')
model_lenet = tf.keras.models.load_model('mnist_lenet_model.h5')

# Define a function to preprocess the image for MLP model
def preprocess_image_mlp(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image_array = np.array(image).astype('float32') / 255.0
    image_array = image_array.flatten()
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Define a function to preprocess the image for CNN and LeNet models
def preprocess_image_cnn(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image_array = np.array(image).astype('float32') / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

# Streamlit app
st.title('Handwritten Digit Recognition with Multiple Models')
st.write('Upload an image of a handwritten digit and the models will predict the digit.')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file)
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image for each model
    image_array_mlp = preprocess_image_mlp(image)
    image_array_cnn = preprocess_image_cnn(image)
    image_array_lenet = preprocess_image_cnn(image)
    
    # Make predictions with each model
    prediction_mlp = model_mlp.predict(image_array_mlp)
    predicted_digit_mlp = np.argmax(prediction_mlp, axis=1)[0]
    
    prediction_cnn = model_cnn.predict(image_array_cnn)
    predicted_digit_cnn = np.argmax(prediction_cnn, axis=1)[0]
    
    prediction_lenet = model_lenet.predict(image_array_lenet)
    predicted_digit_lenet = np.argmax(prediction_lenet, axis=1)[0]
    
    # Display the predictions
    st.write(f'Predicted Digit by MLP Model: {predicted_digit_mlp}')
    st.write(f'Predicted Digit by CNN Model: {predicted_digit_cnn}')
    st.write(f'Predicted Digit by LeNet Model: {predicted_digit_lenet}')
