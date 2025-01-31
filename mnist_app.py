import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from joblib import load
from PIL import Image, ImageOps


# Ladda den färdigtränade modellen
@st.cache_resource
def load_model():
    return load("mnist_rf_model.pkl")

model = load_model()

# Gränssnitt
st.title("Sifferigenkänning med Random Forest by: Markus Tärning")
st.write("Ladda upp en bild på en siffra, så förutspår modellen vad det är.")

# Ladda upp bilden
uploaded_file = st.file_uploader("Ladda upp en bild i PNG/JPG-format", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Visa bilden
    image = Image.open(uploaded_file)
    st.image(image, caption="Uppladdad bild", use_column_width=True)

    # Förbehandling av bilden
    st.write("Förbehandlar bilden...")
    image = ImageOps.grayscale(image)  # Gråskala
    image = image.resize((28, 28))  # Ändra storlek
    image_array = np.array(image)
    image_array = (image_array > 128).astype(np.float32)  # Tröskelvärde
    image_array = image_array.reshape(1, -1)

    # Gör en prediktion
    prediction = model.predict(image_array)
    st.write('Image size: ', image.size)
    st.write(f"**Modellen tror att siffran är:** {prediction[0]}")
    probabilities = model.predict_proba(image_array)
    st.write(f"**Sannolikheter för varje siffra:** {probabilities[0]}")

