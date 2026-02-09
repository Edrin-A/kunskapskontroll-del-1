import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Ladda modellen (cachas så den inte laddas om varje gång)
@st.cache_resource
def load_model():
    return ResNet50(weights='imagenet')

model = load_model()

# Sidtitel
st.title("Bildklassificering med ResNet50")
st.write("Ladda upp en bild så predikterar AI:n vad den föreställer!")

# Filuppladdning
uploaded_file = st.file_uploader("Välj en bild...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Visa uppladdad bild
    img = Image.open(uploaded_file)
    st.image(img, caption="Din uppladdade bild", use_container_width=True)

    # Förbered bilden för prediktion
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Gör prediktion
    with st.spinner("Analyserar bilden..."):
        preds = model.predict(img_array)
        results = decode_predictions(preds, top=5)[0]

    # Visa resultat
    st.subheader("Resultat:")
    for i, (class_id, name, probability) in enumerate(results, 1):
        st.write(f"{i}. **{name}** - {probability*100:.2f}%")
        st.progress(float(probability))
