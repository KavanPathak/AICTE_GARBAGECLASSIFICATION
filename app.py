import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# Page Config
st.set_page_config(
    page_title="Smart Garbage Classifier",
    page_icon="🧠",
    layout="wide"
)

# Custom Styles
st.markdown("""
    <style>
        .stApp {
            background: radial-gradient(circle, #111 0%, #000000 100%);
            color: white;
        }
        h1, h2, h3, h4, h5, h6, span, p, label, div {
            color: white !important;
        }
        .stSidebar {
            background-color: #0c0c0c !important;
            border-right: 3px solid hotpink;
        }
        .stRadio > div {
            background-color: #222;
            padding: 10px;
            border-radius: 10px;
            color: white;
            border: 1px solid deeppink;
        }
        .stButton > button {
            background-color: deeppink;
            color: white;
            border-radius: 8px;
            padding: 0.4em 1em;
            font-weight: bold;
            border: 2px solid white;
        }
        .stFileUploader label {
            color: pink;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Load model
model_path = "model/garbage_classifier_model.keras"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error("Model file not found. Please save it at 'model/garbage_classifier_model.keras'.")
    st.stop()

categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Content", "Image Prediction", "Webcam Capture", "About Project"])

# Home Page
if page == "Home":
    st.title("\U0001F9E0 Smart Garbage Classifier")
    st.markdown("""
    ## ✨ Welcome!
    This is a smart AI-based garbage classification tool that helps you predict the type of waste from an image.

    #### ♻️ Aim:
    To make waste sorting easier, faster, and more efficient using deep learning technology.

    #### 🌍 Why It Matters:
    - Reduces pollution
    - Improves recycling
    - Keeps environment clean
    - Helps smart cities and sustainable systems

    #### 💡 How to Use:
    - Upload a clear image of a garbage item
    - Or use your webcam for real-time capture
    - The model will tell you what type it is: cardboard, glass, metal, paper, plastic, or trash

    > "Waste isn't waste until we waste it."
    """)

# Content Page
elif page == "Content":
    st.title("\U0001F4D8 Learn About Garbage")
    st.markdown("""
    ### 🧾 What is Garbage?
    Garbage refers to unwanted or discarded items we no longer need. It can be food waste, packaging, bottles, newspapers, or even electronics.

    ---

    ### ♻️ Why Classification is Important:
    - Saves valuable materials from ending up in landfills
    - Prevents pollution and diseases
    - Reduces waste management costs
    - Increases public awareness of sustainability

    ---

    ### 🧠 Categories Used in This App:
    | Garbage Type | Common Examples                  |
    |--------------|-----------------------------------|
    | Cardboard    | Cartons, delivery boxes           |
    | Glass        | Bottles, jars                     |
    | Metal        | Cans, foils                       |
    | Paper        | Books, newspapers                 |
    | Plastic      | Bags, food containers             |
    | Trash        | Non-recyclables, mixed waste      |

    ---

    ### 📈 Source & Facts:
    - Based on waste management data and recycling practices
    - Inspired by global efforts to reduce carbon footprint
    - Uses AI and image recognition to help people sort better
    """)

# Prediction Page
elif page == "Image Prediction":
    st.title("\U0001F5BC️ Upload & Predict")
    uploaded_file = st.file_uploader("Upload an image of garbage", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = categories[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.success(f"Predicted Class: `{predicted_class}`")
        st.info(f"Confidence: {confidence:.2f}%")

        # Plot
        fig, ax = plt.subplots()
        ax.bar(categories, prediction[0], color='deeppink')
        ax.set_ylabel('Confidence')
        ax.set_ylim([0, 1])
        ax.set_title('Prediction Probability')
        st.pyplot(fig)

        st.markdown("""
        #### 🔍 Tips:
        - Use good lighting
        - Avoid cluttered backgrounds
        - Try capturing close-up shots
        """)

# Webcam Capture Page
elif page == "Webcam Capture":
    st.title("📹 Capture via Webcam")
    st.warning("Webcam feature is available only when run via localhost.")
    run = st.button('Capture and Predict')

    if run:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(img, caption='Captured Image', use_column_width=True)

            # Resize and predict
            img_resized = cv2.resize(img, (224, 224)) / 255.0
            img_array = np.expand_dims(img_resized, axis=0)
            prediction = model.predict(img_array)
            predicted_class = categories[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            st.success(f"Predicted Class: `{predicted_class}`")
            st.info(f"Confidence: {confidence:.2f}%")

            fig, ax = plt.subplots()
            ax.bar(categories, prediction[0], color='deeppink')
            ax.set_ylabel('Confidence')
            ax.set_ylim([0, 1])
            ax.set_title('Prediction Probability')
            st.pyplot(fig)

        cap.release()

# About Project Page
elif page == "About Project":
    st.title("🌟 Project Details")
    st.markdown("""
    ### 🚀 Purpose
    To help people and communities classify garbage automatically using computer vision. This project is aligned with the global goal of improving waste sorting and recycling habits.

    ### 📊 Real-World Applications
    - Smart bins in urban areas
    - School & college awareness apps
    - Environmental conservation campaigns
    - Automated waste sorting in industries

    ### 🧠 Future Scope
    - Real-time mobile app with live camera
    - Integration with waste pickup apps
    - City-level dashboard for classification stats

    ### 📚 Credits
    - Dataset: TrashNet
    - Model: EfficientNetV2B2
    - Built with: TensorFlow, Streamlit

    > Clean planet, smart tech — together for a better tomorrow.
    """)

