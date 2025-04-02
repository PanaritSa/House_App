import streamlit as st
import sqlite3
import pandas as pd
import os
import cv2
import numpy as np
import pydeck as pdk
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
import pickle
from numpy.linalg import norm

# Database filename
db_name = "house_database.db"

# Load classification model
model_classification = load_model('best_model13cls.keras')

# Load image recognition model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model_recommendation = np.array([base_model, GlobalMaxPooling2D()])

# Load feature data for image recognition
feature_list = np.array(pickle.load(open("featurevector.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))

# Load database
def load_database():
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query("SELECT * FROM houses", conn)
    conn.close()
    return df

# Load data
df = load_database()

# Streamlit UI
st.set_page_config(page_title="House Classification and Recommendation System", layout="centered")
st.image("H_vector.jpg", width=600)
st.title("House Classification and Recommendation System")
st.write("Upload an image to classify it, or select a house style from the dropdown to view related houses.")

# Function to show house details with map
def show_house_details(row):
    st.image(row["image_path"], caption=row["style"], width=500)
    st.write(f"**Address:** {row['address']}")
    st.write(f"**Price:** {row['price']} THB")
    st.write(f"**Bedrooms:** {row['bedrooms']}, **Bathrooms:** {row['bathrooms']}")
    st.write(f"**Area:** {row['area_size']} sqm")
    st.write(f"**Facilities:** {row['facilities']}")
    st.write(f"**Nearby Places:** {row['magnet']}")
    
    # Show Map using PyDeck
    lat, lon = row["latitude"], row["longitude"]
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/streets-v11",
        initial_view_state=pdk.ViewState(
            latitude=lat,
            longitude=lon,
            zoom=15,
            pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame([{"lat": lat, "lon": lon}]),
                get_position="[lon, lat]",
                get_color="[255, 0, 0, 160]",
                get_radius=100,
            )
        ],
    ))
    
    if st.button("Back to Search Results", key=f"back_{row['image_path']}"):
        st.session_state.pop("selected_house", None)
        st.rerun()

# Handle page navigation
if "selected_house" in st.session_state:
    show_house_details(st.session_state["selected_house"])
else:
    # Image Upload for Classification and Recognition
    st.write("### Upload an Image for Classification and Similar House Recommendation")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Classification
        image = load_img(uploaded_file, target_size=(224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        pred_probabilities = model_classification.predict(img_array)[0]
        top_indices = np.argsort(pred_probabilities)[-3:][::-1]
        top_styles = [(df["style"].unique()[i], pred_probabilities[i] * 100) for i in top_indices]
        
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        st.write("### Top 3 Predicted Styles")
        for style, score in top_styles:
            st.write(f"{style}: {score:.2f}%")
        
        # Display matching houses for the top predicted style
        predicted_style = top_styles[0][0]
        prediction_df = df[df["style"] == predicted_style]
        st.write(f"### Houses matching style: {predicted_style}")
        for i, (_, row) in enumerate(prediction_df.iterrows()):
            st.image(row["image_path"], caption=row["style"], width=300)
            if st.button(f"View Details: {row['address']}", key=f"details_{row['image_path']}_{i}"):
                st.session_state["selected_house"] = row
                st.rerun()

    # Retain search functionalities
    house_styles = ["Select House Style"] + sorted(df["style"].unique().tolist())
    selected_style = st.selectbox("Select a house style:", options=house_styles)
    if selected_style and selected_style != "Select House Style":
        filtered_df = df[df["style"] == selected_style]
        st.write(f"### Houses for style: {selected_style}")
        for i, (_, row) in enumerate(filtered_df.iterrows()):
            st.image(row["image_path"], caption=row["style"], width=300)
            if st.button(f"View Details: {row['address']}", key=f"details_{row['image_path']}_{i}"):
                st.session_state["selected_house"] = row
                st.rerun()
