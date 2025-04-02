import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import pydeck as pdk
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalMaxPooling2D
import pickle

# === Load Resources ===
db_name = "house_database.db"
model_classification = load_model('best_model13cls.keras')
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model_recommendation = np.array([base_model, GlobalMaxPooling2D()])
feature_list = np.array(pickle.load(open("featurevector.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))


def load_database():
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query("SELECT * FROM houses", conn)
    conn.close()
    return df


df = load_database()
st.set_page_config(page_title="House Classification and Recommendation System", layout="centered")
st.image("H_vector.jpg", width=600)

# === Session State Init ===
if "page" not in st.session_state:
    st.session_state.page = "Home"
if "selected_house" not in st.session_state:
    st.session_state.selected_house = None
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "previous_page" not in st.session_state:
    st.session_state.previous_page = "Home"

# === Top Menu ===
menu = st.radio("📌 Menu", ["Home", "Classify", "Filter", "Style"], horizontal=True)
if menu != st.session_state.page:
    st.session_state.previous_page = st.session_state.page
    st.session_state.page = menu

# === Back Step Button ===
def back_step():
    st.session_state.page, st.session_state.previous_page = st.session_state.previous_page, st.session_state.page
    st.session_state.selected_house = None
    st.rerun()

# === Show House Details ===
def show_house_details(row):
    st.image(row.get("image_path"), caption=row.get("style"), width=500)
    st.write(f"**Address:** {row.get('address')}")
    st.write(f"**Price:** {row.get('price')} THB")
    st.write(f"**Bedrooms:** {row.get('bedrooms')}, **Bathrooms:** {row.get('bathrooms')}")
    st.write(f"**Area:** {row.get('area_size')} sqm")
    st.write(f"**Facilities:** {row.get('facilities')}")
    st.write(f"**Nearby Places:** {row.get('magnet')}")
    lat, lon = row.get("latitude"), row.get("longitude")
    if lat and lon:
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/streets-v11",
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=15),
            layers=[
                pdk.Layer("ScatterplotLayer",
                          data=pd.DataFrame([{"lat": lat, "lon": lon}]),
                          get_position="[lon, lat]",
                          get_color="[255, 0, 0, 160]",
                          get_radius=100)
            ]
        ))

    if st.button("🔙 Back"):
        back_step()

# === View Details Page ===
if st.session_state.selected_house:
    show_house_details(st.session_state.selected_house)

# === Home Page ===
elif st.session_state.page == "Home":
    st.write("Welcome to the House Classification and Recommendation System!")
    st.write("Choose a menu above to start.")

# === Classify Page ===
elif st.session_state.page == "Classify":
    st.write("### 📷 Upload an Image for Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = load_img(uploaded_file, target_size=(224, 224))
        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        pred = model_classification.predict(img_array)[0]
        top_idx = np.argsort(pred)[-3:][::-1]
        top_styles = [(df["style"].unique()[i], pred[i]*100) for i in top_idx]
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        st.write("### 🎯 Top 3 Predicted Styles")
        for s, score in top_styles:
            st.write(f"{s}: {score:.2f}%")

        selected_style = top_styles[0][0]
        results = df[df["style"] == selected_style]
        st.write(f"### 🏡 Houses matching style: {selected_style}")
        for i, (_, row) in enumerate(results.iterrows()):
            st.image(row["image_path"], caption=row["style"], width=300)
            if st.button(f"View Details: {row['address']}", key=f"classify_{i}"):
                st.session_state.selected_house = row.to_dict()
                st.session_state.previous_page = "Classify"
                st.rerun()

# === Filter Page ===
elif st.session_state.page == "Filter":
    st.write("### 🔍 Search for Houses by Filters")
    col1, col2 = st.columns(2)
    with col1:
        min_price = st.number_input("Minimum Price (THB)", value=0)
    with col2:
        max_price = st.number_input("Maximum Price (THB)", value=100000000)
    location_input = st.text_input("Location (e.g., Sukhumvit, Pattaya)")
    all_nearby = sorted(set(
        place.strip() for sublist in df["magnet"].dropna().str.split(",") for place in sublist
    ))
    selected_nearby = st.multiselect("Nearby Places", options=all_nearby)

    if st.button("Search", key="filter_button"):
        results = df[
            (df["price"].astype(float) >= min_price) &
            (df["price"].astype(float) <= max_price)
        ]
        if location_input:
            results = results[results["address"].str.contains(location_input, case=False, na=False)]
        if selected_nearby:
            results = results[
                results["magnet"].apply(
                    lambda x: any(n in x for n in selected_nearby) if pd.notna(x) else False
                )
            ]
        st.session_state.search_results = results.to_dict(orient="records")
        st.session_state.previous_page = "Filter"
        st.rerun()

    if st.session_state.search_results:
        st.write(f"### Found {len(st.session_state.search_results)} results:")
        for i, row in enumerate(st.session_state.search_results):
            st.image(row["image_path"], caption=row["style"], width=300)
            if st.button(f"View Details: {row['address']}", key=f"filter_result_{i}"):
                st.session_state.selected_house = row
                st.session_state.previous_page = "Filter"
                st.rerun()

# === Style Page ===
elif st.session_state.page == "Style":
    st.write("### 🎨 Choose a House Style")
    styles = ["Select"] + sorted(df["style"].unique().tolist())
    selected_style = st.selectbox("Select a style:", styles)
    if selected_style and selected_style != "Select":
        style_df = df[df["style"] == selected_style]
        st.write(f"### 🏡 Houses with style: {selected_style}")
        for i, (_, row) in enumerate(style_df.iterrows()):
            st.image(row["image_path"], caption=row["style"], width=300)
            if st.button(f"View Details: {row['address']}", key=f"style_result_{i}"):
                st.session_state.selected_house = row.to_dict()
                st.session_state.previous_page = "Style"
                st.rerun()
