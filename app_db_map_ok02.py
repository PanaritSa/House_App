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

# Database filename
db_name = "house_database.db"

# Load classification model
model_classification = load_model('best_model13cls.keras')

# Load image recognition model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model_recommendation = np.array([base_model, GlobalMaxPooling2D()])

# Load feature data
feature_list = np.array(pickle.load(open("featurevector.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))

# Load database
def load_database():
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query("SELECT * FROM houses", conn)
    conn.close()
    return df

df = load_database()

# Set page config
st.set_page_config(page_title="House Classification and Recommendation System", layout="centered")
st.image("H_vector.jpg", width=600)
st.title("House Classification and Recommendation System")
st.write("Upload an image to classify it, or search with filters.")

# Back to Home button
def back_to_home_button():
    if st.button("🏠 Back to Home"):
        st.session_state.pop("selected_house", None)
        st.session_state.pop("search_results", None)
        st.rerun()

# Show house details with map
def show_house_details(row):
    st.image(row.get("image_path"), caption=row.get("style"), width=500)
    st.write(f"**Address:** {row.get('address')}")
    st.write(f"**Price:** {row.get('price')} THB")
    st.write(f"**Bedrooms:** {row.get('bedrooms')}, **Bathrooms:** {row.get('bathrooms')}")
    st.write(f"**Area:** {row.get('area_size')} sqm")
    st.write(f"**Facilities:** {row.get('facilities')}")
    st.write(f"**Nearby Places:** {row.get('magnet')}")

    lat, lon = row.get("latitude"), row.get("longitude")
    if lat is not None and lon is not None:
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

    if st.button("Back to Search Results", key="back_button"):
        st.session_state.pop("selected_house", None)
        st.rerun()

    # ✅ ปุ่มกลับหน้าหลัก
    back_to_home_button()

# Main logic
if "selected_house" in st.session_state:
    show_house_details(st.session_state["selected_house"])

elif "search_results" in st.session_state:
    st.write(f"### 🔎 {len(st.session_state['search_results'])} houses found:")
    for i, row in enumerate(st.session_state["search_results"]):
        st.image(row["image_path"], caption=row["style"], width=300)
        if st.button(f"View Details: {row['address']}", key=f"detail_from_search_{i}"):
            st.session_state["selected_house"] = row
            st.rerun()
    # ✅ ปุ่มกลับหน้าหลัก
    back_to_home_button()

else:
    # Upload Image
    st.write("### Upload an Image for Classification and Similar House Recommendation")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
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

        predicted_style = top_styles[0][0]
        prediction_df = df[df["style"] == predicted_style]
        st.write(f"### Houses matching style: {predicted_style}")
        for i, (_, row) in enumerate(prediction_df.iterrows()):
            st.image(row["image_path"], caption=row["style"], width=300)
            if st.button(f"View Details: {row['address']}", key=f"img_detail_{i}"):
                st.session_state["selected_house"] = row.to_dict()
                st.rerun()

        # ✅ ปุ่มกลับหน้าหลัก
        back_to_home_button()

    # Search Filters
    st.write("### 🔍 Search for Houses by Filters")
    col1, col2 = st.columns(2)
    with col1:
        min_price = st.number_input("Minimum Price (THB)", value=0)
    with col2:
        max_price = st.number_input("Maximum Price (THB)", value=100000000)

    location_input = st.text_input("Location (e.g., Sukhumvit, Pattaya)")

    all_nearby_options = sorted(set(
        place.strip() for sublist in df["magnet"].dropna().str.split(",") for place in sublist
    ))
    selected_nearby = st.multiselect("Nearby Places", options=all_nearby_options)

    if st.button("Search", key="multi_filter_search"):
        filtered_df = df[
            (df["price"].astype(float) >= min_price) &
            (df["price"].astype(float) <= max_price)
        ]

        if location_input:
            filtered_df = filtered_df[filtered_df["address"].str.contains(location_input, case=False, na=False)]

        if selected_nearby:
            filtered_df = filtered_df[
                filtered_df["magnet"].apply(
                    lambda x: any(place.strip() in x for place in selected_nearby) if pd.notna(x) else False
                )
            ]

        st.session_state["search_results"] = filtered_df.to_dict(orient="records")
        st.rerun()

    # Dropdown by style
    house_styles = ["Select House Style"] + sorted(df["style"].unique().tolist())
    selected_style = st.selectbox("Select a house style:", options=house_styles)
    if selected_style and selected_style != "Select House Style":
        filtered_df = df[df["style"] == selected_style]
        st.write(f"### Houses for style: {selected_style}")
        for i, (_, row) in enumerate(filtered_df.iterrows()):
            st.image(row["image_path"], caption=row["style"], width=300)
            if st.button(f"View Details: {row['address']}", key=f"style_detail_{i}"):
                st.session_state["selected_house"] = row.to_dict()
                st.rerun()

    # ✅ ปุ่มกลับหน้าหลัก (กรณีไม่เลือกอะไรแต่เผื่อไว้)
    back_to_home_button()
