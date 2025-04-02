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

# === Load Database ===
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
if "return_page" not in st.session_state:
    st.session_state.return_page = "Home"
if "classify_results" not in st.session_state:
    st.session_state.classify_results = {}
if "style_results" not in st.session_state:
    st.session_state.style_results = None

# === Pagination Function ===
def paginate_results(df, page_key):
    size_key = f"{page_key}_size"
    page_number_key = page_key

    if size_key not in st.session_state:
        st.session_state[size_key] = 10
    if page_number_key not in st.session_state:
        st.session_state[page_number_key] = 1

    new_page_size = st.selectbox(
        "Houses per page:",
        options=[5, 10, 20],
        index=[5, 10, 20].index(st.session_state[size_key]),
        key=f"{size_key}_select"
    )

    if new_page_size != st.session_state[size_key]:
        st.session_state[size_key] = new_page_size
        st.session_state[page_number_key] = 1
        st.rerun()

    page_size = st.session_state[size_key]
    current_page = st.session_state[page_number_key]
    total_items = len(df)
    total_pages = (total_items - 1) // page_size + 1

    if current_page > total_pages:
        st.session_state[page_number_key] = total_pages
        current_page = total_pages

    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_items)

    st.markdown(f"**Showing {start_idx + 1}–{end_idx} of {total_items} houses**")

    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    with col1:
        if st.button("🏠 First", key=f"{page_key}_first"):
            st.session_state[page_number_key] = 1
            st.rerun()
    with col2:
        if st.button("⬅️ Prev", key=f"{page_key}_prev") and current_page > 1:
            st.session_state[page_number_key] -= 1
            st.rerun()
    with col3:
        st.markdown(
            f"<div style='text-align: center; font-weight: bold;'>Page {current_page} of {total_pages}</div>",
            unsafe_allow_html=True
        )
    with col4:
        if st.button("➡️ Next", key=f"{page_key}_next") and current_page < total_pages:
            st.session_state[page_number_key] += 1
            st.rerun()
    with col5:
        if st.button("🔚 Last", key=f"{page_key}_last"):
            st.session_state[page_number_key] = total_pages
            st.rerun()

    return df.iloc[start_idx:end_idx]

# === Top Menu ===
menu = st.radio("📌 Menu", ["Home", "Classify", "Filter", "Style"], horizontal=True)
if menu != st.session_state.page:
    st.session_state.previous_page = st.session_state.page
    st.session_state.page = menu

# === Back Button ===
def back_step():
    st.session_state.page = st.session_state.return_page
    st.session_state.selected_house = None
    st.rerun()

# === House Detail View ===
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

# === Page Handling ===
if st.session_state.selected_house:
    show_house_details(st.session_state.selected_house)

elif st.session_state.page == "Home":
    st.write("Welcome to the House Classification and Recommendation System!")
    st.write("Choose a menu above to start.")

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
        style_labels = df["style"].unique()
        top_styles = [(style_labels[i], pred[i] * 100) for i in top_idx]

        st.image(uploaded_file, caption="Uploaded Image", width=300)
        st.write("### 🎯 Top 3 Predicted Styles")
        for s, score in top_styles:
            st.write(f"{s}: {score:.2f}%")

        st.session_state.classify_results = {s: df[df["style"] == s] for s, _ in top_styles}

    for s, style_df in st.session_state.classify_results.items():
        if not style_df.empty:
            st.write(f"### 🏡 Houses in style: {s}")
            paginated = paginate_results(style_df, page_key=f"classify_page_{s}")
            for i, (_, row) in enumerate(paginated.iterrows()):
                st.image(row["image_path"], caption=row["style"], width=300)
                if st.button(f"View Details: {row['address']}", key=f"classify_{s}_{i}"):
                    st.session_state.selected_house = row.to_dict()
                    st.session_state.return_page = "Classify"
                    st.rerun()

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
        results_df = pd.DataFrame(st.session_state.search_results)
        st.write(f"### Found {len(results_df)} results:")
        paginated = paginate_results(results_df, page_key="filter_page")
        for i, (_, row) in enumerate(paginated.iterrows()):
            st.image(row["image_path"], caption=row["style"], width=300)
            if st.button(f"View Details: {row['address']}", key=f"filter_result_{i}"):
                st.session_state.selected_house = row.to_dict()
                st.session_state.return_page = "Filter"
                st.rerun()

elif st.session_state.page == "Style":
    st.write("### 🎨 Choose a House Style")
    styles = ["Select"] + sorted(df["style"].unique().tolist())
    selected_style = st.selectbox("Select a style:", styles)
    if selected_style and selected_style != "Select":
        style_df = df[df["style"] == selected_style]
        st.session_state.style_results = style_df.to_dict(orient="records")
        st.write(f"### 🏡 Houses with style: {selected_style}")

    if st.session_state.style_results:
        style_df = pd.DataFrame(st.session_state.style_results)
        paginated = paginate_results(style_df, page_key="style_page")
        for i, (_, row) in enumerate(paginated.iterrows()):
            st.image(row["image_path"], caption=row["style"], width=300)
            if st.button(f"View Details: {row['address']}", key=f"style_result_{i}"):
                st.session_state.selected_house = row.to_dict()
                st.session_state.return_page = "Style"
                st.rerun()
