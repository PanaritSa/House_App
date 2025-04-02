import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import pydeck as pdk
import pickle
import os
from PIL import Image, ImageOps, ExifTags
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalMaxPooling2D

# === Load Resources ===
db_name = "house_database.db"
model_classification = load_model("best_model13cls.keras")
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
pooling_layer = GlobalMaxPooling2D()
feature_list = np.array(pickle.load(open("featurevector.pkl", "rb")))
filenames = pickle.load(open("filenames.pkl", "rb"))
filenames = [os.path.normpath(f).replace("\\", "/") for f in filenames]

# === Load Database ===
def load_database():
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query("SELECT * FROM houses", conn)
    conn.close()
    return df

df = load_database()
df["image_path"] = df["image_path"].apply(lambda x: os.path.normpath(x).replace("\\", "/"))

st.set_page_config(page_title="House Finder V5", layout="wide")
st.image("H_vector.jpg", width=600)

# === Session State Init ===
for key, default in {
    "page": "Home",
    "selected_house": None,
    "search_results": None,
    "previous_page": "Home",
    "return_page": "Home",
    "classify_results": {},
    "style_results": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# === Pagination ===
def paginate_results(df, page_key):
    size_key = f"{page_key}_size"
    page_number_key = page_key
    if size_key not in st.session_state:
        st.session_state[size_key] = 10
    if page_number_key not in st.session_state:
        st.session_state[page_number_key] = 1

    page_size = st.selectbox("Houses per page:", [5, 10, 20],
                             index=[5, 10, 20].index(st.session_state[size_key]),
                             key=f"{size_key}_select")
    if page_size != st.session_state[size_key]:
        st.session_state[size_key] = page_size
        st.session_state[page_number_key] = 1
        st.rerun()

    total_items = len(df)
    current_page = st.session_state[page_number_key]
    total_pages = (total_items - 1) // page_size + 1
    current_page = min(current_page, total_pages)
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, total_items)

    st.markdown(f"**Showing {start_idx + 1}–{end_idx} of {total_items} houses**")
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    with col1:
        if st.button("\U0001F3E0 First", key=f"{page_key}_first"):
            st.session_state[page_number_key] = 1
            st.rerun()
    with col2:
        if st.button("\u2B05\uFE0F Prev", key=f"{page_key}_prev") and current_page > 1:
            st.session_state[page_number_key] -= 1
            st.rerun()
    with col3:
        st.markdown(f"<div style='text-align: center; font-weight: bold;'>Page {current_page} of {total_pages}</div>", unsafe_allow_html=True)
    with col4:
        if st.button("\u27A1\uFE0F Next", key=f"{page_key}_next") and current_page < total_pages:
            st.session_state[page_number_key] += 1
            st.rerun()
    with col5:
        if st.button("\U0001F51A Last", key=f"{page_key}_last"):
            st.session_state[page_number_key] = total_pages
            st.rerun()
    return df.iloc[start_idx:end_idx]

# === Image Utilities ===
def correct_orientation(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation, None)
            if orientation_value == 3:
                img = img.rotate(180, expand=True)
            elif orientation_value == 6:
                img = img.rotate(270, expand=True)
            elif orientation_value == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img

def pad_to_square(img):
    desired_size = max(img.size)
    delta_w = desired_size - img.size[0]
    delta_h = desired_size - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    return ImageOps.expand(img, padding, fill='white')

# === Recommendation Functions ===
def get_recommendations(image_path):
    image_path = os.path.normpath(image_path).replace("\\", "/")
    try:
        index = filenames.index(image_path)
    except ValueError:
        return []
    query_feature = feature_list[index].reshape(1, -1)
    similarities = cosine_similarity(feature_list, query_feature).flatten()
    indices = np.argsort(similarities)[-6:-1][::-1]
    return [filenames[i] for i in indices]

def get_similar_images_from_upload(pil_image):
    image = pil_image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature_map = base_model.predict(img_array)
    features = pooling_layer(feature_map).numpy()
    similarities = cosine_similarity(feature_list, features).flatten()
    indices = np.argsort(similarities)[-20:][::-1]
    return [filenames[i] for i in indices]

# === Back and Detail View ===
def back_step():
    st.session_state.page = st.session_state.return_page
    st.session_state.selected_house = None
    st.rerun()

def show_house_details(row):
    st.image(row["image_path"], caption=row["style"], width=500)
    st.write(f"**Address:** {row['address']}")
    st.write(f"**Price:** {row['price']} THB")
    st.write(f"**Bedrooms:** {row['bedrooms']}, **Bathrooms:** {row['bathrooms']}")
    st.write(f"**Area:** {row['area_size']} sqm")
    st.write(f"**Facilities:** {row['facilities']}")
    st.write(f"**Nearby Places:** {row['magnet']}")
    lat, lon = row.get("latitude"), row.get("longitude")
    if lat and lon:
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/streets-v11",
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=15),
            layers=[pdk.Layer("ScatterplotLayer", data=[{"lat": lat, "lon": lon}],
                              get_position="[lon, lat]", get_color="[255, 0, 0, 160]", get_radius=100)]
        ))
    st.subheader("\U0001F3D8 Similar Houses You May Like")
    similar_paths = get_recommendations(row["image_path"])
    sim_df = df[df["image_path"].isin(similar_paths)]
    for _, sim_row in sim_df.iterrows():
        st.image(sim_row["image_path"], width=250, caption=sim_row["style"])
        st.caption(f"{sim_row['address']} — {sim_row['price']} THB")
    if st.button("\U0001F519 Back"):
        back_step()

# === Menu Navigation ===
menu = st.radio("\U0001F4CC Menu", ["Home", "Classify", "Filter", "Style"], horizontal=True)
if menu != st.session_state.page:
    st.session_state.previous_page = st.session_state.page
    st.session_state.page = menu

# === Routing ===
if st.session_state.selected_house:
    show_house_details(st.session_state.selected_house)

elif st.session_state.page == "Home":
    st.write("Welcome to the House Finder App V5!")

elif st.session_state.page == "Classify":
    st.subheader("\U0001F4F7 Upload an Image for Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        image = correct_orientation(image)
        image = pad_to_square(image).resize((224, 224))

        img_array = img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        pred = model_classification.predict(img_array)[0]

        style_labels = df["style"].unique()
        top_idx = np.argsort(pred)[-3:][::-1]
        top_styles = [(style_labels[i], pred[i] * 100) for i in top_idx]

        st.image(image, width=300)
        st.subheader("\U0001F3AF Top 3 Predicted Styles")
        for s, score in top_styles:
            st.write(f"✅ {s}: {score:.2f}%")
            st.session_state.classify_results[s] = df[df["style"] == s]

        similar_image_paths = get_similar_images_from_upload(image)
        sim_df = df[df["image_path"].isin(similar_image_paths)]
        st.write("## \U0001F3D8 Top 20 Similar Houses Based on Uploaded Image")
        paginated = paginate_results(sim_df, page_key="upload_similar")
        for i, (_, row) in enumerate(paginated.iterrows()):
            st.image(row["image_path"], caption=row["style"], width=300)
            if st.button(f"View Details: {row['address']}", key=f"upload_similar_{i}"):
                st.session_state.selected_house = row.to_dict()
                st.session_state.return_page = "Classify"
                st.rerun()
    for s, style_df in st.session_state.classify_results.items():
        if not style_df.empty:
            st.write(f"### \U0001F3E1 Houses in style: {s}")
            paginated = paginate_results(style_df, page_key=f"classify_page_{s}")
            for i, (_, row) in enumerate(paginated.iterrows()):
                st.image(row["image_path"], caption=row["style"], width=300)
                if st.button(f"View Details: {row['address']}", key=f"classify_{s}_{i}"):
                    st.session_state.selected_house = row.to_dict()
                    st.session_state.return_page = "Classify"
                    st.rerun()

elif st.session_state.page == "Filter":
    st.subheader("\U0001F50D Search by Filter + Map + Tags")
    col1, col2 = st.columns(2)
    with col1:
        min_price = st.number_input("Min Price", 0)
    with col2:
        max_price = st.number_input("Max Price", 100000000)
    location_input = st.text_input("Location (e.g., Sukhumvit, Pattaya)")
    all_tags = sorted(set(tag.strip() for tags in df["facilities"].dropna().str.split(",") for tag in tags))
    selected_tags = st.multiselect("\U0001F3F7 Tags (Facilities)", options=all_tags)
    st.map(df[["latitude", "longitude"]].dropna(), zoom=6)
    if st.button("Search", key="filter_button"):
        results = df[
            (df["price"].astype(float) >= min_price) &
            (df["price"].astype(float) <= max_price)
        ]
        if location_input:
            results = results[results["address"].str.contains(location_input, case=False, na=False)]
        if selected_tags:
            results = results[
                results["facilities"].apply(
                    lambda x: any(tag in x for tag in selected_tags) if pd.notna(x) else False
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
    st.subheader("\U0001F3A8 Browse by House Style")
    styles = ["Select"] + sorted(df["style"].unique())
    selected_style = st.selectbox("Select a style:", styles)
    if selected_style != "Select":
        st.session_state.style_results = df[df["style"] == selected_style].to_dict(orient="records")
    if st.session_state.style_results:
        style_df = pd.DataFrame(st.session_state.style_results)
        paginated = paginate_results(style_df, page_key="style_page")
        for i, (_, row) in enumerate(paginated.iterrows()):
            st.image(row["image_path"], caption=row["style"], width=300)
            if st.button(f"View Details: {row['address']}", key=f"style_result_{i}"):
                st.session_state.selected_house = row.to_dict()
                st.session_state.return_page = "Style"
                st.rerun()