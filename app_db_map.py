
# House Finder V5 - Full Streamlit App with All Features

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
from streamlit_folium import st_folium
import folium
from geopy.distance import geodesic

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

if "favorites" not in st.session_state:
    st.session_state.favorites = []

if "swipe_index" not in st.session_state:
    st.session_state.swipe_index = 0

# === Lifestyle Mapping ===
magnet_to_lifestyle = {
    "Hospital": ["#FamilyFriendly", "#SeniorLiving", "#HealthConscious"],
    "School": ["#FamilyFriendly", "#Peaceful", "#EducationOriented"],
    "Shopping Mall": ["#UrbanLife", "#Shopaholic", "#ConvenienceSeeker"],
    "Park": ["#NatureLover", "#ActiveLifestyle", "#PetFriendly"],
    "Public Transport": ["#CarFree", "#UrbanLife", "#Traveler"],
    "University": ["#StudentLife", "#YoungProfessional", "#AcademicLifestyle"],
    "Market": ["#LocalLiving", "#Foodie", "#BudgetFriendly"],
    "Office Building": ["#Workaholic", "#TimeSaver", "#ProfessionalLifestyle"],
    "Restaurant Hub": ["#Foodie", "#NightOwl", "#Socializer"],
    "Cultural Site": ["#ArtLover", "#HistoryBuff", "#PhotographyLover"]
}

facility_to_lifestyle = {
    "Pool, Gym, Parking": ["#HealthConscious", "#FamilyFriendly", "#FitnessLover"],
    "Garden, Security": ["#NatureLover", "#SafeLiving", "#RelaxingVibes"],
    "Balcony, Kitchen": ["#WFH", "#ChefLife", "#SunlightLover"],
    "Smart Home, CCTV": ["#TechSavvy", "#SafeLiving", "#ModernLifestyle"]
}

def extract_lifestyle_tags(magnet_str, facilities_str):
    tags = set()
    if pd.notna(magnet_str):
        for m in magnet_str.split(","):
            tags.update(magnet_to_lifestyle.get(m.strip(), []))
    if pd.notna(facilities_str):
        for fac_key, fac_tags in facility_to_lifestyle.items():
            if all(f in facilities_str for f in fac_key.split(",")):
                tags.update(fac_tags)
    return list(tags)

def match_score(user_tags, house_tags):
    if not user_tags or not house_tags:
        return 0
    return len(set(user_tags) & set(house_tags)) / len(set(user_tags))

# สร้างคอลัมน์ lifestyle_tags
df["lifestyle_tags"] = df.apply(lambda row: extract_lifestyle_tags(row["magnet"], row["facilities"]), axis=1)

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

# สำหรับการคลิกบนแผนที่
if "map_click_location" not in st.session_state:
    st.session_state["map_click_location"] = None
    
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
    
    # === Estimated Mortgage Calculation ===
    price = float(row["price"])
    loan_years = 30
    annual_interest = 0.05
    monthly_interest = annual_interest / 12
    months = loan_years * 12

    if price > 0:
        monthly_payment = price * monthly_interest * (1 + monthly_interest)**months / ((1 + monthly_interest)**months - 1)
        st.write(f"**Estimated Mortgage (30 yrs @ 5%):** {monthly_payment:,.0f} THB / month")
    else:
        st.write("**Estimated Mortgage:** -")

    lat, lon = row.get("latitude"), row.get("longitude")
    if lat and lon:
        icon_data = [{
            "position": [lon, lat],
            "icon": {
                "url": "https://cdn-icons-png.flaticon.com/512/684/684908.png",
                "width": 128,
                "height": 128,
                "anchorY": 128
            }
        }]
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/streets-v11",
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=15),
            layers=[
                pdk.Layer(
                    "IconLayer",
                    data=icon_data,
                    get_icon="icon",
                    get_position="position",
                    size_scale=15,
                    pickable=True
                )
            ]
        ))
    st.subheader("\U0001F3D8 Similar Houses You May Like (Ranked)")
    similar_paths = get_recommendations(row["image_path"])
    sim_df = df[df["image_path"].isin(similar_paths)].copy()
    sim_df["similarity"] = sim_df["image_path"].apply(
        lambda path: cosine_similarity(
            feature_list[filenames.index(path)].reshape(1, -1),
            feature_list[filenames.index(row["image_path"])].reshape(1, -1)
        )[0][0] if path in filenames else 0
    )
    sim_df = sim_df.sort_values(by="similarity", ascending=False)
    for i, (_, sim_row) in enumerate(sim_df.iterrows()):
        st.image(sim_row["image_path"], width=250, caption=f"{sim_row['style']} ({sim_row['similarity']:.2f})")
        st.caption(f"{sim_row['address']} — {sim_row['price']} THB")
        if st.button(f"View Details: {sim_row['address']}", key=f"similar_result_{i}"):
            st.session_state.selected_house = sim_row.to_dict()
            st.session_state.return_page = st.session_state.page
            st.rerun()
    if st.button("\U0001F519 Back"):
        back_step()

# === Menu Navigation ===
# menu = st.selectbox("📌 Menu Navigation", ["Home", "My Lifestyle", "Favorites", "Filter", "Classify", "Style"],
#                     index=["Home", "My Lifestyle", "Favorites", "Filter", "Classify", "Style"].index(st.session_state.page),
#                     key="menu_selector")

# === Horizontal Menu with Emoji Icons ===
st.markdown("### 🧭 เมนูหลัก")

menu_items = [
    {"name": "Home", "icon": "🏠"},
    {"name": "My Lifestyle", "icon": "💖"},
    {"name": "Favorites", "icon": "🌟"},
    {"name": "Filter", "icon": "🔍"},
    {"name": "Classify", "icon": "📷"},
    {"name": "Style", "icon": "🎨"},
]

cols = st.columns(len(menu_items))
for i, item in enumerate(menu_items):
    with cols[i]:
        if st.button(f"{item['icon']}", key=f"menu_button_{item['name']}"):
            st.session_state.previous_page = st.session_state.page
            st.session_state.page = item["name"]
            st.session_state.search_results = None
            st.session_state.style_results = None
            st.session_state.classify_results = {}
            st.session_state.selected_house = None
            st.rerun()
        st.markdown(f"<div style='text-align: center;'>{item['name']}</div>", unsafe_allow_html=True)


# if menu != st.session_state.page:
if st.session_state.page != st.session_state.previous_page:
    # รีเซตสถานะที่เกี่ยวข้องกับผลลัพธ์การค้นหาทุกครั้งที่เปลี่ยนวิธี
    st.session_state.previous_page = st.session_state.page
    # st.session_state.page = menu

    # Clear previous search results when switching methods
    st.session_state.search_results = None
    st.session_state.style_results = None
    st.session_state.classify_results = {}
    st.session_state.selected_house = None

    st.rerun()

# === Routing ===
def home_button():
    if st.button("🏠 Go to Home"):
        st.session_state.page = "Home"
        st.session_state.selected_house = None
        st.rerun()

if st.session_state.selected_house:
    home_button()
    show_house_details(st.session_state.selected_house)

elif st.session_state.page == "Home":
    st.subheader("🗺 Click Anywhere to Find Houses Within 3 KM")

    # สร้างแผนที่ Folium
    m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=11)

    # เพิ่มหมุดบ้านทั้งหมด
    for _, row in df.dropna(subset=["latitude", "longitude"]).iterrows():
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=f"{row['address']}<br>{row['style']}<br>{row['price']} THB",
            icon=folium.Icon(color="blue", icon="home")
        ).add_to(m)

    # ใช้ st_folium เพื่อให้ผู้ใช้คลิก
    map_data = st_folium(m, width=700, height=500)

    if map_data and map_data["last_clicked"]:
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]
        clicked_point = (clicked_lat, clicked_lon)

        st.session_state["map_click_location"] = clicked_point
        st.success(f"📍 Clicked at: {clicked_point}")

        # คำนวณบ้านภายในรัศมี 3 กม.
        def within_radius(row, center, radius_km=3):
            if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
                return geodesic((row["latitude"], row["longitude"]), center).km <= radius_km
            return False

        nearby_df = df[df.apply(lambda row: within_radius(row, clicked_point), axis=1)]

        st.markdown("### 🏡 Houses Within 3 KM")
        if not nearby_df.empty:
            paginated = paginate_results(nearby_df, page_key="folium_radius")
            for i, (_, row) in enumerate(paginated.iterrows()):
                st.image(row["image_path"], caption=row["style"], width=300)
                if st.button(f"View Details: {row['address']}", key=f"folium_result_{i}"):
                    st.session_state.selected_house = row.to_dict()
                    st.session_state.return_page = "Home"
                    st.rerun()
        else:
            st.info("No houses found within 3 KM.")

elif st.session_state.page == "Classify":
    home_button()
    st.subheader("📷 Upload an Image for Classification")
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
        st.subheader("🎨 Top 3 Predicted Styles")
        for s, score in top_styles:
            st.write(f"✅ {s}: {score:.2f}%")
            st.session_state.classify_results[s] = df[df["style"] == s]

        similar_image_paths = get_similar_images_from_upload(image)
        sim_df = df[df["image_path"].isin(similar_image_paths)]
        st.write("## 🏡 Top 20 Similar Houses Based on Uploaded Image")
        paginated = paginate_results(sim_df, page_key="upload_similar")
        for i, (_, row) in enumerate(paginated.iterrows()):
            st.image(row["image_path"], caption=row["style"], width=300)
            if st.button(f"View Details: {row['address']}", key=f"upload_similar_{i}"):
                st.session_state.selected_house = row.to_dict()
                st.session_state.return_page = "Classify"
                st.rerun()

    for s, style_df in st.session_state.classify_results.items():
        if not style_df.empty:
            st.write(f"### 🏘 Houses in style: {s}")
            paginated = paginate_results(style_df, page_key=f"classify_page_{s}")
            for i, (_, row) in enumerate(paginated.iterrows()):
                st.image(row["image_path"], caption=row["style"], width=300)
                if st.button(f"View Details: {row['address']}", key=f"classify_{s}_{i}"):
                    st.session_state.selected_house = row.to_dict()
                    st.session_state.return_page = "Classify"
                    st.rerun()

elif st.session_state.page == "Filter":
    home_button()
    st.subheader("🔍 Search by Filter + Map + Tags")
    col1, col2 = st.columns(2)
    with col1:
        min_price = st.number_input("Min Price", 0)
    with col2:
        max_price = st.number_input("Max Price", 100000000)

    location_input = st.selectbox("Location", ["", "Sukhumvit", "Silom", "Sathorn", "Ari", "Rama 9"])

    all_facility_tags = sorted(set(tag.strip() for tags in df["facilities"].dropna().str.split(",") for tag in tags))
    selected_facility_tags = st.multiselect("🏷 Tags (Facilities)", options=all_facility_tags)

    nearby_tag_options = [
        "Hospital", "School", "Shopping Mall", "Park", "Public Transport",
        "University", "Market", "Office Building", "Restaurant Hub", "Cultural Site"
    ]
    selected_nearby_tags = st.multiselect("📍 Nearby Places", options=nearby_tag_options)

    st.map(df[["latitude", "longitude"]].dropna(), zoom=6)

    if st.button("Search", key="filter_button"):
        results = df[
            (df["price"].astype(float) >= min_price) &
            (df["price"].astype(float) <= max_price)
        ]

        if location_input:
            results = results[results["address"].str.contains(location_input, case=False, na=False)]

        if selected_facility_tags:
            results = results[
                results["facilities"].apply(
                    lambda x: any(tag.strip() in x.split(",") for tag in selected_facility_tags) if pd.notna(x) else False
                )
            ]

        if selected_nearby_tags:
            results = results[
                results["magnet"].apply(
                    lambda x: any(tag.strip() in x.split(",") for tag in selected_nearby_tags) if pd.notna(x) else False
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

    if st.button("Clear Search and Return to Home", key="clear_filter"):
        st.session_state.search_results = None
        st.session_state.page = "Home"
        st.rerun()

# elif st.session_state.page == "Style":
#     home_button()
#     st.subheader("🎨 Browse by House Style")
#     styles = ["Select"] + sorted(df["style"].unique())
#     selected_style = st.selectbox("Select a style:", styles)
#     if selected_style != "Select":
#         st.session_state.style_results = df[df["style"] == selected_style].to_dict(orient="records")

elif st.session_state.page == "Style":
    home_button()
    st.subheader("🎨 เลือกสไตล์บ้านที่คุณสนใจ")

    style_icons = {
        "ML-AR-COLONIAL": "🏛️",
        "ML-AR-Chicago School": "🏙️",
        "ML-AR-Classic": "🕍",
        "ML-AR-MEDITERRANEAN": "🌅",
        "ML-AR-MID CENTURY": "📻",
        "ML-AR-Modern": "🏢",
        "ML-AR-Modern Minimal": "📐",
        "ML-AR-Oriental": "🏯",
        "ML-AR-SCANDINAVIAN": "❄️",
        "ML-AR-THAI": "🛕",
        "ML-AR-TRANSITIONAL": "🔄",
        "ML-AR-TUDOR HOUSE": "🏡",
        "ML-AR-VICTORIAN": "👑",
    }

    available_styles = sorted(df["style"].unique())
    cols = st.columns(4)

    for i, style in enumerate(available_styles):
        icon = style_icons.get(style, "🏠")
        readable_name = style.replace("ML-AR-", "").title()
        with cols[i % 4]:
            if st.button(f"{icon}\n{readable_name}", key=f"style_{style}"):
                st.session_state.style_results = df[df["style"] == style].to_dict(orient="records")
                st.session_state.page = "Style"
                st.rerun()

    if st.session_state.style_results:
        style_df = pd.DataFrame(st.session_state.style_results)
        paginated = paginate_results(style_df, page_key="style_page")
        for i, (_, row) in enumerate(paginated.iterrows()):
            st.image(row["image_path"], caption=row["style"], width=300)
            if st.button(f"View Details: {row['address']}", key=f"style_result_{i}"):
                st.session_state.selected_house = row.to_dict()
                st.session_state.return_page = "Style"
                st.rerun()

# ✅ Routing สำหรับ Favorites
elif st.session_state.page == "Favorites":
    home_button()
    st.title("🌟 บ้านที่คุณถูกใจ (Favorites)")

    if st.session_state.favorites:
        st.write(f"❤️ คุณมีบ้านถูกใจทั้งหมด {len(st.session_state.favorites)} หลัง")
        
        for i, house in enumerate(st.session_state.favorites):
            st.image(house["image_path"], width=300)
            st.write(f"**Address:** {house['address']}")
            st.write(f"**Price:** {house['price']} THB")
            st.write(f"**Style:** {house['style']}")
            st.write(f"**Facilities:** {house['facilities']}")
            st.write(f"**Nearby:** {house['magnet']}")
            st.write(f"**Area:** {house['area_size']} sqm")

            if "lifestyle_tags" in house and house["lifestyle_tags"]:
                st.write(f"**Lifestyle Tags:** {' '.join(house['lifestyle_tags'])}")
            else:
                st.write("**Lifestyle Tags:** -")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🗑️ ลบออกจาก Favorites", key=f"remove_fav_{i}"):
                    del st.session_state.favorites[i]
                    st.rerun()
            with col2:
                if st.button("🏡 ดูบ้านที่คล้ายกัน", key=f"similar_fav_{i}"):
                    similar_paths = get_recommendations(house["image_path"])
                    sim_df = df[df["image_path"].isin(similar_paths)].copy()
                    sim_df["similarity"] = sim_df["image_path"].apply(
                        lambda path: cosine_similarity(
                            feature_list[filenames.index(path)].reshape(1, -1),
                            feature_list[filenames.index(house["image_path"])].reshape(1, -1)
                        )[0][0] if path in filenames else 0
                    )
                    sim_df = sim_df.sort_values(by="similarity", ascending=False)
                    st.markdown("### 🔁 บ้านที่คล้ายกัน")
                    for j, (_, sim_row) in enumerate(sim_df.iterrows()):
                        st.image(sim_row["image_path"], width=250, caption=f"{sim_row['style']} ({sim_row['similarity']:.2f})")
                        st.caption(f"{sim_row['address']} — {sim_row['price']} THB")
                        if st.button(f"ดูรายละเอียด: {sim_row['address']}", key=f"similar_detail_{i}_{j}"):
                            st.session_state.selected_house = sim_row.to_dict()
                            st.session_state.return_page = "Favorites"
                            st.rerun()

            # ✅ ปุ่มเข้า View Details เต็มรูปแบบ
            if st.button("🔎 View Details", key=f"fav_detail_{i}"):
                st.session_state.selected_house = house
                st.session_state.return_page = "Favorites"
                st.rerun()

    else:
        st.info("คุณยังไม่มีบ้านใน Favorites เลย ลองไปกด ❤️ ในหน้า Lifestyle ดูสิ!")
                        
# ✅ ในหน้า My Lifestyle: ปรับปุ่ม ❤️
elif st.session_state.page == "My Lifestyle":
    home_button()
    st.subheader("💖 ค้นหาบ้านที่ตรงกับไลฟ์สไตล์ของคุณ")

    # ถามแบบเลือกหลายข้อ (Checkbox)
    st.markdown("#### 🧠 คุณคิดว่าคุณเป็นคนแบบไหน?")
    personality = st.multiselect("เลือกได้มากกว่า 1 ข้อ", ["#Peaceful", "#Socializer", "#ActiveLifestyle", "#RelaxingVibes", "#FamilyFriendly"])

    st.markdown("#### 🍽️ คุณชอบอาหารแบบไหน?")
    food_pref = st.multiselect("เลือกรูปแบบอาหารที่ชอบ", ["#Foodie", "#BudgetFriendly", "#NightOwl", "#LocalLiving"])

    st.markdown("#### 🚶‍♀️ คุณเดินทางยังไง?")
    transport = st.multiselect("เลือกวิธีเดินทาง", ["#CarFree", "#UrbanLife", "#Traveler", "#TimeSaver"])

    st.markdown("#### 🏃 คุณชอบทำกิจกรรมแบบไหน?")
    activities = st.multiselect("กิจกรรมยามว่างที่คุณชอบ", ["#FitnessLover", "#NatureLover", "#PetFriendly", "#WFH", "#PhotographyLover"])

    st.markdown("#### 🛍️ คุณชอบซื้อของประเภทไหน?")
    shopping = st.multiselect("เลือกสไตล์การช้อปปิ้ง", ["#Shopaholic", "#ConvenienceSeeker", "#LocalLiving"])

    # รวม lifestyle tags จากทุกข้อ
    selected_tags = list(set(personality + food_pref + transport + activities + shopping))

    if selected_tags:
        df["match_score"] = df["lifestyle_tags"].apply(lambda tags: match_score(selected_tags, tags))
        matched_df = df[df["match_score"] > 0].sort_values(by="match_score", ascending=False)
        st.markdown(f"### 🔍 พบ {len(matched_df)} หลังที่เข้ากับไลฟ์สไตล์ของคุณ")

        if "swipe_index" not in st.session_state:
            st.session_state.swipe_index = 0

        matched_df = matched_df.reset_index(drop=True)
        if st.session_state.swipe_index < len(matched_df):
            house = matched_df.iloc[st.session_state.swipe_index]
            st.image(house["image_path"], width=500)
            st.write(f"**Address:** {house['address']}")
            st.write(f"**Price:** {house['price']} THB")
            st.write(f"**Style:** {house['style']}, Area: {house['area_size']} sqm")
            st.write(f"**Facilities:** {house['facilities']}")
            st.write(f"**Nearby:** {house['magnet']}")
            st.write(f"**Lifestyle Tags:** {' '.join(house['lifestyle_tags'])}")
            st.write(f"**Match Score:** {house['match_score']:.2f}")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("❌ ข้ามบ้านนี้"):
                    st.session_state.swipe_index += 1
                    st.rerun()
            with col2:
                if st.button("❤️ เพิ่มเข้ารายการที่ชอบ"):
                    # ป้องกันบ้านซ้ำ
                    already_favorited = any(fav["image_path"] == house["image_path"] for fav in st.session_state.favorites)
                    if not already_favorited:
                        st.session_state.favorites.append(house.to_dict())
                        st.success("✅ เพิ่มบ้านนี้เข้า Favorites แล้ว!")
                    else:
                        st.info("🏷️ บ้านนี้มีอยู่ใน Favorites แล้ว")
                    st.session_state.swipe_index += 1
                    st.rerun()

        else:
            st.info("🏁 You've swiped through all matched houses!")
    else:
        st.info("กรุณาเลือก tag ไลฟ์สไตล์ที่ตรงกับตัวคุณเพื่อเริ่มดูบ้านที่เข้ากันได้ ✨")


if st.session_state.page != "Home" and not st.session_state.selected_house:
    st.markdown("---")
    if st.button("🏠 Return to Home", key="global_home"):
        st.session_state.page = "Home"
        st.rerun()


