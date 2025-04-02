import streamlit as st
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pickle
import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load database connection
def get_db_connection():
    return sqlite3.connect("house_database.db")

# Load house data from the database
def load_house_data():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM houses", conn)
    conn.close()
    return df

# Load models
@st.cache_resource
def initialize_classification_model():
    return load_model('best_model13cls.keras')

@st.cache_resource
def initialize_recommendation_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    return tf.keras.Sequential([base_model, GlobalMaxPooling2D()])

# Load feature vector and filenames
def load_feature_data():
    try:
        feature_list = np.array(pickle.load(open("featurevector.pkl", "rb")))
        filenames = pickle.load(open("filenames.pkl", "rb"))
        return feature_list, filenames
    except Exception as e:
        st.error(f"Error loading feature vector: {e}")
        st.stop()

# Initialize models and data
model_classification = initialize_classification_model()
model_recommendation = initialize_recommendation_model()
feature_list, filenames = load_feature_data()
st.write(f"Total feature vectors loaded: {len(feature_list)}")
st.write(f"Total filenames loaded: {len(filenames)}")
house_data = load_house_data()

# Streamlit App
st.title("🏠 House Recommendation & Classification System")

# Upload an image for classification & recommendation
uploaded_file = st.file_uploader("📸 Upload a house image for classification & recommendation", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_cv is not None:
        img_cv = cv2.resize(img_cv, (224, 224))
        st.image(img_cv, caption='Uploaded Image', use_column_width=True)
    else:
        st.error("⚠️ Error: Unable to read image. Please upload a valid image file.")
        st.stop()
    
    # Classification
    st.write("### 🏡 Classification Results")
    img_array = preprocess_input(np.expand_dims(img_cv, axis=0))
    pred = model_classification.predict(img_array)
    pred_probabilities = pred[0]
    top_indices = pred_probabilities.argsort()[-3:][::-1]
    top_labels = [(house_data['style'].unique()[i], pred_probabilities[i]) for i in top_indices]
    
    fig, ax = plt.subplots()
    ax.barh([label for label, _ in top_labels], [score * 100 for _, score in top_labels], color='skyblue')
    ax.set_title("🏆 Top 3 Predictions")
    st.pyplot(fig)
    
    # Recommendation
    st.write("### 🔍 Recommended Houses")
    feature_vector = model_recommendation.predict(np.expand_dims(img_cv, axis=0)).flatten()
    st.write(f"Feature vector shape: {feature_vector.shape}")
    feature_vector = feature_vector / norm(feature_vector)
    
    neighbors = NearestNeighbors(n_neighbors=10, metric='euclidean')
    neighbors.fit(feature_list)
    _, indices = neighbors.kneighbors([feature_vector])
    st.write(f"Indices returned by NearestNeighbors: {indices}")
    
    for i in indices[0][:10]: # Limit results to 10
        image_path = filenames[i]
        st.write(f"Checking for image path: {image_path}")
        matching_house = house_data[house_data['image_path'] == image_path]
        if matching_house.empty:
                st.write("⚠️ No matching house found in database for this image!")
        if not matching_house.empty:
            row = matching_house.iloc[0]
            st.image(row['image_path'], caption=f"{row['style']} - {row['price']:,} THB", use_column_width=True)
            st.write(f"🏠 {row['bedrooms']} Bedrooms | 🛁 {row['bathrooms']} Bathrooms | 📏 {row['area_size']} sq.m.")
            st.write(f"🛠 Facilities: {row['facilities']}")
            st.write("---")

# User input: Select filters
st.write("### 🏠 Filter Houses")
style_filter = st.selectbox("🏡 Select House Style:", ["All"] + sorted(house_data['style'].unique().tolist()))
price_range = st.slider("💰 Price Range (Million THB)", 1, 50, (1, 50))
bedrooms_filter = st.slider("🛏 Bedrooms", 1, 6, (1, 6))
bathrooms_filter = st.slider("🛁 Bathrooms", 1, 5, (1, 5))

if st.button("🔎 Search Houses"):
    # Apply filters
    def filter_houses(df, style, price, bedrooms, bathrooms):
        filtered_df = df[(df['price'] >= price[0] * 1_000_000) & (df['price'] <= price[1] * 1_000_000)]
        filtered_df = filtered_df[(filtered_df['bedrooms'] >= bedrooms[0]) & (filtered_df['bedrooms'] <= bedrooms[1])]
        filtered_df = filtered_df[(filtered_df['bathrooms'] >= bathrooms[0]) & (filtered_df['bathrooms'] <= bathrooms[1])]
        if style != "All":
            filtered_df = filtered_df[filtered_df['style'] == style]
        return filtered_df

    filtered_houses = filter_houses(house_data, style_filter, price_range, bedrooms_filter, bathrooms_filter)

    # Pagination setup
    results_per_page = 10
    total_pages = max(1, (len(filtered_houses) - 1) // results_per_page + 1)
    page = st.number_input("📄 Page", min_value=1, max_value=total_pages, step=1, value=1)
    start_idx = (page - 1) * results_per_page
    end_idx = start_idx + results_per_page
    paginated_houses = filtered_houses.iloc[start_idx:end_idx]

    st.write(f"### 🏡 Found {len(filtered_houses)} matching houses")
    for _, row in paginated_houses.iterrows():
        st.image(row['image_path'], caption=f"{row['style']} - {row['price']:,} THB", use_column_width=True)
        st.write(f"🏠 {row['bedrooms']} Bedrooms | 🛁 {row['bathrooms']} Bathrooms | 📏 {row['area_size']} sq.m.")
        st.write(f"🛠 Facilities: {row['facilities']}")
        st.write("---")
